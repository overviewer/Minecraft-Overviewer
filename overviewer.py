#!/usr/bin/env python

#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

import platform
import sys

# quick version check
if not (sys.version_info[0] == 2 and sys.version_info[1] >= 6):
    print "Sorry, the Overviewer requires at least Python 2.6 to run"
    if sys.version_info[0] >= 3:
        print "and will not run on Python 3.0 or later"
    sys.exit(1)

import os
import os.path
import re
import subprocess
import multiprocessing
import time
import logging
from optparse import OptionParser

from overviewer_core import util
from overviewer_core import textures
from overviewer_core import optimizeimages, world
from overviewer_core import configParser, tileset, assetmanager, dispatcher

helptext = """
%prog [--rendermodes=...] [options] <World> <Output Dir>
%prog --config=<config file> [options]"""

def configure_logger(loglevel=logging.INFO, verbose=False):
    """Configures the root logger to our liking

    For a non-standard loglevel, pass in the level with which to configure the handler.

    For a more verbose options line, pass in verbose=True

    This function may be called more than once.

    """

    logger = logging.getLogger()

    outstream = sys.stderr

    if platform.system() == 'Windows':
        # Our custom output stream processor knows how to deal with select ANSI
        # color escape sequences
        outstream = util.WindowsOutputStream()
        formatter = util.ANSIColorFormatter(verbose)

    elif sys.stderr.isatty():
        # terminal logging with ANSI color
        formatter = util.ANSIColorFormatter(verbose)

    else:
        # Let's not assume anything. Just text.
        formatter = util.DumbFormatter(verbose)

    if hasattr(logger, 'overviewerHandler'):
        # we have already set up logging so just replace the formatter
        # this time with the new values
        logger.overviewerHandler.setFormatter(formatter)
        logger.setLevel(loglevel)

    else:
        # Save our handler here so we can tell which handler was ours if the
        # function is called again
        logger.overviewerHandler = logging.StreamHandler(outstream)
        logger.overviewerHandler.setFormatter(formatter)
        logger.addHandler(logger.overviewerHandler)
        logger.setLevel(loglevel)

def build_fake_settings(worldpath):
    """Builds and returns a renders dict as if it was parsed from a settings
    file and returned with get_render_things()

    This is used for the simple command line usage with no config file

    """
    from overviewer_core import settingsDefinition, rendermodes
    world = {}
    # Seed this render with all the defaults
    for defaultname, defaultinfo in settingsDefinition.render['values'].iteritems():
        if 'default' in defaultinfo:
            world[defaultname] = defaultinfo['default']
    # Set required items for the render. If any new required items without
    # defaults are added, this will need to be updated.
    worlds = {'world': worldpath}
    world['worldname'] = 'world'
    world['title'] = "Overviewer Render"
    world['rendermode'] = rendermodes.normal

    renders = {worldpath: world}

    # The following is mostly a copy/paste of the code in
    # MultiWorldParser.validate(). Someone make MultiWorldParser more
    # extensible to avoid this!
    origs = dict()
    for key in world:
        definition = settingsDefinition.render['values'][key]
        val = definition['validator'](world[key], world = worlds)
        if definition.get('save_orig', False):
            origs[key + "_orig"] = world[key]
        world[key] = val

    world['name'] = worldpath
    world.update(origs)
    renders[worldpath] = world
    return renders

def main():
    # bootstrap the logger with defaults
    configure_logger()

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    
    #avail_rendermodes = c_overviewer.get_render_modes()
    avail_north_dirs = ['lower-left', 'upper-left', 'upper-right', 'lower-right', 'auto']
   
    # Parse for basic options
    parser = OptionParser(usage=helptext)
    parser.add_option("--config", dest="config", action="store", help="Specify the config file to use.")
    
    # Useful one-time render modifiers:
    parser.add_option("--forcerender", dest="forcerender", action="store_true",
            help="Force re-rendering the entire map.")
    parser.add_option("--check-tiles", dest="checktiles", action="store_true",
            help="Check each tile on disk and re-render old tiles")
    parser.add_option("--no-tile-checks", dest="notilechecks", action="store_true",
            help="Only render tiles that come from chunks that have changed since the last render (the default)")

    # Useful one-time debugging options:
    parser.add_option("--check-terrain", dest="check_terrain", action="store_true",
            help="Prints the location and hash of terrain.png, useful for debugging terrain.png problems")
    parser.add_option("-V", "--version", dest="version",
            help="Displays version information and then exits", action="store_true")

    # Log level options:
    parser.add_option("-q", "--quiet", dest="quiet", action="count", default=0,
            help="Print less output. You can specify this option multiple times.")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", default=0,
            help="Print more output. You can specify this option multiple times.")

    options, args = parser.parse_args()

    # re-configure the logger now that we've processed the command line options
    configure_logger(logging.INFO + 10*options.quiet - 10*options.verbose,
            options.verbose > 0)

    ##########################################################################
    # This section of main() runs in response to any one-time options we have,
    # such as -V for version reporting
    if options.version:
        print "Minecraft Overviewer %s" % util.findGitVersion(),
        print "(%s)" % util.findGitHash()[:7]
        try:
            import overviewer_core.overviewer_version as overviewer_version
            print "built on %s" % overviewer_version.BUILD_DATE
            if options.verbose > 0:
                print "Build machine: %s %s" % (overviewer_version.BUILD_PLATFORM, overviewer_version.BUILD_OS)
        except ImportError:
            print "(build info not found)"
        return 0

    if options.check_terrain:
        import hashlib
        from overviewer_core.textures import Textures
        # TODO custom textures path?
        tex = Textures()

        try:
            f = tex.find_file("terrain.png", verbose=True)
        except IOError:
            logging.error("Could not find the file terrain.png")
            return 1

        h = hashlib.sha1()
        h.update(f.read())
        logging.info("Hash of terrain.png file is: `%s`", h.hexdigest())
        return 0

    # if no arguments are provided, print out a helpful message
    if len(args) == 0 and not options.config:
        # first provide an appropriate error for bare-console users
        # that don't provide any options
        if util.is_bare_console():
            print "\n"
            print "The Overviewer is a console program.  Please open a Windows command prompt"
            print "first and run Overviewer from there.   Further documentation is available at"
            print "http://docs.overviewer.org/\n"
        else:
            # more helpful message for users who know what they're doing
            logging.error("You must either specify --config or give me a world directory and output directory")
            parser.print_help()
            list_worlds()
        return 1
    
    ##########################################################################
    # This section does some sanity checking on the command line options passed
    # in. It checks to see if --config was given that no worldname/destdir were
    # given, and vice versa
    if options.config and args:
        print
        print "If you specify --config, you need to specify the world to render as well as"
        print "the destination in the config file, not on the command line."
        print "Put something like this in your config file:"
        print "worlds['myworld'] = %r" % args[0]
        print "outputdir = %r" % (args[1] if len(args) > 1 else "/path/to/output")
        print
        logging.error("Cannot specify both --config and worldname / output directory")
        parser.print_help()
        return 1
    
    if not options.config and len(args) < 2:
        logging.error("You must specify both the world directory and an output directory")
        parser.print_help()
        return 1
    if not options.config and len(args) > 2:
        # it's possible the user has a space in one of their paths but didn't
        # properly escape it attempt to detect this case
        for start in range(len(args)):
            if not os.path.exists(args[start]):
                for end in range(start+1, len(args)+1):
                    if os.path.exists(" ".join(args[start:end])):
                        logging.warning("It looks like you meant to specify \"%s\" as your world dir or your output\n\
dir but you forgot to put quotes around the directory, since it contains spaces." % " ".join(args[start:end]))
                        return 1
        logging.error("Too many command line arguments")
        parser.print_help()
        return 1

    #########################################################################
    # These two blocks of code unify config-file mode and command-line mode.
    # When the blocks have exited, they are expected to have set the following
    # vars:
    # destdir - the output directory
    # renders - the dict heirarchy 
    if not options.config:
        # No config file mode.
        worldpath, destdir = map(os.path.expanduser, args)
        logging.debug("Using %r as the world directory", worldpath)
        logging.debug("Using %r as the output directory", destdir)
        
        renders = build_fake_settings(worldpath)

    else:
        # Parse the config file
        mw_parser = configParser.MultiWorldParser(options.config)
        mw_parser.parse()
        try:
            mw_parser.validate()
        except Exception:
            logging.exception("Please investigate these errors in settings.py then try running Overviewer again")
            return 1

        try:
            destdir = mw_parser.outputdir
        except AttributeError:
            # Will get caught by the error check just below
            logging.debug("Attribute error while getting the outputdir from the config file. Will error in just a sec")
            destdir = ""
        else:
            logging.debug("outputdir from parser: %r", destdir)

        renders = mw_parser.get_render_things()

    ############################################################
    # Final validation and creation of the destination directory
    if not destdir:
        logging.error("You must specify the output directory in your config file.")
        logging.error("e.g. outputdir = '/path/to/outputdir'")
        return 1
    if not os.path.exists(destdir):
        try:
            os.mkdir(destdir)
        except OSError:
            logging.exception("Could not create the output directory.")
            return 1


    ########################################################################
    # Now we start the actual processing, now that all the configuration has
    # been gathered and validated
    logging.info("Welcome to Minecraft Overviewer!")
    logging.debug("Current log level: {0}".format(logging.getLogger().level))
       
    # create our asset manager... ASSMAN
    assetMrg = assetmanager.AssetManager(destdir)

    tilesets = []

    # saves us from creating the same World object over and over again
    worldcache = {}
    # same for textures
    texcache = {}

    for render_name, render in renders.iteritems():
        logging.debug("Found the following render thing: %r", render)

        # find or create the world object
        if (render['worldname'] not in worldcache):
            w = world.World(render['worldname'])
            worldcache[render['worldname']] = w
        else:
            w = worldcache[render['worldname']]
        
        # find or create the textures object
        texopts = util.dict_subset(render, ["texturepath", "bgcolor", "northdirection"])
        texopts_key = tuple(texopts.items())
        if texopts_key not in texcache:
            tex = textures.Textures(**texopts)
            tex.generate()
            texcache[texopts_key] = tex
        else:
            tex = texcache[texopts_key]

        rset = w.get_regionset(render['dimension'])
        if rset == None: # indicates no such dimension was found:
            logging.error("Sorry, you requested dimension '%s' for %s, but I couldn't find it", render['dimension'], render_name)
            return 1
        if (render['northdirection'] > 0):
            rset = rset.rotate(render['northdirection'])
        logging.debug("Using RegionSet %r", rset) 

        # create our TileSet from this RegionSet
        tileset_dir = os.path.abspath(os.path.join(destdir, render_name))
        print "tileset_dir: %r" % tileset_dir
        if not os.path.exists(tileset_dir):
            os.mkdir(tileset_dir)

        # only pass to the TileSet the options it really cares about
        tileSetOpts = util.dict_subset(render, ["name", "imgformat", "renderchecks", "rerenderprob", "bgcolor", "imgquality", "optimizeimg", "rendermode", "worldname_orig", "title", "dimension"])
        tset = tileset.TileSet(rset, assetMrg, tex, tileSetOpts, tileset_dir)
        tilesets.append(tset)

   
    # multiprocessing dispatcher
    dispatch = dispatcher.MultiprocessingDispatcher()
    def print_status(*args):
        logging.info("Status callback: %r", args)
    dispatch.render_all(tilesets, print_status)
    dispatch.close()

    assetMrg.finalize(tilesets)
    return 0

def list_worlds():
    "Prints out a brief summary of saves found in the default directory"
    print 
    worlds = world.get_worlds()
    if not worlds:
        print 'No world saves found in the usual place'
        return
    print "Detected saves:"

    # get max length of world name
    worldNameLen = max([len(str(x)) for x in worlds] + [len("World")])

    formatString = "%-" + str(worldNameLen) + "s | %-8s | %-8s | %-16s "
    print formatString % ("World", "Size", "Playtime", "Modified")
    print formatString % ("-"*worldNameLen, "-"*8, "-"*8, '-'*16)
    for name, info in sorted(worlds.iteritems()):
        if isinstance(name, basestring) and name.startswith("World") and len(name) == 6:
            try:
                world_n = int(name[-1])
                # we'll catch this one later, when it shows up as an
                # integer key
                continue
            except ValueError:
                pass
        timestamp = time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(info['LastPlayed'] / 1000))
        playtime = info['Time'] / 20
        playstamp = '%d:%02d' % (playtime / 3600, playtime / 60 % 60)
        size = "%.2fMB" % (info['SizeOnDisk'] / 1024. / 1024.)
        print formatString % (name, size, playstamp, timestamp)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        ret = main()
        util.exit(ret)
    except Exception, e:
        logging.exception("""An error has occurred. This may be a bug. Please let us know!
See http://docs.overviewer.org/en/latest/index.html#help

This is the error that occurred:""")
        util.exit(1)
