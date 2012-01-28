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
%prog [OPTIONS] <World # / Name / Path to World> <tiles dest dir>"""

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

def main():
    # bootstrap the logger with defaults
    configure_logger()

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    
    #avail_rendermodes = c_overviewer.get_render_modes()
    avail_north_dirs = ['lower-left', 'upper-left', 'upper-right', 'lower-right', 'auto']
   
    # revert to a vanilla OptionParser for right now 
    parser = OptionParser(usage=helptext)
    parser.add_option("-V", "--version", dest="version", help="Displays version information and then exits", action="store_true")
    parser.add_option("-p", "--processes", dest="procs", help="How many worker processes to start. Default %s" % cpus, default=cpus, action="store", type="int")
    #parser.add_option("-z", "--zoom", dest="zoom", help="Sets the zoom level manually instead of calculating it. This can be useful if you have outlier chunks that make your world too big. This value will make the highest zoom level contain (2**ZOOM)^2 tiles", action="store", type="int", advanced=True)
    #parser.add_option("--regionlist", dest="regionlist", help="A file containing, on each line, a path to a regionlist to update. Instead of scanning the world directory for regions, it will just use this list. Normal caching rules still apply.")
    #parser.add_option("--forcerender", dest="forcerender", help="Force re-rendering the entire map (or the given regionlist). Useful for re-rendering without deleting it.", action="store_true")
    #parser.add_option("--stochastic-render", dest="stochastic_render", help="Rerender a non-updated tile randomly, with the given probability (between 0 and 1). Useful for incrementally updating a map with a new mode.", type="float", advanced=True, default=0.0, metavar="PROBABILITY")
    #parser.add_option("--rendermodes", dest="rendermode", help="Specifies the render types, separated by ',', ':', or '/'. Use --list-rendermodes to list them all.", type="choice", required=True, default=avail_rendermodes[0], listify=True)
    #parser.add_option("--rendermode-options", dest="rendermode_options", default={}, advanced=True, help="Used to specify options for different rendermodes.  Only useful in a settings.py file")
    #parser.add_option("--custom-rendermodes", dest="custom_rendermodes", default={}, advanced=True, help="Used to define custom rendermodes.  Only useful in a settings.py file")
    #parser.add_option("--imgformat", dest="imgformat", help="The image output format to use. Currently supported: png(default), jpg.", advanced=True )
    #parser.add_option("--imgquality", dest="imgquality", default=95, help="Specify the quality of image output when using imgformat=\"jpg\".", type="int", advanced=True)
    #parser.add_option("--bg-color", dest="bg_color", help="Configures the background color for the GoogleMap output.  Specify in #RRGGBB format", advanced=True, type="string", default="#1A1A1A")
    #parser.add_option("--optimize-img", dest="optimizeimg", help="If using png, perform image file size optimizations on the output. Specify 1 for pngcrush, 2 for pngcrush+advdef and 3 for pngcrush-advdef with more aggressive settings. This may double (or more) render times, but will produce up to 30% smaller images. NOTE: requires corresponding programs in $PATH or %PATH%", advanced=True)
    #parser.add_option("--web-assets-hook", dest="web_assets_hook", help="If provided, run this function after the web assets have been copied, but before actual tile rendering begins. It should accept a MapGen object as its only argument.", action="store", metavar="FUNCTION", type="function", advanced=True)
    #parser.add_option("--web-assets-path", dest="web_assets_path", help="Specifies a non-standard web_assets directory to use. Files here will overwrite the default web assets.", metavar="PATH", type="string", advanced=True)
    #parser.add_option("--textures-path", dest="textures_path", help="Specifies a non-standard textures path, from which terrain.png and other textures are loaded.", metavar="PATH", type="string", advanced=True)
    parser.add_option("--check-terrain", dest="check_terrain", help="Prints the location and hash of terrain.png, useful for debugging terrain.png problems", action="store_true")
    parser.add_option("-q", "--quiet", dest="quiet", action="count", default=0, help="Print less output. You can specify this option multiple times.")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", default=0, help="Print more output. You can specify this option multiple times.")
    #parser.add_option("--skip-js", dest="skipjs", action="store_true", help="Don't output marker.js")
    #parser.add_option("--no-signs", dest="nosigns", action="store_true", help="Don't output signs to markers.js")
    #parser.add_option("--north-direction", dest="north_direction", action="store", help="Specifies which corner of the screen north will point to. Defaults to whatever the current map uses, or lower-left for new maps. Valid options are: " + ", ".join(avail_north_dirs) + ".", type="choice", default="auto", choices=avail_north_dirs)
    #parser.add_option("--changelist", dest="changelist", action="store", help="Output list of changed tiles to file. If the file exists, its contents will be overwritten.",advanced=True)
    #parser.add_option("--changelist-format", dest="changelist_format", action="store", help="Output relative or absolute paths for --changelist. Only valid when --changelist is used", type="choice", default="auto", choices=["auto", "relative","absolute"],advanced=True)
    parser.add_option("--display-config", dest="display_config", action="store_true", help="Display the configuration parameters, but don't render the map.  Requires all required options to be specified")
    #parser.add_option("--write-config", dest="write_config", action="store_true", help="Writes out a sample config file", commandLineOnly=True)

    options, args = parser.parse_args()

    # re-configure the logger now that we've processed the command line options
    configure_logger(logging.INFO + 10*options.quiet - 10*options.verbose,
            options.verbose > 0)

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

    if options.display_config:
        # just display the config file and exit
        parser.display_config()
        return 0
    
    # TODO remove advanced help?  needs discussion
    # TODO right now, we will not let users specify worlds to render on the command line.  
    # TODO in the future, we need to also let worlds be specified on the command line

    # if no arguments are provided, print out a helpful message
    if len(args) == 0:
        # first provide an appropriate error for bare-console users
        # that don't provide any options
        if util.is_bare_console():
            print "\n"
            print "The Overviewer is a console program.  Please open a Windows command prompt"
            print "first and run Overviewer from there.   Further documentation is available at"
            print "http://docs.overviewer.org/\n"
        else:
            # more helpful message for users who know what they're doing
            logging.error("You need to give me your world number or directory")
            parser.print_help()
            list_worlds()
        return 1
    
    # for multiworld, we must specify the *outputdir* on the command line
    elif len(args) == 1:
        logging.debug("Using %r as the output_directory", args[0])
        destdir = os.path.expanduser(args[0])
    elif len(args) == 2: # TODO support this usecase
        worlddir = os.path.expanduser(args[0])
        destdir = os.path.expanduser(args[1])

    if len(args) > 2:
        # it's possible the user has a space in one of their paths but didn't properly escape it
        # attempt to detect this case
        for start in range(len(args)):
            if not os.path.exists(args[start]):
                for end in range(start+1, len(args)+1):
                    if os.path.exists(" ".join(args[start:end])):
                        logging.warning("It looks like you meant to specify \"%s\" as your world dir or your output\n\
dir but you forgot to put quotes around the directory, since it contains spaces." % " ".join(args[start:end]))
                        return 1

    # TODO regionlists are per-world
    #if options.regionlist:
    #    regionlist = map(str.strip, open(options.regionlist, 'r'))
    #else:
    #    regionlist = None

    # TODO imgformat is per-world
    #if options.imgformat:
    #    if options.imgformat not in ('jpg','png'):
    #        parser.error("Unknown imgformat!")
    #    else:
    #        imgformat = options.imgformat
    #else:
    #    imgformat = 'png'

    # TODO optimzeimg is per-world
    #if options.optimizeimg:
    #    optimizeimg = int(options.optimizeimg)
    #    optimizeimages.check_programs(optimizeimg)
    #else:
    #    optimizeimg = None

    # TODO north_direction is per-world
    #if options.north_direction:
    #    north_direction = options.north_direction
    #else:
    #    north_direction = 'auto'

    # TODO reimplement changelists
    #if options.changelist:
    #    try:
    #        changefile = open(options.changelist,'w+')
    #    except IOError as e:
    #        logging.error("Unable to open file %s to use for changelist." % options.changelist)
    #        logging.error("I/O Error: %s" % e.strerror)
    #        return 1

    #if options.changelist_format != "auto" and not options.changelist:
    #    logging.error("changelist_format specified without changelist.")
    #    return 1
    #if options.changelist_format == "auto":
    #    options.changelist_format = "relative"

    logging.info("Welcome to Minecraft Overviewer!")
    logging.debug("Current log level: {0}".format(logging.getLogger().level))
       
    
    # make sure that the textures can be found
    try:
        #textures.generate(path=options.textures_path)
        tex = textures.Textures()
        tex.generate()
    except IOError, e:
        logging.error(str(e))
        return 1

    # look at our settings.py file
    mw_parser = configParser.MultiWorldParser("settings.py")
    mw_parser.parse()
    try:
        mw_parser.validate()
    except Exception:
        logging.error("Please investigate these errors in settings.py then try running Overviewer again")
        return 1


    # create our asset manager... ASSMAN
    assetMrg = assetmanager.AssetManager(destdir)

    render_things = mw_parser.get_render_things()
    tilesets = []

    # once we've made sure that everything validations, we can check to 
    # make sure the destdir exists
    if not os.path.exists(destdir):
        os.mkdir(destdir)

    # saves us from creating the same World object over and over again
    worldcache = {}

    for render_name in render_things:
        render = render_things[render_name]
        logging.debug("Found the following render thing: %r", render)

        if (render['worldname'] not in worldcache):
            w = world.World(render['worldname'])
            worldcache[render['worldname']] = w
        else:
            w = worldcache[render['worldname']]


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
