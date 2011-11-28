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

if not (sys.version_info[0] == 2 and sys.version_info[1] >= 6):
    print "Sorry, the Overviewer requires at least Python 2.6 to run"
    if sys.version_info[0] >= 3:
        print "and will not run on Python 3.0 or later"
    sys.exit(1)

isBareConsole = False

if platform.system() == 'Windows':
    try:
        import ctypes
        GetConsoleProcessList = ctypes.windll.kernel32.GetConsoleProcessList
        num = GetConsoleProcessList(ctypes.byref(ctypes.c_int(0)), ctypes.c_int(1))
        if (num == 1):
            isBareConsole = True

    except Exception:
        pass

import os
import os.path
import re
import subprocess
import multiprocessing
import time
import logging
from overviewer_core import util

def doExit(msg=None, code=1, wait=None, consoleMsg=True):
    '''Exits Overviewer.  If `wait` is None, the default
     will be true is 'isBareConsole' is true'''
    global isBareConsole
    if msg:
        print msg

    if wait == None:
        if isBareConsole:
            if consoleMsg:
                print "\n"
                print "The Overviewer is a console program.  Please open a Windows command prompt"
                print "first and run Overviewer from there.   Further documentation is available at"
                print "http://docs.overviewer.org/\n"
            print "Press [Enter] to close this window."
            raw_input()
    else:
        if wait:
            if consoleMsg:
                print "\n"
                print "The Overviewer is a console program.  Please open a Windows command prompt"
                print "first and run Overviewer from there.   Further documentation is available at"
                print "http://docs.overviewer.org/\n"
            print "Press [Enter] to close this window."
            raw_input()

    sys.exit(code) 


this_dir = util.get_program_path()

# make sure the c_overviewer extension is available
try:
    from overviewer_core import c_overviewer
except ImportError:
    ## if this is a frozen windows package, the following error messages about
    ## building the c_overviewer extension are not appropriate
    if hasattr(sys, "frozen"):
        print "Something has gone wrong importing the c_overviewer extension.  Please"
        print "make sure the 2008 and 2010 redistributable packages from Microsoft"
        print "are installed."
        doExit()


    ## try to find the build extension
    ext = os.path.join(this_dir, "overviewer_core", "c_overviewer.%s" % ("pyd" if platform.system() == "Windows" else "so"))
    if os.path.exists(ext):
        print "Something has gone wrong importing the c_overviewer extension.  Please"
        print "make sure it is up-to-date (clean and rebuild)"
        doExit()

    import traceback
    traceback.print_exc()

    print ""
    print "You need to compile the c_overviewer module to run Minecraft Overviewer."
    print "Run `python setup.py build`, or see the README for details."
    doExit()

from overviewer_core import textures

if hasattr(sys, "frozen"):
    pass # we don't bother with a compat test since it should always be in sync
elif "extension_version" in dir(c_overviewer):
    # check to make sure the binary matches the headers
    if os.path.exists(os.path.join(this_dir, "overviewer_core", "src", "overviewer.h")):
        with open(os.path.join(this_dir, "overviewer_core", "src", "overviewer.h")) as f:
            lines = f.readlines()
            lines = filter(lambda x: x.startswith("#define OVERVIEWER_EXTENSION_VERSION"), lines)
            if lines:
                l = lines[0]
                if int(l.split()[2].strip()) != c_overviewer.extension_version():
                    print "Please rebuild your c_overviewer module.  It is out of date!"
                    doExit(code=1, consoleMsg=True)
else:
    print "Please rebuild your c_overviewer module.  It is out of date!"
    doExit()

from overviewer_core.configParser import ConfigOptionParser
from overviewer_core import optimizeimages, world, quadtree
from overviewer_core import googlemap, rendernode

# definitions of built-in custom modes
# usually because what used to be a mode became an option
# for example, night mode
builtin_custom_rendermodes = {
    'night' : {
        'parent' : 'lighting',
        'label' : 'Night',
        'description' : 'like "lighting", except at night',
        'options' : {'night' : True}
    },

    'smooth-night' : {
        'parent' : 'smooth-lighting',
        'label' : 'Smooth Night',
        'description' : 'like "lighting", except smooth and at night',
        'options' : {'night' : True}
    },
}

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
    
    avail_rendermodes = c_overviewer.get_render_modes()
    avail_north_dirs = ['lower-left', 'upper-left', 'upper-right', 'lower-right', 'auto']
    
    parser = ConfigOptionParser(usage=helptext, config="settings.py")
    parser.add_option("-V", "--version", dest="version", helptext="Displays version information and then exits", action="store_true")
    parser.add_option("-p", "--processes", dest="procs", helptext="How many worker processes to start. Default %s" % cpus, default=cpus, action="store", type="int")
    parser.add_option("-z", "--zoom", dest="zoom", helptext="Sets the zoom level manually instead of calculating it. This can be useful if you have outlier chunks that make your world too big. This value will make the highest zoom level contain (2**ZOOM)^2 tiles", action="store", type="int", advanced=True)
    parser.add_option("--regionlist", dest="regionlist", helptext="A file containing, on each line, a path to a regionlist to update. Instead of scanning the world directory for regions, it will just use this list. Normal caching rules still apply.")
    parser.add_option("--forcerender", dest="forcerender", helptext="Force re-rendering the entire map (or the given regionlist). Useful for re-rendering without deleting it.", action="store_true")
    parser.add_option("--stochastic-render", dest="stochastic_render", helptext="Rerender a non-updated tile randomly, with the given probability (between 0 and 1). Useful for incrementally updating a map with a new mode.", type="float", advanced=True, default=0.0, metavar="PROBABILITY")
    parser.add_option("--rendermodes", dest="rendermode", helptext="Specifies the render types, separated by ',', ':', or '/'. Use --list-rendermodes to list them all.", type="choice", required=True, default=avail_rendermodes[0], listify=True)
    parser.add_option("--list-rendermodes", dest="list_rendermodes", action="store_true", helptext="List available render modes and exit.", commandLineOnly=True)
    parser.add_option("--rendermode-options", dest="rendermode_options", default={}, advanced=True, helptext="Used to specify options for different rendermodes.  Only useful in a settings.py file")
    parser.add_option("--custom-rendermodes", dest="custom_rendermodes", default={}, advanced=True, helptext="Used to define custom rendermodes.  Only useful in a settings.py file")
    parser.add_option("--imgformat", dest="imgformat", helptext="The image output format to use. Currently supported: png(default), jpg.", advanced=True )
    parser.add_option("--imgquality", dest="imgquality", default=95, helptext="Specify the quality of image output when using imgformat=\"jpg\".", type="int", advanced=True)
    parser.add_option("--bg-color", dest="bg_color", helptext="Configures the background color for the GoogleMap output.  Specify in #RRGGBB format", advanced=True, type="string", default="#1A1A1A")
    parser.add_option("--optimize-img", dest="optimizeimg", helptext="If using png, perform image file size optimizations on the output. Specify 1 for pngcrush, 2 for pngcrush+advdef and 3 for pngcrush-advdef with more aggressive settings. This may double (or more) render times, but will produce up to 30% smaller images. NOTE: requires corresponding programs in $PATH or %PATH%", advanced=True)
    parser.add_option("--web-assets-hook", dest="web_assets_hook", helptext="If provided, run this function after the web assets have been copied, but before actual tile rendering begins. It should accept a MapGen object as its only argument.", action="store", metavar="FUNCTION", type="function", advanced=True)
    parser.add_option("--web-assets-path", dest="web_assets_path", helptext="Specifies a non-standard web_assets directory to use. Files here will overwrite the default web assets.", metavar="PATH", type="string", advanced=True)
    parser.add_option("--textures-path", dest="textures_path", helptext="Specifies a non-standard textures path, from which terrain.png and other textures are loaded.", metavar="PATH", type="string", advanced=True)
    parser.add_option("--check-terrain", dest="check_terrain", helptext="Prints the location and hash of terrain.png, useful for debugging terrain.png problems", action="store_true", advanced=False, commandLineOnly=True)
    parser.add_option("-q", "--quiet", dest="quiet", action="count", default=0, helptext="Print less output. You can specify this option multiple times.")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", default=0, helptext="Print more output. You can specify this option multiple times.")
    parser.add_option("--skip-js", dest="skipjs", action="store_true", helptext="Don't output marker.js or regions.js")
    parser.add_option("--no-signs", dest="nosigns", action="store_true", helptext="Don't output signs to markers.js")
    parser.add_option("--north-direction", dest="north_direction", action="store", helptext="Specifies which corner of the screen north will point to. Defaults to whatever the current map uses, or lower-left for new maps. Valid options are: " + ", ".join(avail_north_dirs) + ".", type="choice", default="auto", choices=avail_north_dirs)
    parser.add_option("--changelist", dest="changelist", action="store", helptext="Output list of changed tiles to file. If the file exists, its contents will be overwritten.",advanced=True)
    parser.add_option("--changelist-format", dest="changelist_format", action="store", helptext="Output relative or absolute paths for --changelist. Only valid when --changelist is used", type="choice", default="auto", choices=["auto", "relative","absolute"],advanced=True)
    parser.add_option("--display-config", dest="display_config", action="store_true", helptext="Display the configuration parameters, but don't render the map.  Requires all required options to be specified", commandLineOnly=True)
    #parser.add_option("--write-config", dest="write_config", action="store_true", helptext="Writes out a sample config file", commandLineOnly=True)

    options, args = parser.parse_args()

    # re-configure the logger now that we've processed the command line options
    configure_logger(logging.INFO + 10*options.quiet - 10*options.verbose,
            options.verbose > 0)

    if options.version:
        try:
            import overviewer_core.overviewer_version as overviewer_version
            print "Minecraft-Overviewer %s" % overviewer_version.VERSION
            print "Git commit: %s" % overviewer_version.HASH
            print "built on %s" % overviewer_version.BUILD_DATE
            print "Build machine: %s %s" % (overviewer_version.BUILD_PLATFORM, overviewer_version.BUILD_OS)
        except Exception:
            print "version info not found"
            pass
        doExit(code=0, consoleMsg=False)

    # setup c_overviewer rendermode customs / options
    for mode in builtin_custom_rendermodes:
        c_overviewer.add_custom_render_mode(mode, builtin_custom_rendermodes[mode])
    for mode in options.custom_rendermodes:
        c_overviewer.add_custom_render_mode(mode, options.custom_rendermodes[mode])
    for mode in options.rendermode_options:
        c_overviewer.set_render_mode_options(mode, options.rendermode_options[mode])
    
    
    # Expand user dir in directories strings
    if options.textures_path:
        options.textures_path = os.path.expanduser(options.textures_path)
    if options.web_assets_path:
        options.web_assets_path = os.path.expanduser(options.web_assets_path)
        
    
    if options.list_rendermodes:
        list_rendermodes()
        doExit(code=0, consoleMsg=False)

    if options.check_terrain:
        import hashlib
        from overviewer_core.textures import _find_file
        if options.textures_path:
            textures._find_file_local_path = options.textures_path

        try:
            f = _find_file("terrain.png", verbose=True)
        except IOError:
            logging.error("Could not find the file terrain.png")
            doExit(code=1, consoleMsg=False)

        h = hashlib.sha1()
        h.update(f.read())
        logging.info("Hash of terrain.png file is: `%s`", h.hexdigest())
        doExit(code=0, consoleMsg=False)
        
    if options.advanced_help:
        parser.advanced_help()
        doExit(code=0, consoleMsg=False)

    if len(args) < 1:
        logging.error("You need to give me your world number or directory")
        parser.print_help()
        list_worlds()
        doExit(code=1, consoleMsg=True)
    worlddir = os.path.expanduser(args[0])

    if len(args) > 2:
        # it's possible the user has a space in one of their paths but didn't properly escape it
        # attempt to detect this case
        for start in range(len(args)):
            if not os.path.exists(args[start]):
                for end in range(start+1, len(args)+1):
                    if os.path.exists(" ".join(args[start:end])):
                        logging.warning("It looks like you meant to specify \"%s\" as your world dir or your output\n\
dir but you forgot to put quotes around the directory, since it contains spaces." % " ".join(args[start:end]))
                        doExit(code=1, consoleMsg=False)

    if not os.path.exists(worlddir):
        # world given is either world number, or name
        worlds = world.get_worlds()
        
        # if there are no worlds found at all, exit now
        if not worlds:
            parser.print_help()
            logging.error("Invalid world path")
            doExit(code=1, consoleMsg=False)
        
        try:
            worldnum = int(worlddir)
            worlddir = worlds[worldnum]['path']
        except ValueError:
            # it wasn't a number or path, try using it as a name
            try:
                worlddir = worlds[worlddir]['path']
            except KeyError:
                # it's not a number, name, or path
                parser.print_help()
                logging.error("Invalid world name or path")
                doExit(code=1, consoleMsg=False)
        except KeyError:
            # it was an invalid number
            parser.print_help()
            logging.error("Invalid world number")
            doExit(code=1, consoleMsg=False)
    
    # final sanity check for worlddir
    if not os.path.exists(os.path.join(worlddir, 'level.dat')):
        logging.error("Invalid world path -- does not contain level.dat")
        doExit(code=1, consoleMsg=False)

    if len(args) < 2:
        logging.error("Where do you want to save the tiles?")
        doExit(code=1, consoleMsg=False)
    elif len(args) > 2:
        parser.print_help()
        logging.error("Sorry, you specified too many arguments")
        doExit(code=1, consoleMsg=False)


    destdir = os.path.expanduser(args[1])
    if options.display_config:
        # just display the config file and exit
        parser.display_config()
        doExit(code=0, consoleMsg=False)


    if options.regionlist:
        regionlist = map(str.strip, open(options.regionlist, 'r'))
    else:
        regionlist = None

    if options.imgformat:
        if options.imgformat not in ('jpg','png'):
            parser.error("Unknown imgformat!")
        else:
            imgformat = options.imgformat
    else:
        imgformat = 'png'

    if options.optimizeimg:
        optimizeimg = int(options.optimizeimg)
        optimizeimages.check_programs(optimizeimg)
    else:
        optimizeimg = None

    if options.north_direction:
        north_direction = options.north_direction
    else:
        north_direction = 'auto'

    if options.changelist:
        try:
            changefile = open(options.changelist,'w+')
        except IOError as e:
            logging.error("Unable to open file %s to use for changelist." % options.changelist)
            logging.error("I/O Error: %s" % e.strerror)
            doExit(code=1, consoleMsg=False)

    if options.changelist_format != "auto" and not options.changelist:
        logging.error("changelist_format specified without changelist.")
        doExit(code=1, consoleMsg=False)
    if options.changelist_format == "auto":
        options.changelist_format = "relative"

    logging.info("Welcome to Minecraft Overviewer!")
    logging.debug("Current log level: {0}".format(logging.getLogger().level))
       
    useBiomeData = os.path.exists(os.path.join(worlddir, 'biomes'))
    if not useBiomeData:
        logging.info("Notice: Not using biome data for tinting")
    
    # make sure that the textures can be found
    try:
        textures.generate(path=options.textures_path)
    except IOError, e:
        logging.error(str(e))
        doExit(code=1, consoleMsg=False)
    
    # First do world-level preprocessing. This scans the world hierarchy, reads
    # in the region files and caches chunk modified times, and determines the
    # chunk bounds (max and min in both dimensions)
    w = world.World(worlddir, destdir, useBiomeData=useBiomeData, regionlist=regionlist, north_direction=north_direction)
    if north_direction == 'auto':
        north_direction = w.persistentData['north_direction']
        options.north_direction = north_direction
    elif (w.persistentData['north_direction'] != north_direction and
            not options.forcerender and
            not w.persistentDataIsNew
            ):
        logging.error("Conflicting north-direction setting!")
        logging.error("Overviewer.dat gives previous north-direction as "+w.persistentData['north_direction'])
        logging.error("Requested north-direction was "+north_direction)
        logging.error("To change north-direction of an existing render, use --forcerender")
        doExit(code=1, consoleMsg=False)
    
    # A couple other things we need to figure out about the world:
    w.determine_bounds()
    w.find_true_spawn()

    logging.info("Rendering the following tilesets: %s", ",".join(options.rendermode))

    bgcolor = (int(options.bg_color[1:3],16),
               int(options.bg_color[3:5],16),
               int(options.bg_color[5:7],16),
               0)

    # Create the quadtrees. There is one quadtree per rendermode requested, and
    # therefore, per output directory hierarchy of tiles. Each quadtree
    # individually computes its depth and size. The constructor computes the
    # depth of the tree, while the go() method re-arranges tiles if the current
    # depth differs from the computed depth.
    q = []
    qtree_args = {'depth' : options.zoom,
                  'imgformat' : imgformat,
                  'imgquality' : options.imgquality,
                  'optimizeimg' : optimizeimg,
                  'bgcolor' : bgcolor,
                  'forcerender' : options.forcerender,
                  'rerender_prob' : options.stochastic_render
                  }
    for rendermode in options.rendermode:
        if rendermode == 'normal':
            qtree = quadtree.QuadtreeGen(w, destdir, rendermode=rendermode, tiledir='tiles', **qtree_args)
        else:
            qtree = quadtree.QuadtreeGen(w, destdir, rendermode=rendermode, **qtree_args)
        q.append(qtree)
    
    # Make sure the quadtrees are the correct depth
    for qtree in q:
        qtree.check_depth()

    # create the distributed render
    r = rendernode.RenderNode(q, options)
    # for the pool_initializer
    r.builtin_custom_rendermodes = builtin_custom_rendermodes
    
    # write out the map and web assets
    m = googlemap.MapGen(q, configInfo=options)
    m.go(options.procs)
    
    # render the tiles!
    r.go(options.procs)

    # finish up the map
    m.finalize()

    if options.changelist:
        changed=[]
        for tile in r.rendered_tiles:
            if options.changelist_format=="absolute":
                tile=os.path.abspath(tile)
            changed.append(tile)
            for zl in range(q[0].p - 1):
                tile=os.path.dirname(tile)
                changed.append("%s.%s" % (tile, imgformat))
        #Quick and nasty way to remove duplicate entries
        changed=list(set(changed))
        changed.sort()
        for path in changed:
            changefile.write("%s\n" % path)
        changefile.close()

def list_rendermodes():
    "Prints out a pretty list of supported rendermodes"
    
    def print_mode_tree(line_max, mode, prefix='', last=False):
        "Prints out a mode tree for the given mode, with an indent."
        
        try:
            info = c_overviewer.get_render_mode_info(mode)
        except ValueError:
            info = {}
        
        print prefix + '+-',  mode,
        
        if 'description' in info:
            print " " * (line_max - len(prefix) - len(mode) - 2),
            print info['description']
        else:
            print
        
        children = c_overviewer.get_render_mode_children(mode)
        for child in children:
            child_last = (child == children[-1])
            if last:
                child_prefix = '  '
            else:
                child_prefix = '| '
            print_mode_tree(line_max, child, prefix=prefix + child_prefix, last=child_last)
    
    avail_rendermodes = c_overviewer.get_render_modes()
    line_lengths = {}
    parent_modes = []
    for mode in avail_rendermodes:
        inherit = c_overviewer.get_render_mode_inheritance(mode)
        if not inherit[0] in parent_modes:
            parent_modes.append(inherit[0])
        line_lengths[mode] = 2 * len(inherit) + 1 + len(mode)
    
    line_length = max(line_lengths.values())
    for mode in parent_modes:
        print_mode_tree(line_length, mode, last=(mode == parent_modes[-1]))

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
        main()
    except Exception, e:
        if e.message == "Exiting":
            logging.info("Exiting...")
            doExit(code=0, wait=False)
        logging.exception("""An error has occurred. This may be a bug. Please let us know!
See http://docs.overviewer.org/en/latest/index.html#help

This is the error that occurred:""")
        doExit(code=1, consoleMsg=False)
