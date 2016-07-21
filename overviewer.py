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
    print("Sorry, the Overviewer requires at least Python 2.6 to run")
    if sys.version_info[0] >= 3:
        print("and will not run on Python 3.0 or later")
    sys.exit(1)

import os
import os.path
import re
import subprocess
import multiprocessing
import time
import logging
from optparse import OptionParser, OptionGroup

from overviewer_core import util
from overviewer_core import logger
from overviewer_core import textures
from overviewer_core import optimizeimages, world
from overviewer_core import configParser, tileset, assetmanager, dispatcher
from overviewer_core import cache
from overviewer_core import observer

helptext = """
%prog [--rendermodes=...] [options] <World> <Output Dir>
%prog --config=<config file> [options]"""

def main():
    # bootstrap the logger with defaults
    logger.configure()

    if os.name == "posix":
        if os.geteuid() == 0:
            logging.warning("You are running Overviewer as root. "
                            "It is recommended that you never do this, "
                            "as it is dangerous for your system. If you are running "
                            "into permission errors, fix your file/directory "
                            "permissions instead. Overviewer does not need access to "
                            "critical system resources and therefore does not require "
                            "root access.")

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1

    #avail_rendermodes = c_overviewer.get_render_modes()
    avail_north_dirs = ['lower-left', 'upper-left', 'upper-right', 'lower-right', 'auto']

    # Parse for basic options
    parser = OptionParser(usage=helptext, add_help_option=False)
    parser.add_option("-h", "--help", dest="help", action="store_true",
            help="show this help message and exit")
    parser.add_option("-c", "--config", dest="config", action="store", help="Specify the config file to use.")
    parser.add_option("-p", "--processes", dest="procs", action="store", type="int",
            help="The number of local worker processes to spawn. Defaults to the number of CPU cores your computer has")

    parser.add_option("--pid", dest="pid", action="store", help="Specify the pid file to use.")
    # Options that only apply to the config-less render usage
    parser.add_option("--rendermodes", dest="rendermodes", action="store",
            help="If you're not using a config file, specify which rendermodes to render with this option. This is a comma-separated list.")

    # Useful one-time render modifiers:
    parser.add_option("--forcerender", dest="forcerender", action="store_true",
            help="Force re-rendering the entire map.")
    parser.add_option("--check-tiles", dest="checktiles", action="store_true",
            help="Check each tile on disk and re-render old tiles")
    parser.add_option("--no-tile-checks", dest="notilechecks", action="store_true",
            help="Only render tiles that come from chunks that have changed since the last render (the default)")

    # Useful one-time debugging options:
    parser.add_option("--check-terrain", dest="check_terrain", action="store_true",
            help="Tries to locate the texture files. Useful for debugging texture problems.")
    parser.add_option("-V", "--version", dest="version",
            help="Displays version information and then exits", action="store_true")
    parser.add_option("--check-version", dest="checkversion",
            help="Fetchs information about the latest version of Overviewer", action="store_true")
    parser.add_option("--update-web-assets", dest='update_web_assets', action="store_true",
            help="Update web assets. Will *not* render tiles or update overviewerConfig.js")

    # Log level options:
    parser.add_option("-q", "--quiet", dest="quiet", action="count", default=0,
            help="Print less output. You can specify this option multiple times.")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", default=0,
            help="Print more output. You can specify this option multiple times.")
    parser.add_option("--simple-output", dest="simple", action="store_true", default=False,
            help="Use a simple output format, with no colors or progress bars")

    # create a group for "plugin exes" (the concept of a plugin exe is only loosly defined at this point)
    exegroup = OptionGroup(parser, "Other Scripts",
            "These scripts may accept different arguments than the ones listed above")
    exegroup.add_option("--genpoi", dest="genpoi", action="store_true",
            help="Runs the genPOI script")
    exegroup.add_option("--skip-scan", dest="skipscan", action="store_true",
            help="When running GenPOI, don't scan for entities")
    exegroup.add_option("--skip-players", dest="skipplayers", action="store_true",
            help="When running GenPOI, don't get player data")

    parser.add_option_group(exegroup)

    options, args = parser.parse_args()

    # first thing to do is check for stuff in the exegroup:
    if options.genpoi:
        # remove the "--genpoi" option from sys.argv before running genPI
        sys.argv.remove("--genpoi")
        #sys.path.append(".")
        g = __import__("overviewer_core.aux_files", {}, {}, ["genPOI"])
        g.genPOI.main()
        return 0
    if options.help:
        parser.print_help()
        return 0

    # re-configure the logger now that we've processed the command line options
    logger.configure(logging.INFO + 10*options.quiet - 10*options.verbose,
                     verbose=options.verbose > 0,
                     simple=options.simple)

    ##########################################################################
    # This section of main() runs in response to any one-time options we have,
    # such as -V for version reporting
    if options.version:
        print("Minecraft Overviewer %s" % util.findGitVersion()),
        print("(%s)" % util.findGitHash()[:7])
        try:
            import overviewer_core.overviewer_version as overviewer_version
            print("built on %s" % overviewer_version.BUILD_DATE)
            if options.verbose > 0:
                print("Build machine: %s %s" % (overviewer_version.BUILD_PLATFORM, overviewer_version.BUILD_OS))
                print("Read version information from %r"% overviewer_version.__file__)
        except ImportError:
            print("(build info not found)")
        if options.verbose > 0:
            print("Python executable: %r" % sys.executable)
            print(sys.version)
        if not options.checkversion:
            return 0
    if options.checkversion:
        print("Currently running Minecraft Overviewer %s" % util.findGitVersion()),
        print("(%s)" % util.findGitHash()[:7])
        try:
            import urllib
            import json
            latest_ver = json.loads(urllib.urlopen("http://overviewer.org/download.json").read())['src']
            print("Latest version of Minecraft Overviewer %s (%s)" % (latest_ver['version'], latest_ver['commit'][:7]))
            print("See http://overviewer.org/downloads for more information")
        except Exception:
            print("Failed to fetch latest version info.")
            if options.verbose > 0:
                import traceback
                traceback.print_exc()
            else:
                print("Re-run with --verbose for more details")
            return 1
        return 0


    if options.pid:
        if os.path.exists(options.pid):
            try:
                with open(options.pid, 'r') as fpid:
                    pid = int(fpid.read())
                    if util.pid_exists(pid):
                        print("Already running (pid exists) - exiting..")
                        return 0
            except (IOError, ValueError):
                pass
        with open(options.pid,"w") as f:
            f.write(str(os.getpid()))
    # if --check-terrain was specified, but we have NO config file, then we cannot
    # operate on a custom texture path.  we do terrain checking with a custom texture
    # pack later on, after we've parsed the config file
    if options.check_terrain and not options.config:
        import hashlib
        from overviewer_core.textures import Textures
        tex = Textures()

        logging.info("Looking for a few common texture files...")
        try:
            f = tex.find_file("assets/minecraft/textures/blocks/sandstone_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/grass_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/diamond_ore.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/planks_acacia.png", verbose=True)
        except IOError:
            logging.error("Could not find any texture files.")
            return 1

        return 0

    # if no arguments are provided, print out a helpful message
    if len(args) == 0 and not options.config:
        # first provide an appropriate error for bare-console users
        # that don't provide any options
        if util.is_bare_console():
            print("\n")
            print("The Overviewer is a console program.  Please open a Windows command prompt")
            print("first and run Overviewer from there.   Further documentation is available at")
            print("http://docs.overviewer.org/\n")
            print("\n")
            print("For a quick-start guide on Windows, visit the following URL:\n")
            print("http://docs.overviewer.org/en/latest/win_tut/windowsguide/\n")

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
        print()
        print("If you specify --config, you need to specify the world to render as well as")
        print("the destination in the config file, not on the command line.")
        print("Put something like this in your config file:")
        print("worlds['myworld'] = %r" % args[0])
        print("outputdir = %r" % (args[1] if len(args) > 1 else "/path/to/output"))
        print()
        logging.error("Cannot specify both --config AND a world + output directory on the command line.")
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
    # These two halfs of this if statement unify config-file mode and
    # command-line mode.
    mw_parser = configParser.MultiWorldParser()

    if not options.config:
        # No config file mode.
        worldpath, destdir = map(os.path.expanduser, args)
        logging.debug("Using %r as the world directory", worldpath)
        logging.debug("Using %r as the output directory", destdir)

        mw_parser.set_config_item("worlds", {'world': worldpath})
        mw_parser.set_config_item("outputdir", destdir)

        rendermodes = ['lighting']
        if options.rendermodes:
            rendermodes = options.rendermodes.replace("-","_").split(",")

        # Now for some good defaults
        renders = util.OrderedDict()
        for rm in rendermodes:
            renders["world-" + rm] = {
                    "world": "world",
                    "title": "Overviewer Render (%s)" % rm,
                    "rendermode": rm,
                    }
        mw_parser.set_config_item("renders", renders)

    else:
        if options.rendermodes:
            logging.error("You cannot specify --rendermodes if you give a config file. Configure your rendermodes in the config file instead")
            parser.print_help()
            return 1

        # Parse the config file
        try:
            mw_parser.parse(os.path.expanduser(options.config))
        except configParser.MissingConfigException as e:
            # this isn't a "bug", so don't print scary traceback
            logging.error(str(e))
            util.nice_exit(1)

    # Add in the command options here, perhaps overriding values specified in
    # the config
    if options.procs:
        mw_parser.set_config_item("processes", options.procs)

    # Now parse and return the validated config
    try:
        config = mw_parser.get_validated_config()
    except Exception as ex:
        if options.verbose:
            logging.exception("An error was encountered with your configuration. See the info below.")
        else: # no need to print scary traceback! just
            logging.error("An error was encountered with your configuration.")
            logging.error(str(ex))
        return 1

    if options.check_terrain: # we are already in the "if configfile" branch
        logging.info("Looking for a few common texture files...")
        for render_name, render in config['renders'].iteritems():
            logging.info("Looking at render %r", render_name)

            # find or create the textures object
            texopts = util.dict_subset(render, ["texturepath"])

            tex = textures.Textures(**texopts)
            f = tex.find_file("assets/minecraft/textures/blocks/sandstone_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/grass_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/diamond_ore.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/blocks/planks_oak.png", verbose=True)
        return 0

    ############################################################
    # Final validation steps and creation of the destination directory
    logging.info("Welcome to Minecraft Overviewer!")
    logging.debug("Current log level: {0}".format(logging.getLogger().level))

    # Override some render configdict options depending on one-time command line
    # modifiers
    if (
            bool(options.forcerender) +
            bool(options.checktiles) +
            bool(options.notilechecks)
            ) > 1:
        logging.error("You cannot specify more than one of --forcerender, "+
        "--check-tiles, and --no-tile-checks. These options conflict.")
        parser.print_help()
        return 1

    def set_renderchecks(checkname, num):
        for name, render in config['renders'].iteritems():
            if render.get('renderchecks', 0) == 3:
                logging.warning(checkname + " ignoring render " + repr(name) + " since it's marked as \"don't render\".")
            else:
                render['renderchecks'] = num
        
    if options.forcerender:
        logging.info("Forcerender mode activated. ALL tiles will be rendered")
        set_renderchecks("forcerender", 2)
    elif options.checktiles:
        logging.info("Checking all tiles for updates manually.")
        set_renderchecks("checktiles", 1)
    elif options.notilechecks:
        logging.info("Disabling all tile mtime checks. Only rendering tiles "+
        "that need updating since last render")
        set_renderchecks("notilechecks", 0)

    if not config['renders']:
        logging.error("You must specify at least one render in your config file. See the docs if you're having trouble")
        return 1

    #####################
    # Do a few last minute things to each render dictionary here
    for rname, render in config['renders'].iteritems():
        # Convert render['world'] to the world path, and store the original
        # in render['worldname_orig']
        try:
            worldpath = config['worlds'][render['world']]
        except KeyError:
            logging.error("Render %s's world is '%s', but I could not find a corresponding entry in the worlds dictionary.",
                    rname, render['world'])
            return 1
        render['worldname_orig'] = render['world']
        render['world'] = worldpath

        # If 'forcerender' is set, change renderchecks to 2
        if render.get('forcerender', False):
            render['renderchecks'] = 2

        # check if overlays are set, if so, make sure that those renders exist
        if render.get('overlay', []) != []:
            for x in render.get('overlay'):
                if x != rname:
                    try:
                        renderLink = config['renders'][x]
                    except KeyError:
                        logging.error("Render %s's overlay is '%s', but I could not find a corresponding entry in the renders dictionary.",
                                rname, x)
                        return 1
                else:
                    logging.error("Render %s's overlay contains itself.", rname)
                    return 1

    destdir = config['outputdir']
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
    # create our asset manager... ASSMAN
    assetMrg = assetmanager.AssetManager(destdir, config.get('customwebassets', None), config.get('google_api_key', "insert-your-api-key"))

    # If we've been asked to update web assets, do that and then exit 
    if options.update_web_assets:
        assetMrg.output_noconfig()
        logging.info("Web assets have been updated")
        return 0

    # The changelist support.
    changelists = {}
    for render in config['renders'].itervalues():
        if 'changelist' in render:
            path = render['changelist']
            if path not in changelists:
                out = open(path, "w")
                logging.debug("Opening changelist %s (%s)", out, out.fileno())
                changelists[path] = out
            else:
                out = changelists[path]
            render['changelist'] = out.fileno()

    tilesets = []

    # saves us from creating the same World object over and over again
    worldcache = {}
    # same for textures
    texcache = {}

    # Set up the cache objects to use
    caches = []
    caches.append(cache.LRUCache(size=100))
    if config.get("memcached_host", False):
        caches.append(cache.Memcached(config['memcached_host']))
    # TODO: optionally more caching layers here

    renders = config['renders']
    for render_name, render in renders.iteritems():
        logging.debug("Found the following render thing: %r", render)

        # find or create the world object
        try:
            w = worldcache[render['world']]
        except KeyError:
            w = world.World(render['world'])
            worldcache[render['world']] = w

        # find or create the textures object
        texopts = util.dict_subset(render, ["texturepath", "bgcolor", "northdirection"])
        texopts_key = tuple(texopts.items())
        if texopts_key not in texcache:
            tex = textures.Textures(**texopts)
            logging.info("Generating textures...")
            tex.generate()
            logging.debug("Finished generating textures")
            texcache[texopts_key] = tex
        else:
            tex = texcache[texopts_key]
    
        try:
            logging.debug("Asking for regionset %r" % render['dimension'][1])
            rset = w.get_regionset(render['dimension'][1])
        except IndexError:
            logging.error("Sorry, I can't find anything to render!  Are you sure there are .mca files in the world directory?")
            return 1
        if rset == None: # indicates no such dimension was found:
            logging.warn("Sorry, you requested dimension '%s' for %s, but I couldn't find it", render['dimension'][0], render_name)
            continue

        #################
        # Apply any regionset transformations here

        # Insert a layer of caching above the real regionset. Any world
        # tranformations will pull from this cache, but their results will not
        # be cached by this layer. This uses a common pool of caches; each
        # regionset cache pulls from the same underlying cache object.
        rset = world.CachedRegionSet(rset, caches)

        # If a crop is requested, wrap the regionset here
        if "crop" in render:
            rsets = []
            for zone in render['crop']:
                rsets.append(world.CroppedRegionSet(rset, *zone))
        else:
            rsets = [rset]

        # If this is to be a rotated regionset, wrap it in a RotatedRegionSet
        # object
        if (render['northdirection'] > 0):
            newrsets = []
            for r in rsets:
                r = world.RotatedRegionSet(r, render['northdirection'])
                newrsets.append(r)
            rsets = newrsets

        ###############################
        # Do the final prep and create the TileSet object

        # create our TileSet from this RegionSet
        tileset_dir = os.path.abspath(os.path.join(destdir, render_name))

        # only pass to the TileSet the options it really cares about
        render['name'] = render_name # perhaps a hack. This is stored here for the asset manager
        tileSetOpts = util.dict_subset(render, ["name", "imgformat", "renderchecks", "rerenderprob", "bgcolor", "defaultzoom", "imgquality", "optimizeimg", "rendermode", "worldname_orig", "title", "dimension", "changelist", "showspawn", "overlay", "base", "poititle", "maxzoom", "showlocationmarker", "minzoom"])
        tileSetOpts.update({"spawn": w.find_true_spawn()}) # TODO find a better way to do this
        for rset in rsets:
            tset = tileset.TileSet(w, rset, assetMrg, tex, tileSetOpts, tileset_dir)
            tilesets.append(tset)

    # If none of the requested dimenstions exist, tilesets will be empty
    if not tilesets:
        logging.error("There are no tilesets to render!  There's nothing to do, so exiting.")
        return 1


    # Do tileset preprocessing here, before we start dispatching jobs
    logging.info("Preprocessing...")
    for ts in tilesets:
        ts.do_preprocessing()

    # Output initial static data and configuration
    assetMrg.initialize(tilesets)

    # multiprocessing dispatcher
    if config['processes'] == 1:
        dispatch = dispatcher.Dispatcher()
    else:
        dispatch = dispatcher.MultiprocessingDispatcher(
            local_procs=config['processes'])
    dispatch.render_all(tilesets, config['observer'])
    dispatch.close()

    assetMrg.finalize(tilesets)

    for out in changelists.itervalues():
        logging.debug("Closing %s (%s)", out, out.fileno())
        out.close()

    if config['processes'] == 1:
        logging.debug("Final cache stats:")
        for c in caches:
            logging.debug("\t%s: %s hits, %s misses", c.__class__.__name__, c.hits, c.misses)
    if options.pid:
        os.remove(options.pid)

    logging.info("Your render has been written to '%s', open index.html to view it" % destdir)    
        
    return 0

def list_worlds():
    "Prints out a brief summary of saves found in the default directory"
    print
    worlds = world.get_worlds()
    if not worlds:
        print('No world saves found in the usual place')
        return
    print("Detected saves:")

    # get max length of world name
    worldNameLen = max([len(x) for x in worlds] + [len("World")])

    formatString = "%-" + str(worldNameLen) + "s | %-8s | %-16s | %s "
    print(formatString % ("World", "Playtime", "Modified", "Path"))
    print(formatString % ("-"*worldNameLen, "-"*8, '-'*16, '-'*4))
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
        path = info['path']
        print(formatString % (name, playstamp, timestamp, path))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        ret = main()
        util.nice_exit(ret)
    except textures.TextureException as e:
        # this isn't a "bug", so don't print scary traceback
        logging.error(str(e))
        util.nice_exit(1)
    except Exception as e:
        logging.exception("""An error has occurred. This may be a bug. Please let us know!
See http://docs.overviewer.org/en/latest/index.html#help

This is the error that occurred:""")
        util.nice_exit(1)
