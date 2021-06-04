#!/usr/bin/env python3

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

from __future__ import print_function

import platform
import sys

# quick version check
if sys.version_info[0] == 2 or (sys.version_info[0] == 3 and sys.version_info[1] < 4):
    print("Sorry, the Overviewer requires at least Python 3.4 to run.")
    sys.exit(1)

import os
import os.path
import re
import subprocess
import multiprocessing
import time
import logging
from argparse import ArgumentParser
from collections import OrderedDict

from overviewer_core import util
from overviewer_core import logger
from overviewer_core import textures
from overviewer_core import optimizeimages, world
from overviewer_core import config_parser, tileset, assetmanager, dispatcher
from overviewer_core import cache
from overviewer_core import observer
from overviewer_core.nbt import CorruptNBTError

helptext = """
%(prog)s [--rendermodes=...] [options] <World> <Output Dir>
%(prog)s --config=<config file> [options]"""


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
            with open("/etc/redhat-release", "r") as release_f:
                rel_contents = release_f.read()
                try:
                    major_rel = re.search(r'\d(\.\d+)?', rel_contents).group(0).split('.')[0]
                    if major_rel == "6":
                        logging.warning(
                            "We will be dropping support for this release of your distribution "
                            "soon. Please upgrade as soon as possible, or you will not receive "
                            "future Overviewer updates.")
                except AttributeError:
                    pass
        except IOError:
            pass

    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1

    avail_north_dirs = ['lower-left', 'upper-left', 'upper-right', 'lower-right', 'auto']

    # Parse for basic options
    parser = ArgumentParser(usage=helptext)
    parser.add_argument("-c", "--config", dest="config", action="store",
                        help="Specify the config file to use.")
    parser.add_argument("-p", "--processes", dest="procs", action="store", type=int,
                        help="The number of local worker processes to spawn. Defaults to the "
                        "number of CPU cores your computer has.")

    parser.add_argument("--pid", dest="pid", action="store", help="Specify the pid file to use.")
    # Options that only apply to the config-less render usage
    parser.add_argument("--rendermodes", dest="rendermodes", action="store",
                        help="If you're not using a config file, specify which rendermodes to "
                        "render with this option. This is a comma-separated list.")
    parser.add_argument("world", nargs='?',
                        help="Path or name of the world you want to render.")
    parser.add_argument("output", nargs='?',
                        help="Output directory for the rendered map.")

    # Useful one-time render modifiers:
    render_modifiers = parser.add_mutually_exclusive_group()
    render_modifiers.add_argument("--forcerender", dest="forcerender", action="store_true",
                                  help="Force re-render the entire map.")
    render_modifiers.add_argument("--check-tiles", dest="checktiles", action="store_true",
                                  help="Check each tile on disk and re-render old tiles.")
    render_modifiers.add_argument("--no-tile-checks", dest="notilechecks", action="store_true",
                                  help="Only render tiles that come from chunks that have changed "
                                  "since the last render (the default).")

    # Useful one-time debugging options:
    parser.add_argument("--check-terrain", dest="check_terrain", action="store_true",
                        help="Try to locate the texture files. Useful for debugging texture"
                        " problems.")
    parser.add_argument("-V", "--version", dest="version",
                        help="Display version information and then exits.", action="store_true")
    parser.add_argument("--check-version", dest="checkversion",
                        help="Fetch information about the latest version of Overviewer.",
                        action="store_true")
    parser.add_argument("--update-web-assets", dest='update_web_assets', action="store_true",
                        help="Update web assets. Will *not* render tiles or update "
                        "overviewerConfig.js.")

    # Log level options:
    parser.add_argument("-q", "--quiet", dest="quiet", action="count", default=0,
                        help="Print less output. You can specify this option multiple times.")
    parser.add_argument("-v", "--verbose", dest="verbose", action="count", default=0,
                        help="Print more output. You can specify this option multiple times.")
    parser.add_argument("--simple-output", dest="simple", action="store_true", default=False,
                        help="Use a simple output format, with no colors or progress bars.")

    # create a group for "plugin exes"
    # (the concept of a plugin exe is only loosely defined at this point)
    exegroup = parser.add_argument_group("Other Scripts", "These scripts may accept different "
                                         "arguments than the ones listed above.")
    exegroup.add_argument("--genpoi", dest="genpoi", action="store_true",
                          help="Run the genPOI script.")
    exegroup.add_argument("--skip-scan", dest="skipscan", action="store_true",
                          help="When running GenPOI, don't scan for entities.")
    exegroup.add_argument("--skip-players", dest="skipplayers", action="store_true",
                          help="When running GenPOI, don't scan player data.")

    args, unknowns = parser.parse_known_args()

    # Check for possible shell quoting issues
    if len(unknowns) > 0 and args.world and args.output:
        possible_mistakes = []
        for i in range(len(unknowns) + 1):
            possible_mistakes.append(" ".join([args.world, args.output] + unknowns[:i]))
            possible_mistakes.append(" ".join([args.output] + unknowns[:i]))
        for mistake in possible_mistakes:
            if os.path.exists(mistake):
                logging.warning("Looks like you tried to make me use {0} as an argument, but "
                                "forgot to quote the argument correctly. Try using \"{0}\" "
                                "instead if the spaces are part of the path.".format(mistake))
                parser.error("Too many arguments.")
        parser.error("Too many arguments.")

    # first thing to do is check for stuff in the exegroup:
    if args.genpoi:
        # remove the "--genpoi" option from sys.argv before running genPI
        sys.argv.remove("--genpoi")
        g = __import__("overviewer_core.aux_files", {}, {}, ["genPOI"])
        g.genPOI.main()
        return 0

    # re-configure the logger now that we've processed the command line options
    logger.configure(logging.INFO + 10 * args.quiet - 10 * args.verbose,
                     verbose=args.verbose > 0, simple=args.simple)

    ##########################################################################
    # This section of main() runs in response to any one-time options we have,
    # such as -V for version reporting
    if args.version:
        print("Minecraft Overviewer %s" % util.findGitVersion() +
              " (%s)" % util.findGitHash()[:7])
        try:
            import overviewer_core.overviewer_version as overviewer_version
            print("built on %s" % overviewer_version.BUILD_DATE)
            if args.verbose > 0:
                print("Build machine: %s %s" % (overviewer_version.BUILD_PLATFORM,
                                                overviewer_version.BUILD_OS))
                print("Read version information from %r" % overviewer_version.__file__)
        except ImportError:
            print("(build info not found)")
        if args.verbose > 0:
            print("Python executable: %r" % sys.executable)
            print(sys.version)
        if not args.checkversion:
            return 0
    if args.checkversion:
        print("Currently running Minecraft Overviewer %s" % util.findGitVersion() +
              " (%s)" % util.findGitHash()[:7])
        try:
            from urllib import request
            import json
            latest_ver = json.loads(request.urlopen("http://overviewer.org/download.json")
                                    .read())['src']
            print("Latest version of Minecraft Overviewer %s (%s)" % (latest_ver['version'],
                                                                      latest_ver['commit'][:7]))
            print("See https://overviewer.org/downloads for more information.")
        except Exception:
            print("Failed to fetch latest version info.")
            if args.verbose > 0:
                import traceback
                traceback.print_exc()
            else:
                print("Re-run with --verbose for more details.")
            return 1
        return 0

    if args.pid:
        if os.path.exists(args.pid):
            try:
                with open(args.pid, 'r') as fpid:
                    pid = int(fpid.read())
                    if util.pid_exists(pid):
                        print("Overviewer is already running (pid exists) - exiting.")
                        return 0
            except (IOError, ValueError):
                pass
        with open(args.pid, "w") as f:
            f.write(str(os.getpid()))
    # if --check-terrain was specified, but we have NO config file, then we cannot
    # operate on a custom texture path.  we do terrain checking with a custom texture
    # pack later on, after we've parsed the config file
    if args.check_terrain and not args.config:
        import hashlib
        from overviewer_core.textures import Textures
        tex = Textures()

        logging.info("Looking for a few common texture files...")
        try:
            f = tex.find_file("assets/minecraft/textures/block/sandstone_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/grass_block_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/diamond_ore.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/acacia_planks.png", verbose=True)
            # 1.16
            f = tex.find_file("assets/minecraft/textures/block/ancient_debris_top.png",
                              verbose=True)
        except IOError:
            logging.error("Could not find any texture files.")
            return 1

        return 0

    # if no arguments are provided, print out a helpful message
    if not (args.world and args.output) and not args.config:
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
            logging.error("You must either specify --config or give me a world directory "
                          "and output directory.")
            parser.print_help()
            list_worlds()
        return 1

    ##########################################################################
    # This section does some sanity checking on the command line options passed
    # in. It checks to see if --config was given that no worldname/destdir were
    # given, and vice versa
    if args.config and (args.world and args.output):
        print()
        print("If you specify --config, you need to specify the world to render as well as "
              "the destination in the config file, not on the command line.")
        print("Put something like this in your config file:")
        print("worlds['myworld'] = %r" % args[0])
        print("outputdir = %r" % (args[1] if len(args) > 1 else "/path/to/output"))
        print()
        logging.error("You cannot specify both --config AND a world + output directory on the "
                      "command line.")
        parser.print_help()
        return 1

    if not args.config and (args.world or args.output) and not (args.world and args.output):
        logging.error("You must specify both the world directory and an output directory")
        parser.print_help()
        return 1

    #########################################################################
    # These two halfs of this if statement unify config-file mode and
    # command-line mode.
    mw_parser = config_parser.MultiWorldParser()

    if not args.config:
        # No config file mode.
        worldpath, destdir = map(os.path.expanduser, [args.world, args.output])
        logging.debug("Using %r as the world directory", worldpath)
        logging.debug("Using %r as the output directory", destdir)

        mw_parser.set_config_item("worlds", {'world': worldpath})
        mw_parser.set_config_item("outputdir", destdir)

        rendermodes = ['lighting']
        if args.rendermodes:
            rendermodes = args.rendermodes.replace("-", "_").split(",")

        # Now for some good defaults
        renders = OrderedDict()
        for rm in rendermodes:
            renders["world-" + rm] = {
                "world": "world",
                "title": "Overviewer Render (%s)" % rm,
                "rendermode": rm,
            }
        mw_parser.set_config_item("renders", renders)

    else:
        if args.rendermodes:
            logging.error("You cannot specify --rendermodes if you give a config file. "
                          "Configure your rendermodes in the config file instead.")
            parser.print_help()
            return 1

        # Parse the config file
        try:
            mw_parser.parse(os.path.expanduser(args.config))
        except config_parser.MissingConfigException as e:
            # this isn't a "bug", so don't print scary traceback
            logging.error(str(e))
            util.nice_exit(1)

    # Add in the command options here, perhaps overriding values specified in
    # the config
    if args.procs:
        mw_parser.set_config_item("processes", args.procs)

    # Now parse and return the validated config
    try:
        config = mw_parser.get_validated_config()
    except Exception as ex:
        if args.verbose:
            logging.exception("An error was encountered with your configuration. "
                              "See the information below.")
        else:   # no need to print scary traceback!
            logging.error("An error was encountered with your configuration.")
            logging.error(str(ex))
        return 1

    if args.check_terrain:   # we are already in the "if configfile" branch
        logging.info("Looking for a few common texture files...")
        for render_name, render in config['renders'].items():
            logging.info("Looking at render %r.", render_name)

            # find or create the textures object
            texopts = util.dict_subset(render, ["texturepath"])

            tex = textures.Textures(**texopts)
            f = tex.find_file("assets/minecraft/textures/block/sandstone_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/grass_block_top.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/diamond_ore.png", verbose=True)
            f = tex.find_file("assets/minecraft/textures/block/oak_planks.png", verbose=True)
        return 0

    ############################################################
    # Final validation steps and creation of the destination directory
    logging.info("Welcome to Minecraft Overviewer version %s (%s)!" % (util.findGitVersion(), util.findGitHash()[:7]))
    logging.debug("Current log level: {0}.".format(logging.getLogger().level))

    def set_renderchecks(checkname, num):
        for name, render in config['renders'].items():
            if render.get('renderchecks', 0) == 3:
                logging.warning(checkname + " ignoring render " + repr(name) + " since it's "
                                "marked as \"don't render\".")
            else:
                render['renderchecks'] = num

    if args.forcerender:
        logging.info("Forcerender mode activated. ALL tiles will be rendered.")
        set_renderchecks("forcerender", 2)
    elif args.checktiles:
        logging.info("Checking all tiles for updates manually.")
        set_renderchecks("checktiles", 1)
    elif args.notilechecks:
        logging.info("Disabling all tile mtime checks. Only rendering tiles "
                     "that need updating since last render.")
        set_renderchecks("notilechecks", 0)

    if not config['renders']:
        logging.error("You must specify at least one render in your config file. Check the "
                      "documentation at http://docs.overviewer.org if you're having trouble.")
        return 1

    #####################
    # Do a few last minute things to each render dictionary here
    for rname, render in config['renders'].items():
        # Convert render['world'] to the world path, and store the original
        # in render['worldname_orig']
        try:
            worldpath = config['worlds'][render['world']]
        except KeyError:
            logging.error("Render %s's world is '%s', but I could not find a corresponding entry "
                          "in the worlds dictionary.", rname, render['world'])
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
                        logging.error("Render %s's overlay is '%s', but I could not find a "
                                      "corresponding entry in the renders dictionary.", rname, x)
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
    assetMrg = assetmanager.AssetManager(destdir, config.get('customwebassets', None))

    # If we've been asked to update web assets, do that and then exit
    if args.update_web_assets:
        assetMrg.output_noconfig()
        logging.info("Web assets have been updated.")
        return 0

    # The changelist support.
    changelists = {}
    for render in config['renders'].values():
        if 'changelist' in render:
            path = render['changelist']
            if path not in changelists:
                out = open(path, "w")
                logging.debug("Opening changelist %s (%s).", out, out.fileno())
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
    # TODO: optionally more caching layers here

    renders = config['renders']
    for render_name, render in renders.items():
        logging.debug("Found the following render thing: %r", render)

        # find or create the world object
        try:
            w = worldcache[render['world']]
        except KeyError:
            try:
                w = world.World(render['world'])
            except CorruptNBTError as e:
                logging.error("Failed to open world %r.", render['world'])
                raise e
            except world.UnsupportedVersion as e:
                for ln in str(e).split('\n'):
                    logging.error(ln)
                sys.exit(1)

            worldcache[render['world']] = w

        # find or create the textures object
        texopts = util.dict_subset(render, ["texturepath", "bgcolor", "northdirection"])
        texopts_key = tuple(texopts.items())
        if texopts_key not in texcache:
            tex = textures.Textures(**texopts)
            logging.info("Generating textures...")
            tex.generate()
            logging.debug("Finished generating textures.")
            texcache[texopts_key] = tex
        else:
            tex = texcache[texopts_key]

        try:
            logging.debug("Asking for regionset %r." % render['dimension'][1])
            rset = w.get_regionset(render['dimension'][1])
        except IndexError:
            logging.error("Sorry, I can't find anything to render!  Are you sure there are .mca "
                          "files in the world directory of %s?" % render['world'])
            return 1
        if rset is None:    # indicates no such dimension was found
            logging.warning("Sorry, you requested dimension '%s' for %s, but I couldn't find it.",
                         render['dimension'][0], render_name)
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
        render['name'] = render_name    # perhaps a hack. This is stored here for the asset manager
        tileSetOpts = util.dict_subset(render, [
            "name", "imgformat", "renderchecks", "rerenderprob", "bgcolor", "defaultzoom",
            "imgquality", "imglossless", "optimizeimg", "rendermode", "worldname_orig", "title",
            "dimension", "changelist", "showspawn", "overlay", "base", "poititle", "maxzoom",
            "showlocationmarker", "minzoom", "center"])
        tileSetOpts.update({"spawn": w.find_true_spawn()})  # TODO find a better way to do this
        for rset in rsets:
            tset = tileset.TileSet(w, rset, assetMrg, tex, tileSetOpts, tileset_dir)
            tilesets.append(tset)

    # If none of the requested dimenstions exist, tilesets will be empty
    if not tilesets:
        logging.error("There are no tilesets to render! There's nothing to do, so exiting.")
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

    for out in changelists.values():
        logging.debug("Closing %s (%s).", out, out.fileno())
        out.close()

    if config['processes'] == 1:
        logging.debug("Final cache stats:")
        for c in caches:
            logging.debug("\t%s: %s hits, %s misses", c.__class__.__name__, c.hits, c.misses)
    if args.pid:
        os.remove(args.pid)

    logging.info("Your render has been written to '%s', open index.html to view it." % destdir)

    return 0


def list_worlds():
    "Prints out a brief summary of saves found in the default directory"
    print()
    worlds = world.get_worlds()
    if not worlds:
        print('No world saves found in the usual place.')
        return
    print("Detected saves:")

    # get max length of world name
    worldNameLen = max([len(x) for x in worlds] + [len("World")])

    formatString = "%-" + str(worldNameLen) + "s | %-8s | %-16s | %s "
    print(formatString % ("World", "Playtime", "Modified", "Path"))
    print(formatString % ("-" * worldNameLen, "-" * 8, '-' * 16, '-' * 4))
    for name, info in sorted(worlds.items()):
        if isinstance(name, str) and name.startswith("World") and len(name) == 6:
            try:
                world_n = int(name[-1])
                # we'll catch this one later, when it shows up as an
                # integer key
                continue
            except ValueError:
                pass
        if info['LastPlayed'] > 0:
            timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(info['LastPlayed'] / 1000))
        else:
            timestamp = ""
        if info['Time'] > 0:
            playtime = info['Time'] / 20
            playstamp = '%d:%02d' % (playtime / 3600, playtime / 60 % 60)
        else:
            playstamp = ""
        path = info['path']
        print(formatString % (name, playstamp, timestamp, path))
    found_corrupt = any([x.get("IsCorrupt") for x in worlds.values()])
    if found_corrupt:
        print()
        print("An error has been detected in one or more of your worlds (see the above table).")
        print("This is usually due to a corrupt level.dat file. Corrupt worlds need to be "
              "repaired before Overviewer can render them.")


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
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Aborting.")
        util.nice_exit(2)
