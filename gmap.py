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

import sys
if sys.version_info[0] != 2 and sys.version_info[1] < 6:
    print "Sorry, the Overviewer requires at least Python 2.6 to run"
    sys.exit(1)

import os
import os.path
from optparse import OptionParser
import re
import multiprocessing
import time
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s")

import world
import quadtree

helptext = """
%prog [OPTIONS] <World # / Path to World> <tiles dest dir>
%prog -d <World # / Path to World / Path to cache dir> [tiles dest dir]"""

def main():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    parser = OptionParser(usage=helptext)
    parser.add_option("-p", "--processes", dest="procs", help="How many worker processes to start. Default %s" % cpus, default=cpus, action="store", type="int")
    parser.add_option("-z", "--zoom", dest="zoom", help="Sets the zoom level manually instead of calculating it. This can be useful if you have outlier chunks that make your world too big. This value will make the highest zoom level contain (2**ZOOM)^2 tiles", action="store", type="int")
    parser.add_option("-d", "--delete", dest="delete", help="Clear all caches. Next time you render your world, it will have to start completely over again. This is probably not a good idea for large worlds. Use this if you change texture packs and want to re-render everything.", action="store_true")
    parser.add_option("--cachedir", dest="cachedir", help="Sets the directory where the Overviewer will save chunk images, which is an intermediate step before the tiles are generated. You must use the same directory each time to gain any benefit from the cache. If not set, this defaults to your world directory.")
    parser.add_option("--chunklist", dest="chunklist", help="A file containing, on each line, a path to a chunkfile to update. Instead of scanning the world directory for chunks, it will just use this list. Normal caching rules still apply.")
    parser.add_option("--imgformat", dest="imgformat", help="The image output format to use. Currently supported: png(default), jpg. NOTE: png will always be used as the intermediate image format.")
    parser.add_option("--optimize-img", dest="optimizeimg", help="If using png, perform image file size optimizations on the output. Specify 1 for pngcrush, 2 for pngcrush+optipng+advdef. This may double (or more) render times, but will produce up to 30% smaller images. NOTE: requires corresponding programs in $PATH or %PATH%")
    parser.add_option("-q", "--quiet", dest="quiet", action="count", default=0, help="Print less output. You can specify this option multiple times.")
    parser.add_option("-v", "--verbose", dest="verbose", action="count", default=0, help="Print more output. You can specify this option multiple times.")
    parser.add_option("--skip-js", dest="skipjs", action="store_true", help="Don't output marker.js or regions.js")

    options, args = parser.parse_args()

    if len(args) < 1:
        print "You need to give me your world number or directory"
        parser.print_help()
        list_worlds()
        sys.exit(1)
    worlddir = args[0]

    if not os.path.exists(worlddir):
        try:
            worldnum = int(worlddir)
            worlddir = world.get_worlds()[worldnum]['path']
        except (ValueError, KeyError):
            print "Invalid world number or directory"
            parser.print_help()
            sys.exit(1)

    if not options.cachedir:
        cachedir = worlddir
    else:
        cachedir = options.cachedir

    if len(args) != 2:
        if options.delete:
            return delete_all(cachedir, None)
        parser.error("Where do you want to save the tiles?")

    destdir = args[1]

    if options.delete:
        return delete_all(cachedir, destdir)

    if options.chunklist:
        chunklist = open(options.chunklist, 'r')
    else:
        chunklist = None

    if options.imgformat:
        if options.imgformat not in ('jpg','png'):
            parser.error("Unknown imgformat!")
        else:
            imgformat = options.imgformat
    else:
        imgformat = 'png'

    if options.optimizeimg:
        optimizeimg = options.optimizeimg
    else:
        optimizeimg = None

    logging.getLogger().setLevel(
        logging.getLogger().level + 10*options.quiet)
    logging.getLogger().setLevel(
        logging.getLogger().level - 10*options.verbose)

    logging.info("Welcome to Minecraft Overviewer!")
    logging.debug("Current log level: {0}".format(logging.getLogger().level))

    # First generate the world's chunk images
    w = world.WorldRenderer(worlddir, cachedir, chunklist=chunklist)
    w.go(options.procs)

    # Now generate the tiles
    q = quadtree.QuadtreeGen(w, destdir, depth=options.zoom, imgformat=imgformat, optimizeimg=optimizeimg)
    q.write_html(options.skipjs)
    q.go(options.procs)

def delete_all(worlddir, tiledir):
    # First delete all images in the world dir
    imgre = r"img\.[^.]+\.[^.]+\.nocave\.\w+\.png$"
    matcher = re.compile(imgre)

    for dirpath, dirnames, filenames in os.walk(worlddir):
        for f in filenames:
            if matcher.match(f):
                filepath = os.path.join(dirpath, f)
                logging.info("Deleting {0}".format(filepath))
                os.unlink(filepath)

    # Now delete all /hash/ files in the tile dir.
    if tiledir:
        for dirpath, dirnames, filenames in os.walk(tiledir):
            for f in filenames:
                if f.endswith(".hash"):
                    filepath = os.path.join(dirpath, f)
                    logging.info("Deleting {0}".format(filepath))
                    os.unlink(filepath)

def list_worlds():
    "Prints out a brief summary of saves found in the default directory"
    print 
    worlds = world.get_worlds()
    if not worlds:
        print 'No world saves found in the usual place'
        return
    print "Detected saves:"
    for num, info in sorted(worlds.iteritems()):
        timestamp = time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(info['LastPlayed'] / 1000))
        playtime = info['Time'] / 20
        playstamp = '%d:%02d' % (playtime / 3600, playtime / 60 % 60)
        size = "%.2fMB" % (info['SizeOnDisk'] / 1024. / 1024.)
        print "World %s: %s Playtime: %s Modified: %s" % (num, size, playstamp, timestamp)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
