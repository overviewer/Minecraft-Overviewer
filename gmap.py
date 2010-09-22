#!/usr/bin/python

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


import os
import sys
import os.path
from optparse import OptionParser
import re
import multiprocessing
import time

import world
import quadtree

helptext = """
%prog [-p PROCS] [-d] <World # / Path to World> <tiles dest dir>
"""

def main():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    parser = OptionParser(usage=helptext)
    parser.add_option("-p", "--processes", dest="procs", help="How many chunks to render in parallel. A good number for this is the number of cores in your computer. Default %s" % cpus, default=cpus, action="store", type="int")
    parser.add_option("-z", "--zoom", dest="zoom", help="Sets the zoom level manually instead of calculating it. This can be useful if you have outlier chunks that make your world too big. This value will make the highest zoom level contain (2**ZOOM)^2 tiles", action="store", type="int")
    parser.add_option("-d", "--delete", dest="delete", help="Clear all caches. Next time you render your world, it will have to start completely over again. This is probably not a good idea for large worlds. Use this if you change texture packs and want to re-render everything.", action="store_true")

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


    if len(args) != 2:
        parser.error("Where do you want to save the tiles?")
    destdir = args[1]

    if options.delete:
        return delete_all(worlddir, destdir)
    # First generate the world's chunk images
    w = world.WorldRenderer(worlddir)
    w.go(options.procs)

    # Now generate the tiles
    q = quadtree.QuadtreeGen(w, destdir, depth=options.zoom)
    q.go(options.procs)

def delete_all(worlddir, tiledir):
    # First delete all images in the world dir
    imgre = r"img\.[^.]+\.[^.]+\.nocave\.\w+\.png$"
    matcher = re.compile(imgre)

    for dirpath, dirnames, filenames in os.walk(worlddir):
        for f in filenames:
            if matcher.match(f):
                filepath = os.path.join(dirpath, f)
                print "Deleting {0}".format(filepath)
                os.unlink(filepath)

    # Now delete all /hash/ files in the tile dir.
    for dirpath, dirnames, filenames in os.walk(tiledir):
        for f in filenames:
            if f.endswith(".hash"):
                filepath = os.path.join(dirpath, f)
                print "Deleting {0}".format(filepath)
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
    main()
