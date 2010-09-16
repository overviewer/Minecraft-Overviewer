#!/usr/bin/python

import os
import sys
import os.path
from optparse import OptionParser
import re
import multiprocessing

import world
import quadtree

helptext = """
%prog [-p PROCS] [-d] <Path to World> <tiles dest dir>
"""

def main():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    parser = OptionParser(usage=helptext)
    parser.add_option("-p", "--processes", dest="procs", help="How many chunks to render in parallel. A good number for this is the number of cores in your computer. Default %s" % cpus, default=cpus, action="store", type="int")
    parser.add_option("-d", "--delete", dest="delete", help="Clear all caches. Next time you render your world, it will have to start completely over again. This is probably not a good idea for large worlds. Use this if you change texture packs and want to re-render everything.", action="store_true")

    options, args = parser.parse_args()

    if len(args) < 1:
        print "You need to give me your world directory"
        parser.print_help()
        sys.exit(1)
    worlddir = args[0]

    if len(args) != 2:
        parser.error("Where do you want to save the tiles?")
    destdir = args[1]

    if options.delete:
        return delete_all(worlddir, destdir)

    # First generate the world's chunk images
    w = world.WorldRenderer(worlddir)
    w.go(options.procs)

    # Now generate the tiles
    q = quadtree.QuadtreeGen(w, destdir)
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

if __name__ == "__main__":
    main()
