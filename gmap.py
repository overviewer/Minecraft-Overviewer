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
%prog [-p PROCS] <Path to World> <tiles dest dir>
"""

def main():
    try:
        cpus = multiprocessing.cpu_count()
    except NotImplementedError:
        cpus = 1
    parser = OptionParser(usage=helptext)
    parser.add_option("-p", "--processes", dest="procs", help="How many chunks to render in parallel. A good number for this is the number of cores in your computer. Default %s" % cpus, default=cpus, action="store", type="int")

    options, args = parser.parse_args()

    if len(args) < 1:
        print "You need to give me your world directory"
        parser.print_help()
        sys.exit(1)
    worlddir = args[0]

    if len(args) != 2:
        parser.error("Where do you want to save the tiles?")
    destdir = args[1]

    # First generate the world's chunk images
    w = world.WorldRenderer(worlddir)
    w.go(options.procs)

    # Now generate the tiles
    q = quadtree.QuadtreeGen(w, destdir)
    q.go(options.procs)

if __name__ == "__main__":
    main()
