#!/usr/bin/python

import os
import sys
import os.path
from optparse import OptionParser
import re

import world

helptext = """
%prog [-p PROCS] <Path to World> <tiles dest dir>
"""

def main():
    parser = OptionParser(usage=helptext)
    parser.add_option("-p", "--processes", dest="procs", help="How many chunks to render in parallel. A good number for this is 1 more than the number of cores in your computer. Default 2", default=2, action="store", type="int")

    options, args = parser.parse_args()

    if len(args) < 1:
        print "You need to give me your world directory"
        parser.print_help()
        sys.exit(1)
    worlddir = args[0]

    if len(args) != 2:
        parser.error("Where do you want to save the tiles?")
    destdir = args[1]

    print "Scanning chunks"
    all_chunks = world.find_chunkfiles(worlddir)

    # Translate chunks from diagonal coordinate system
    mincol, maxcol, minrow, maxrow, chunks = world.convert_coords(all_chunks)

    print "processing chunks in background"
    results = world.render_chunks_async(chunks, False, options.procs)

    print "Generating quad tree. This may take a while and has no progress bar right now, so sit tight."

    zoom = world.generate_quadtree(results, mincol, maxcol, minrow, maxrow, destdir)

    print "DONE"

    print "Writing out html file"
    write_html(destdir, zoom+1)

def write_html(path, zoomlevel):
    templatepath = os.path.join(os.path.split(__file__)[0], "template.html")
    html = open(templatepath, 'r').read()
    html = html.replace(
            "{maxzoom}", str(zoomlevel))
            
    with open(os.path.join(path, "index.html"), 'w') as output:
        output.write(html)

if __name__ == "__main__":
    main()
