#!/usr/bin/python

usage = "python contrib/%prog [OPTIONS] <regionfilename>"

description = """
This script will delete files from the old chunk-based cache, a lot
like the old `gmap.py -d World/` command. You should only use this if
you're updating from an older version of Overviewer, and you want to
clean up your world folder.
"""

from optparse import OptionParser
import sys
import re
import os.path
import logging

# sys.path wrangling, so we can access Overviewer code
overviewer_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.insert(0, overviewer_dir)

import nbt
import chunk

def main():
    parser = OptionParser(usage=usage, description=description)
#    parser.add_option("-d", "--dry-run", dest="dry", action="store_true",
#                      help="Don't actually delete anything. Best used with -v.")
    
    opt, args = parser.parse_args()
    
    if not len(args) == 1:
        parser.print_help()
        sys.exit(1)
                
    regionfile = args[0]
      
    if not os.path.exists(regionfile):
        parser.print_help()
        print "\nFile not found"
        sys.exit(1)
    chunk_pass = 0    
    chunk_total = 0
    print( "Loading region: %s" % ( regionfile))
    try:
        mcr = nbt.load_region(regionfile)
    except IOError, e:
        print("Error opening regionfile. It may be corrupt. %s"%( e))
        pass
    if mcr is not None:
        try:
            chunks = mcr.get_chunk_info(False)
        except IOError, e:
            print("Error opening regionfile(bad header info). It may be corrupt. %s"%( e))
            chunks = []
            pass            
        for x, y in chunks:
            chunk_total += 1
            #try:
            chunk_data = mcr.load_chunk(x, y)
            if chunk_data is None:
               print("Chunk %s:%s is unexpectedly empty"%(x, y))
            else:
                try:
                    processed = chunk_data.read_all()
                    if processed == []:
                        print("Chunk %s:%s is an unexpectedly empty set"%(x, y))
                    else:
                        chunk_pass += 1
                except Exception, e:
                    print("Error opening chunk (%i, %i) It may be corrupt. %s"%( x, y, e))
    else:                
        print("Error opening regionfile.")
    print("Done; Passed %s/%s"%(chunk_pass,chunk_total))
if __name__ == "__main__":
    main()
