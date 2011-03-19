#!/usr/bin/python

usage = "python contrib/%prog [OPTIONS] (<regionfilename>)*"

description = """
This script will valide a minecraft region file for errors
"""

from optparse import OptionParser
import sys
import re
import os.path
import logging



def main():
    # sys.path wrangling, so we can access Overviewer code
    overviewer_dir = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    sys.path.insert(0, overviewer_dir)
    import nbt
    #import chunk
    #import quadtree
    parser = OptionParser(usage=usage, description=description)
    parser.add_option("-r", "--regions", dest="regiondir", help="Use path to the regions instead of a list of files")
    parser.add_option("-v", dest="verbose", action="store_true", help="Lists why a chunk in a region failed")
    
    opt, args = parser.parse_args()
    
    if opt.regiondir:
        if os.path.exists(opt.regiondir):
            for dirpath, dirnames, filenames in os.walk(opt.regiondir, 'region'):        
                if not dirnames and filenames and "DIM-1" not in dirpath:
                    for f in filenames:
                        if f.startswith("r.") and f.endswith(".mcr"):
                            p = f.split(".")
                            args.append(os.path.join(dirpath, f))
                        
    if len(args) < 1:
        print "You must list at least one region file"
        parser.print_help()
        sys.exit(1)
   
    for regionfile in args:
        _,shortname = os.path.split(regionfile)
        chunk_pass = 0    
        chunk_total = 0          
        if not os.path.exists(regionfile):
            print("File not found: %s"%( regionfile))
            #print("Region:%s Passed %s/%s"%(shortname,chunk_pass,chunk_total))
            continue          
        try:
            mcr = nbt.load_region(regionfile)
        except IOError, e:
            if opt.verbose:
                print("Error opening regionfile. It may be corrupt. %s"%( e))
            continue
        if mcr is not None:
            try:
                chunks = mcr.get_chunk_info(False)
            except IOError, e:
                if opt.verbose:
                    print("Error opening regionfile(bad header info). It may be corrupt. %s"%( e))
                chunks = []
                continue
            except Exception, e:
                if opt.verbose:
                    print("Error opening regionfile (%s): %s"%( regionfile,e))
                continue
            for x, y in chunks:
                chunk_total += 1
                try:
                    chunk_data = mcr.load_chunk(x, y)
                except Exception, e:
                    if opt.verbose:
                        print("Error reading chunk (%i,%i): %s"%(x,y,e))
                    continue
                if chunk_data is None:
                    if opt.verbose:
                        print("Chunk %s:%s is unexpectedly empty"%(x, y))
                    continue
                else:
                    try:
                        processed = chunk_data.read_all()
                        if processed == []:
                            if opt.verbose:
                                print("Chunk %s:%s is an unexpectedly empty set"%(x, y))
                            continue
                        else:
                            chunk_pass += 1
                    except Exception, e:
                        if opt.verbose:
                            print("Error opening chunk (%i, %i) It may be corrupt. %s"%( x, y, e))
                        continue
        else:           
            if opt.verbose:    
                print("Error opening regionfile.")
            continue
         
        print("Region:%s Passed %s/%s"%(shortname,chunk_pass,chunk_total))
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print "Caught Ctrl-C"
        
