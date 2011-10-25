#!/usr/bin/python

"""Deletes files from the old chunk-based cache"""


usage = "python contrib/%prog [OPTIONS] <World # / Name / Path to World>"

description = """
This script will delete files from the old chunk-based cache, a lot
like the old `overviewer.py -d World/` command. You should only use this if
you're updating from an older version of Overviewer, and you want to
clean up your world folder.
"""

from optparse import OptionParser
import sys
import re
import os.path

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

from overviewer_core import world
from overviewer import list_worlds

def main():
    parser = OptionParser(usage=usage, description=description)
    parser.add_option("-d", "--dry-run", dest="dry", action="store_true",
                      help="Don't actually delete anything. Best used with -v.")
    parser.add_option("-k", "--keep-dirs", dest="keep", action="store_true",
                      help="Keep the world directories intact, even if they are empty.")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Log each and every file that is deleted.")
    
    opt, args = parser.parse_args()
    
    if not len(args) == 1:
        parser.print_help()
        sys.exit(1)
        
    worlddir = args[0]

    if not os.path.exists(worlddir):
        # world given is either world number, or name
        worlds = world.get_worlds()
        
        # if there are no worlds found at all, exit now
        if not worlds:
            parser.print_help()
            print "\nInvalid world path"
            sys.exit(1)
        
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
                print "Invalid world name or path"
                sys.exit(1)
        except KeyError:
            # it was an invalid number
            parser.print_help()
            print "Invalid world number"
            sys.exit(1)
    
    files_deleted = 0
    dirs_deleted = 0
    
    imgre = re.compile(r'img\.[^.]+\.[^.]+\.nocave\.\w+\.png$')
    for dirpath, dirnames, filenames in os.walk(worlddir, topdown=False):
        for f in filenames:
            if imgre.match(f):
                filepath = os.path.join(dirpath, f)
                if opt.verbose:
                    print "Deleting %s" % (filepath,)
                if not opt.dry:
                    os.unlink(filepath)
                    files_deleted += 1
        
        if not opt.keep:
            if len(os.listdir(dirpath)) == 0:
                if opt.verbose:
                    print "Deleting %s" % (dirpath,)
                if not opt.dry:
                    os.rmdir(dirpath)
                    dirs_deleted += 1
    
    print "%i files and %i directories deleted." % (files_deleted, dirs_deleted)

if __name__ == "__main__":
    main()
