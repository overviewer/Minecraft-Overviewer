#!/usr/bin/python

'''
Generate a region list to rerender certain chunks

This is used to force the regeneration of any chunks that contain a certain
blockID.  The output is a chunklist file that is suitable to use with the
--chunklist option to overviewer.py.

Example:

python contrib/rerenderBlocks.py --ids=46,79,91 --world=world/> regionlist.txt
    python overviewer.py --regionlist=regionlist.txt world/ output_dir/

This will rerender any chunks that contain either TNT (46), Ice (79), or 
a Jack-O-Lantern (91)
'''

from optparse import OptionParser
import sys,os
import re

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

from overviewer_core import nbt
from overviewer_core import world
from overviewer_core.chunk import get_blockarray

parser = OptionParser()
parser.add_option("--ids", dest="ids", type="string")
parser.add_option("--world", dest="world", type="string")


options, args = parser.parse_args()

if not options.world or not options.ids:
    parser.print_help()
    sys.exit(1)

if not os.path.exists(options.world):
    raise Exception("%s does not exist" % options.world)

ids = map(lambda x: int(x),options.ids.split(","))
sys.stderr.write("Searching for these blocks: %r...\n" % ids)


matcher = re.compile(r"^r\..*\.mcr$")

for dirpath, dirnames, filenames in os.walk(options.world):
    for f in filenames:
        if matcher.match(f):
            full = os.path.join(dirpath, f)
            r = nbt.load_region(full, 'lower-left')
            chunks = r.get_chunks()
            found = False
            for x,y in chunks:
                chunk = r.load_chunk(x,y).read_all()                
                blocks = get_blockarray(chunk[1]['Level'])
                for i in ids:
                    if chr(i) in blocks:
                        print full
                        found = True
                        break
                if found:
                    break


