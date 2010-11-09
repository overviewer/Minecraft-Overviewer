#!/usr/bin/python

'''
This is used to force the regeneration of any chunks that contain a certain
blockID.  The output is a chunklist file that is suitable to use with the
--chunklist option to gmap.py.

Example:

python contrib/rerenderBlocks.py --ids=46,79,91 --world=world/> chunklist.txt
    python gmap.py --chunklist=chunklist.txt world/ output_dir/

This will rerender any chunks that contain either TNT (46), Ice (79), or 
a Jack-O-Lantern (91)
'''

from optparse import OptionParser

import sys
sys.path.insert(0,".")

import nbt
from chunk import get_blockarray_fromfile
import os
import re

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


matcher = re.compile(r"^c\..*\.dat$")

for dirpath, dirnames, filenames in os.walk(options.world):
    for f in filenames:
        if matcher.match(f):
            full = os.path.join(dirpath, f)
            blocks = get_blockarray_fromfile(full)
            for i in ids:
                if i in blocks:
                    print full
                    break

