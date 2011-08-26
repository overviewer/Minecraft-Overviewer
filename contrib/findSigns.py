#!/usr/bin/python

'''
Updates overviewer.dat file sign info

This script will scan through every chunk looking for signs and write out an
updated overviewer.dat file.  This can be useful if your overviewer.dat file
is either out-of-date or non-existant.  

To run, simply give a path to your world directory and the path to your
output directory. For example:

    python contrib/findSigns.py ../world.test/ output_dir/

Once that is done, simply re-run the overviewer to generate markers.js:

    python overviewer.py ../world.test/ output_dir/

'''
import sys
import re
import os
import cPickle

sys.path.append(".")
from overviewer_core import nbt

from pprint import pprint

worlddir = sys.argv[1]
outputdir = sys.argv[2]
if os.path.exists(worlddir):
    print "Scanning chunks in ", worlddir
else:
    sys.exit("Bad WorldDir")

if os.path.exists(outputdir):
    print "Output directory is ", outputdir
else:
    sys.exit("Bad OutputDir")

matcher = re.compile(r"^r\..*\.mcr$")

POI = []

for dirpath, dirnames, filenames in os.walk(worlddir):
    for f in filenames:
        if matcher.match(f):
            print f
            full = os.path.join(dirpath, f)
            r = nbt.load_region(full)
            chunks = r.get_chunks()
            for x,y in chunks:
                chunk = r.load_chunk(x,y).read_all()                
                data = chunk[1]['Level']['TileEntities']
                for entity in data:
                    if entity['id'] == 'Sign':
                        msg=' \n'.join([entity['Text1'], entity['Text2'], entity['Text3'], entity['Text4']])
                        #print "checking -->%s<--" % msg.strip()
                        if msg.strip():
                            newPOI = dict(type="sign",
                                            x= entity['x'],
                                            y= entity['y'],
                                            z= entity['z'],
                                            msg=msg,
                                            chunk= (entity['x']/16, entity['z']/16),
                                           )
                            POI.append(newPOI)
                            print "Found sign at (%d, %d, %d): %r" % (newPOI['x'], newPOI['y'], newPOI['z'], newPOI['msg'])


if os.path.isfile(os.path.join(worlddir, "overviewer.dat")):
    print "Overviewer.dat detected in WorldDir - this is no longer the correct location\n"
    print "You may wish to delete the old file. A new overviewer.dat will be created\n"
    print "Old file: ", os.path.join(worlddir, "overviewer.dat")

pickleFile = os.path.join(outputdir,"overviewer.dat")
with open(pickleFile,"wb") as f:
    cPickle.dump(dict(POI=POI), f)

