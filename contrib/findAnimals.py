#!/usr/bin/python

'''
Updates overviewer.dat file sign info

This script will scan through every chunk looking for signs and write out an
updated overviewer.dat file.  This can be useful if your overviewer.dat file
is either out-of-date or non-existant.  

To run, simply give a path to your world directory and the path to your
output directory. For example:

    python contrib/findAnimals.py ../world.test/ output_dir/ 

An optional north direction may be specified as follows:
    
    python contrib/findAnimals.py ../world.test/ output_dir/ lower-right

Valid options are upper-left, upper-right, lower-left and lower-right.
If no direction is specified, lower-left is assumed

Once that is done, simply re-run the overviewer to generate markers.js:

    python overviewer.py ../world.test/ output_dir/

'''
import sys
import re
import os
import cPickle

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))

from overviewer_core import nbt

print "finding animals, fuck yeah!"


from pprint import pprint
if len(sys.argv) < 3:
    sys.exit("Usage: %s <worlddir> <outputdir> [north_direction]" % sys.argv[0])
    
worlddir = sys.argv[1]
outputdir = sys.argv[2]

directions=["upper-left","upper-right","lower-left","lower-right"]
if len(sys.argv) < 4:
    print "No north direction specified - assuming lower-left"
    north_direction="lower-left"
else:
    north_direction=sys.argv[3]

if (north_direction not in directions):
    print north_direction, " is not a valid direction"
    sys.exit("Bad north-direction")

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
            # force lower-left so chunks are loaded in correct positions
            r = nbt.load_region(full, 'lower-left')
            chunks = r.get_chunks()
            for x,y in chunks:
                chunk = r.load_chunk(x,y).read_all()                
                data = chunk[1]['Level']['Entities']
                for entity in data:
                    if entity['id'] == 'Cow':
                        newPOI = dict(type="cow",
                                        x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='moo', chunk= (entity['Pos'][0]/16, entity['Pos'][2]/16),
                                       )
                        POI.append(newPOI)
                    elif entity['id'] == 'Sheep':
                        newPOI = dict(type="sheep",
                                        x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='baa', chunk= (entity['Pos'][0]/16, entity['Pos'][2]/16),
                                       )
                        POI.append(newPOI)
                    elif entity['id'] == 'Pig':
                        newPOI = dict(type="pig",
                                        x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='oink',chunk= (entity['Pos'][0]/16, entity['Pos'][2]/16),
                                       )
                        POI.append(newPOI)
                    elif entity['id'] == 'Chicken':
                        newPOI = dict(type="chicken",
                                        x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='cluck', chunk= (entity['Pos'][0]/16, entity['Pos'][2]/16),
                                       )
                        POI.append(newPOI)
                    elif entity['id'] == 'Squid':
                        newPOI = dict(type="squid",
                                        x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='squelch', chunk= (entity['Pos'][0]/16, entity['Pos'][2]/16),
                                       )
                        POI.append(newPOI)
                    print "Found %s at (%d, %d, %d)" % (newPOI['type'], newPOI['x'], newPOI['y'], newPOI['z'])


if os.path.isfile(os.path.join(worlddir, "overviewer.dat")):
    print "Overviewer.dat detected in WorldDir - this is no longer the correct location\n"
    print "You may wish to delete the old file. A new overviewer.dat will be created\n"
    print "Old file: ", os.path.join(worlddir, "overviewer.dat")

pickleFile = os.path.join(outputdir,"overviewer.dat")
with open(pickleFile,"wb") as f:
    cPickle.dump(dict(POI=POI,north_direction=north_direction), f)

