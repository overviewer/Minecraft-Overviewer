#!/bin/bash

# Usage: ./update-markers.sh
# Example: while true; do /minecraft/apps/update-markers.sh; sleep 3; done

MCHOME=/minecraft
GMAP=$MCHOME/apps/Minecraft-Overviewer/gmap.py
WORLD=$MCHOME/world
OUTPUT=$MCHOME/maps

# Make sure we are in the right directory
cd $MCHOME

# Update markers
python $GMAP --markers $WORLD $OUTPUT
RETURNVAL=$?
if [ $RETURNVAL -ne 0 ] ; then
 echo "Update failed"
 exit 1
fi

