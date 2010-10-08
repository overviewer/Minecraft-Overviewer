#!/bin/bash

# Usage: ./resume-incremental.sh 
# Example: ./resume-incremental.sh

MCHOME=/minecraft
GMAP=$MCHOME/apps/Minecraft-Overviewer/gmap.py
WORLD=$MCHOME/world_snapshot
CACHE=$MCHOME/cache
OUTPUT=$MCHOME/maps/

LOGPATH=`ls /minecraft/logs/*.log -t1 | head -n1`
CHUNKLIST=`ls /minecraft/logs/*.rsync -t1 | head -n1`


START=$(date +%s)

echo "Start at: $START" >> $LOGPATH

# Make sure we are in the right directory
cd $MCHOME

# Run incremental update
echo "Start incremental update at: $(date +%s)" >> $LOGPATH
echo "python $GMAP --cachedir=$CACHE --chunklist=$CHUNKLIST $WORLD $OUTPUT" >> $LOGPATH 
python $GMAP --cachedir=$CACHE --chunklist=$CHUNKLIST $WORLD $OUTPUT 
if [ $? -ne 0 ]; then
   sleep 120;
   echo "python returned error, sleeping before retrying";
fi

echo "End incremental update at: $(date +%s)" >> $LOGPATH

# Calculate end time
END=$(date +%s)
DIFF=$(( $END - $START))

echo "It took $DIFF seconds"
echo "It took $DIFF seconds" >> $LOGPATH
let "MINS=$DIFF / 60"
let "HOURS=$MINS / 60"
echo " or $MINS minutes"
echo " or $MINS minutes" >> $LOGPATH
echo " or $HOURS hours"
echo " or $HOURS hours" >> $LOGPATH

echo "End at: $END" >> $LOGPATH
echo "DIFF: $DIFF" >> $LOGPATH 
