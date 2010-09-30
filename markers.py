#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.

import functools
import os
import os.path
import numpy
import math
import sys
import chunk
import nbt
import json

"""
This module has routines related to exporting markers to overlay on a generated map

"""


def base36encode(number, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    '''
    Convert an integer to a base36 string.
    '''
    if not isinstance(number, (int, long)):
        raise TypeError('number must be an integer')
    
    newn = abs(number)
    
    # Special case for zero
    if number == 0:
        return '0'
    
    base36 = ''
    while newn != 0:
        newn, i = divmod(newn, len(alphabet))
        base36 = alphabet[i] + base36
    
    if number < 0:
        return "-" + base36
    return base36
    
    
class MarkerGenerator(object):
    """Generates all markers to overlay on map.
    worlddir is the path to the minecraft world
    
    """
    def __init__(self, worlddir, destdir):
        self.worlddir = worlddir
        self.destdir = destdir
        
        #  stores Points Of Interest to be mapped with markers
        #  a list of dictionaries, see below for an example
        self.POI = []

    def addSpawn(self):
        """Adds the true spawn location to self.POI."""  

        ## read spawn info from level.dat
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]
        spawnX = data['Data']['SpawnX']
        spawnY = data['Data']['SpawnY']
        spawnZ = data['Data']['SpawnZ']

        self.POI.append( dict(x=spawnX, y=spawnY, z=spawnZ, msg="Spawn", id=0))

    def addLabels(self):
        """Adds the labels from the server to self.POI."""
        ## read label info from mapper-labels.txt
        path = os.path.join(self.worlddir, "mapper-labels.txt")
        print "Adding labels to map"
        self.addMarkers(path);
        
        
    def addPlayers(self):
        """Adds the players positions from the server to self.POI."""

        ## read label info from mapper-labels.txt
        path = os.path.join(self.worlddir, "mapper-playerpos.txt")
        print "Adding players to map"
        self.addMarkers(path);

    def addHomes(self):
        """Adds the players homes from the server to self.POI."""
        ## read label info from mapper-labels.txt
        path = os.path.join(self.worlddir, "mapper-homes.txt")
        print "Adding homes to map"
        self.addMarkers(path);
    
    def addMarkers(self, path):
        """Add marker to array"""
        
        try:
            if os.path.exists(path):
                fileobj = open(path, "rb")
                #print "Adding markers to map"
                for line in fileobj:
                    #print "marker found: "+line
                    split = line.split(":");
                    if (len(split) >= 5):
                        text = split[0]
                        locX = math.trunc(float(split[1]))
                        locY = math.trunc(float(split[2]))
                        locZ = math.trunc(float(split[3]))
                        id = split[4]
                        
                        self.POI.append( dict(x=locX, y=locY, z=locZ, msg=text, id=id))
                        
                    else:
                        continue;
        except (Exception):
            print "Exception while reading markers";
            pass;
        
       
       
    
    def write_markers(self):
        """Writes out markers.js"""
                    

        with open(os.path.join(self.destdir, "markers.js"), 'w') as output:
            output.write("var markerData=%s" % json.dumps(self.POI))


        with open(os.path.join(self.destdir, "markers.json"), 'w') as output:
            output.write(json.dumps(self.POI))

            
    def go(self, procs):
        """Starts the generation of markers"""
        
        #print "Adding markers"

        self.addSpawn()
        self.addLabels()
        self.addPlayers()
        self.addHomes()
        self.write_markers()
