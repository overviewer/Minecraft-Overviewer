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

base36decode = functools.partial(int, base=36)

def _convert_coords(chunks):
    """Takes the list of (chunkx, chunky, chunkfile) where chunkx and chunky
    are in the chunk coordinate system, and figures out the row and column in
    the image each one should be.

    returns mincol, maxcol, minrow, maxrow, chunks_translated
    chunks_translated is a list of (col, row, filename)
    """
    chunks_translated = []
    # columns are determined by the sum of the chunk coords, rows are the
    # difference
    item = chunks[0]
    mincol = maxcol = item[0] + item[1]
    minrow = maxrow = item[1] - item[0]
    for c in chunks:
        col = c[0] + c[1]
        mincol = min(mincol, col)
        maxcol = max(maxcol, col)
        row = c[1] - c[0]
        minrow = min(minrow, row)
        maxrow = max(maxrow, row)
        chunks_translated.append((col, row, c[2]))

    return mincol, maxcol, minrow, maxrow, chunks_translated


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
    
    
def catch_keyboardinterrupt(func):
    """Decorator that catches a keyboardinterrupt and raises a real exception
    so that multiprocessing will propagate it properly"""
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print "Ctrl-C caught!"
            raise Exception("Exiting")
        except:
            import traceback
            traceback.print_exc()
            raise
    return newfunc

@catch_keyboardinterrupt
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
        spawnY = data['Data']['SpawnY'] ## REMEMBER! Z-level is swapped with Ys
        spawnZ = data['Data']['SpawnZ']
        
        self.addMarker(spawnX, spawnY, spawnZ, "Spawn")

    def addLabels(self):
        """Adds the labels from the server to self.POI."""

        ## read label info from mapper-labels.txt
        ## TODO! for each line, extract text:x:Z:y

        path = os.path.join(self.worlddir, "mapper-labels.txt")
        
        #try:
        if os.path.exists(path):
            fileobj = open(path, "rb")
            print "Adding markers to map"
            for line in fileobj:
                print "marker found: "+line
                split = line.split(":");
                if (len(split) == 5):
                    text = split[0]
                    locX = math.trunc(float(split[1]))
                    locY = math.trunc(float(split[2]))
                    locZ = math.trunc(float(split[3]))
                    
                    self.addMarker(locX, locY, locZ, text)
                    
                else:
                    continue;
        #except (IOError):
         #   print "Exception while reading label";
         #   pass;

        

  

    def addMarker(self, locX, locY, locZ, text):
        """The spawn Y coordinate is almost always the
        default of 64.  Find the first air block above
        that point for the true spawn location"""

        
        ## The chunk that holds the spawn location 
        chunkX = locX/16
        chunkZ = locZ/16

        ## The filename of this chunk
        chunkFile = "%s/%s/c.%s.%s.dat" % (base36encode(chunkX % 64), 
                                           base36encode(chunkZ % 64),
                                           base36encode(chunkX),
                                           base36encode(chunkZ))


        data=nbt.load(os.path.join(self.worlddir, chunkFile))[1]
        level = data['Level']
        blockArray = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))

        ## The block for spawn *within* the chunk
        inChunkX = locX - (chunkX*16)
        inChunkZ = locZ - (chunkZ*16)

        ## find the first air block
        while (blockArray[inChunkX, inChunkZ, locY] != 0):
            locY += 1
       

        self.POI.append( dict(x=locX, y=locY, z=locZ, msg=text))
       
       
    
    def write_markers(self):
        """Writes out markers.js"""
                    

        with open(os.path.join(self.destdir, "markers.js"), 'w') as output:
            output.write("var markerData=%s" % json.dumps(self.POI))


    def go(self, procs):
        """Starts the generation of markers"""
        
        print "Adding markers"

        self.addSpawn()
        self.addLabels()
        self.write_markers()
