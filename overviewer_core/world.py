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
from glob import glob
import multiprocessing
import Queue
import sys
import logging
import cPickle
import collections
import itertools

import numpy

from chunk import ChunkCorrupt
import nbt
import textures
import time

"""
This module has routines for extracting information about available worlds

"""

base36decode = functools.partial(int, base=36)
cached = collections.defaultdict(dict)

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

class World(object):
    """Does world-level preprocessing to prepare for QuadtreeGen
    worlddir is the path to the minecraft world
    """

    mincol = maxcol = minrow = maxrow = 0
    
    def __init__(self, worlddir, outputdir, useBiomeData=False, regionlist=None, north_direction="auto"):
        self.worlddir = worlddir
        self.outputdir = outputdir
        self.useBiomeData = useBiomeData
        self.north_direction = north_direction
        
        # figure out chunk format is in use
        # if not mcregion, error out early
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]['Data']
        #print data
        if not ('version' in data and data['version'] == 19132):
            logging.error("Sorry, This version of Minecraft-Overviewer only works with the new McRegion chunk format")
            sys.exit(1)

        #  stores Points Of Interest to be mapped with markers
        #  a list of dictionaries, see below for an example
        self.POI = []

        # if it exists, open overviewer.dat, and read in the data structure
        # info self.persistentData.  This dictionary can hold any information
        # that may be needed between runs.
        # Currently only holds into about POIs (more more details, see quadtree)
        
        self.oldPickleFile = os.path.join(self.worlddir, "overviewer.dat")
        self.pickleFile = os.path.join(self.outputdir, "overviewer.dat")    
        
        if os.path.exists(self.oldPickleFile):
            logging.warning("overviewer.dat detected in WorldDir - this is no longer the correct location")
            if os.path.exists(self.pickleFile):
                # new file exists, so make a note of it
                logging.warning("you should delete the `overviewer.dat' file in your world directory")
            else:
                # new file does not exist, so move the old one
                logging.warning("Moving overviewer.dat to OutputDir")
                import shutil
                try:
                    # make sure destination dir actually exists
                    try:
                        os.mkdir(self.outputdir)
                    except OSError: # already exists, or failed
                        pass
                    shutil.move(self.oldPickleFile, self.pickleFile)
                    logging.info("overviewer.dat moved")
                except BaseException as ex:
                    logging.error("Unable to move overviewer.dat")
                    logging.debug(ex.str())

        if os.path.exists(self.pickleFile):
            self.persistentDataIsNew = False;
            with open(self.pickleFile,"rb") as p:
                self.persistentData = cPickle.load(p)
                if not self.persistentData.get('north_direction', False):
                    # this is a pre-configurable-north map, so add the north_direction key
                    self.persistentData['north_direction'] = 'lower-left'
        else:
            # some defaults, presumably a new map
            self.persistentData = dict(POI=[], north_direction='lower-left')
            self.persistentDataIsNew = True # indicates that the values in persistentData are new defaults, and it's OK to override them
        
        # handle 'auto' north
        if self.north_direction == 'auto':
            self.north_direction = self.persistentData['north_direction']
            north_direction = self.north_direction
        
        #find region files, or load the region list
        #this also caches all the region file header info
        logging.info("Scanning regions")
        regionfiles = {}
        self.regions = {}
        if regionlist:
            self.regionlist = map(os.path.abspath, regionlist) # a list of paths
        else:
            self.regionlist = None
        for x, y, regionfile in self._iterate_regionfiles(regionlist):
            mcr = self.reload_region(regionfile) 
            mcr.get_chunk_info()
            regionfiles[(x,y)]	= (x,y,regionfile,mcr)
        self.regionfiles = regionfiles	
        # set the number of region file handles we will permit open at any time before we start closing them
#        self.regionlimit = 1000
        # the max number of chunks we will keep before removing them (includes emptry chunks)
        self.chunklimit = 1024 
        self.chunkcount = 0
        self.empty_chunk = [None,None]
        logging.debug("Done scanning regions")
        
 
    def get_region_path(self, chunkX, chunkY):
        """Returns the path to the region that contains chunk (chunkX, chunkY)
        """
        _, _, regionfile,_ = self.regionfiles.get((chunkX//32, chunkY//32),(None,None,None,None));
        return regionfile
            
    def load_from_region(self,filename, x, y):
        #we need to manage the chunk cache
        regioninfo = self.regions[filename]
        if regioninfo is None:
            return None
        chunks = regioninfo[2]
        chunk_data = chunks.get((x,y))
        if chunk_data is None:        
            #prune the cache if required
            if self.chunkcount > self.chunklimit: #todo: make the emptying the chunk cache slightly less crazy
                [self.reload_region(regionfile) for regionfile in self.regions if regionfile <> filename]
                self.chunkcount = 0
            self.chunkcount += 1  

            nbt = self.load_region(filename).load_chunk(x, y)
            if nbt is None:
                chunks[(x,y)] = self.empty_chunk
                return None ## return none.  I think this is who we should indicate missing chunks
                #raise IOError("No such chunk in region: (%i, %i)" % (x, y))                 

            #we cache the transformed data, not it's raw form
            data = nbt.read_all()    
            level = data[1]['Level']
            chunk_data = level
            chunk_data['Blocks'] = numpy.array(numpy.rot90(numpy.frombuffer(
                    level['Blocks'], dtype=numpy.uint8).reshape((16,16,128)),
                    self._get_north_rotations()))
            chunk_data['Data'] = numpy.array(numpy.rot90(numpy.frombuffer(
                    level['Data'], dtype=numpy.uint8).reshape((16,16,64)),
                    self._get_north_rotations()))
            chunk_data['SkyLight'] = numpy.array(numpy.rot90(numpy.frombuffer(
                    level['SkyLight'], dtype=numpy.uint8).reshape((16,16,64)),
                    self._get_north_rotations()))
            chunk_data['BlockLight'] = numpy.array(numpy.rot90(numpy.frombuffer(
                    level['BlockLight'], dtype=numpy.uint8).reshape((16,16,64)),
                    self._get_north_rotations()))
            #chunk_data = {}
            #chunk_data['skylight'] = chunk.get_skylight_array(level)
            #chunk_data['blocklight'] = chunk.get_blocklight_array(level)
            #chunk_data['blockarray'] = chunk.get_blockdata_array(level)
            #chunk_data['TileEntities'] = chunk.get_tileentity_data(level)
            
            chunks[(x,y)] = [level,time.time()]
        else:
            chunk_data = chunk_data[0]
        return chunk_data      
      
    #used to reload a changed region
    def reload_region(self,filename):
        if self.regions.get(filename) is not None:
            self.regions[filename][0].closefile()
        chunkcache = {}    
        mcr = nbt.MCRFileReader(filename, self.north_direction)
        self.regions[filename] = (mcr,os.path.getmtime(filename),chunkcache)
        return mcr
        
    def load_region(self,filename):    
        return self.regions[filename][0]
        
    def get_region_mtime(self,filename):                  
        return (self.regions[filename][0],self.regions[filename][1])        
        
    def convert_coords(self, chunkx, chunky):
        """Takes a coordinate (chunkx, chunky) where chunkx and chunky are
        in the chunk coordinate system, and figures out the row and column
        in the image each one should be. Returns (col, row)."""
        
        # columns are determined by the sum of the chunk coords, rows are the
        # difference
        # change this function, and you MUST change unconvert_coords
        return (chunkx + chunky, chunky - chunkx)
    
    def unconvert_coords(self, col, row):
        """Undoes what convert_coords does. Returns (chunkx, chunky)."""
        
        # col + row = chunky + chunky => (col + row)/2 = chunky
        # col - row = chunkx + chunkx => (col - row)/2 = chunkx
        return ((col - row) / 2, (col + row) / 2)
    
    def findTrueSpawn(self):
        """Adds the true spawn location to self.POI.  The spawn Y coordinate
        is almost always the default of 64.  Find the first air block above
        that point for the true spawn location"""

        ## read spawn info from level.dat
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]
        disp_spawnX = spawnX = data['Data']['SpawnX']
        spawnY = data['Data']['SpawnY']
        disp_spawnZ = spawnZ = data['Data']['SpawnZ']
        if self.north_direction == 'upper-left':
            temp = spawnX
            spawnX = -spawnZ
            spawnZ = temp
        elif self.north_direction == 'upper-right':
            spawnX = -spawnX
            spawnZ = -spawnZ
        elif self.north_direction == 'lower-right':
            temp = spawnX
            spawnX = spawnZ
            spawnZ = -temp
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX/16
        chunkY = spawnZ/16

        try:
            ## The filename of this chunk
            chunkFile = self.get_region_path(chunkX, chunkY)
            if chunkFile is not None:
                data = nbt.load_from_region(chunkFile, chunkX, chunkY, self.north_direction)
                if data is not None:
                    level = data[1]['Level']
                    blockArray = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))
                
                    ## The block for spawn *within* the chunk
                    inChunkX = spawnX - (chunkX*16)
                    inChunkZ = spawnZ - (chunkY*16)
                
                    ## find the first air block
                    while (blockArray[inChunkX, inChunkZ, spawnY] != 0):
                        spawnY += 1
                        if spawnY == 128:
                            break
        except ChunkCorrupt:
            #ignore corrupt spawn, and continue
            pass
        self.POI.append( dict(x=disp_spawnX, y=spawnY, z=disp_spawnZ,
                msg="Spawn", type="spawn", chunk=(chunkX, chunkY)))
        self.spawn = (disp_spawnX, spawnY, disp_spawnZ)

    def go(self, procs):
        """Scan the world directory, to fill in
        self.{min,max}{col,row} for use later in quadtree.py. This
        also does other world-level processing."""
        
        logging.info("Scanning chunks")
        # find the dimensions of the map, in region files
        minx = maxx = miny = maxy = 0
        found_regions = False
        for x, y in self.regionfiles:
            found_regions = True
            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)
        if not found_regions:
            logging.error("Error: No chunks found!")
            sys.exit(1)
        logging.debug("Done scanning chunks")
        
        # turn our region coordinates into chunk coordinates
        minx = minx * 32
        miny = miny * 32
        maxx = maxx * 32 + 32
        maxy = maxy * 32 + 32
        
        # Translate chunks to our diagonal coordinate system
        mincol = maxcol = minrow = maxrow = 0
        for chunkx, chunky in [(minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]:
            col, row = self.convert_coords(chunkx, chunky)
            mincol = min(mincol, col)
            maxcol = max(maxcol, col)
            minrow = min(minrow, row)
            maxrow = max(maxrow, row)
        
        #logging.debug("map size: (%i, %i) to (%i, %i)" % (mincol, minrow, maxcol, maxrow))

        self.mincol = mincol
        self.maxcol = maxcol
        self.minrow = minrow
        self.maxrow = maxrow

        self.findTrueSpawn()

    def _get_north_rotations(self):
        if self.north_direction == 'upper-left':
            return 1
        elif self.north_direction == 'upper-right':
            return 2
        elif self.north_direction == 'lower-right':
            return 3
        elif self.north_direction == 'lower-left':
            return 0

    def _iterate_regionfiles(self,regionlist=None):
        """Returns an iterator of all of the region files, along with their 
        coordinates

        Note: the regionlist here will be used to determinte the size of the
        world. 

        Returns (regionx, regiony, filename)"""
        join = os.path.join
        if regionlist is not None:
            for path in regionlist:
                path = path.strip()
                f = os.path.basename(path)
                if f.startswith("r.") and f.endswith(".mcr"):
                    p = f.split(".")
                    logging.debug("Using path %s from regionlist", f)
                    x = int(p[1])
                    y = int(p[2])
                    if self.north_direction == 'upper-left':
                        temp = x
                        x = -y-1
                        y = temp
                    elif self.north_direction == 'upper-right':
                        x = -x-1
                        y = -y-1
                    elif self.north_direction == 'lower-right':
                        temp = x
                        x = y
                        y = -temp-1
                    yield (x, y, join(self.worlddir, 'region', f))
                else:
                    logging.warning("Ignore path '%s' in regionlist", f)

        else:                    
            for path in glob(os.path.join(self.worlddir, 'region') + "/r.*.*.mcr"):
                dirpath, f = os.path.split(path)
                p = f.split(".")
                x = int(p[1])
                y = int(p[2])
                if self.north_direction == 'upper-left':
                    temp = x
                    x = -y-1
                    y = temp
                elif self.north_direction == 'upper-right':
                    x = -x-1
                    y = -y-1
                elif self.north_direction == 'lower-right':
                    temp = x
                    x = y
                    y = -temp-1
                yield (x, y, join(dirpath, f))

def get_save_dir():
    """Returns the path to the local saves directory
      * On Windows, at %APPDATA%/.minecraft/saves/
      * On Darwin, at $HOME/Library/Application Support/minecraft/saves/
      * at $HOME/.minecraft/saves/

    """
    
    savepaths = []
    if "APPDATA" in os.environ:
        savepaths += [os.path.join(os.environ['APPDATA'], ".minecraft", "saves")]
    if "HOME" in os.environ:
        savepaths += [os.path.join(os.environ['HOME'], "Library",
                "Application Support", "minecraft", "saves")]
        savepaths += [os.path.join(os.environ['HOME'], ".minecraft", "saves")]

    for path in savepaths:
        if os.path.exists(path):
            return path

def get_worlds():
    "Returns {world # or name : level.dat information}"
    ret = {}
    save_dir = get_save_dir()

    # No dirs found - most likely not running from inside minecraft-dir
    if save_dir is None:
        return None

    for dir in os.listdir(save_dir):
        world_dat = os.path.join(save_dir, dir, "level.dat")
        if not os.path.exists(world_dat): continue
        info = nbt.load(world_dat)[1]
        info['Data']['path'] = os.path.join(save_dir, dir)
        if dir.startswith("World") and len(dir) == 6:
            try:
                world_n = int(dir[-1])
                ret[world_n] = info['Data']
            except ValueError:
                pass
        if 'LevelName' in info['Data'].keys():
            ret[info['Data']['LevelName']] = info['Data']

    return ret
