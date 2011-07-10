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

import chunk
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
    
    def __init__(self, worlddir, useBiomeData=False,regionlist=None):
        self.worlddir = worlddir
        self.useBiomeData = useBiomeData
                
        #find region files, or load the region list
        #this also caches all the region file header info
        logging.info("Scanning regions")
        regionfiles = {}
        self.regions = {}
        if regionlist:
            self.regionlist = map(os.path.abspath, regionlist) # a list of paths
        else:
            self.regionlist = None
        for x, y, regionfile in self._iterate_regionfiles():
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
        # TODO maybe store this with the tiles, not with the world?
        self.pickleFile = os.path.join(self.worlddir, "overviewer.dat")
        if os.path.exists(self.pickleFile):
            with open(self.pickleFile,"rb") as p:
                self.persistentData = cPickle.load(p)
        else:
            # some defaults
            self.persistentData = dict(POI=[])
        
 
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
        mcr = nbt.MCRFileReader(filename)
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
        # difference (TODO: be able to change direction of north)
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
        spawnX = data['Data']['SpawnX']
        spawnY = data['Data']['SpawnY']
        spawnZ = data['Data']['SpawnZ']
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX/16
        chunkY = spawnZ/16

        ## The filename of this chunk
        chunkFile = self.get_region_path(chunkX, chunkY)
        
        if chunkFile is not None:
            data = nbt.load_from_region(chunkFile, chunkX, chunkY)[1]
            if data is not None:
                level = data['Level']
                blockArray = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))
                
                ## The block for spawn *within* the chunk
                inChunkX = spawnX - (chunkX*16)
                inChunkZ = spawnZ - (chunkY*16)
                
                ## find the first air block
                while (blockArray[inChunkX, inChunkZ, spawnY] != 0):
                    spawnY += 1
                    if spawnY == 128:
                        break

        self.POI.append( dict(x=spawnX, y=spawnY, z=spawnZ, 
                msg="Spawn", type="spawn", chunk=(chunkX, chunkY)))
        self.spawn = (spawnX, spawnY, spawnZ)

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
                    yield (int(p[1]), int(p[2]), join(self.worlddir, 'region', f))        
                else:
                    logging.warning("Ignore path '%s' in regionlist", f)

        else:                    
            for path in glob(os.path.join(self.worlddir, 'region') + "/r.*.*.mcr"):
                dirpath, f = os.path.split(path)
                p = f.split(".")
                yield (int(p[1]), int(p[2]), join(dirpath, f))

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
