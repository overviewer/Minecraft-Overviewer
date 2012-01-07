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
import time

import numpy

import nbt
import textures
import util

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

    # see RegionSet.rotate.  These values are chosen so that they can be passed directly to rot90
    UPPER_LEFT  = 1 ## - Return the world such that north is down the -Z axis (no rotation)
    UPPER_RIGHT = 2 ## - Return the world such that north is down the +X axis (rotate 90 degrees clockwise)
    LOWER_RIGHT = 3 ## - Return the world such that north is down the +Z axis (rotate 180 degrees)
    LOWER_LEFT  = 0 ## - Return the world such that north is down the -X axis (rotate 90 degrees counterclockwise)
    
    def __init__(self, worlddir):
        self.worlddir = worlddir

        self.regionsets = []
       
        # Scan worlddir to try to identify all region sets
        if not os.path.exists(os.path.join(self.worlddir, "level.dat")):
            raise Exception("level.dat not found in %s" % self.worlddir)

        for root, dirs, files in os.walk(self.worlddir):
            # any .mcr files in this directory?
            mcrs = filter(lambda x: x.endswith(".mcr"), files)
            if mcrs:
                # construct a regionset object for this
                rset = RegionSet(self, root)
                if root == os.path.join(self.worlddir, "region"):
                    self.regionsets.insert(0, rset)
                else:
                    self.regionsets.append(rset)
        
        # TODO consider reordering self.regionsets so that the 'default' region is first

        # TODO move a lot of the following code into the RegionSet


        # figure out chunk format is in use
        # if not mcregion, error out early
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]['Data']
        if not ('version' in data and data['version'] == 19132):
            logging.error("Sorry, This version of Minecraft-Overviewer only works with the new McRegion chunk format")
            sys.exit(1)

        # TODO move levelname into the regionsets?
        if 'LevelName' in data:
            # level.dat should have the LevelName attribute so we'll use that
            self.name = data['LevelName']
        else:
            # but very old ones might not? so we'll just go with the world dir name if they don't
            self.name = os.path.basename(os.path.realpath(self.worlddir))

       
        # TODO figure out where to handle regionlists
        
        self.useBiomeData = os.path.exists(os.path.join(worlddir, 'biomes'))
        if not self.useBiomeData:
            logging.info("Notice: Not using biome data for tinting")

    def get_level_dat_data(self):
        """Returns a dictionary representing the level.dat data for this World"""
        return nbt.load(os.path.join(self.worlddir, "level.dat"))
 
    def get_regionsets(self):
        return self.regionsets
    def get_regionset(self, index):
        return self.regionsets[index]
      
        
        
    def get_region_mtime(self,filename):                  
        return (self.regions[filename][0],self.regions[filename][1])        
    
    def find_true_spawn(self):
        """Adds the true spawn location to self.POI.  The spawn Y coordinate
        is almost always the default of 64.  Find the first air block above
        that point for the true spawn location"""

        ## read spawn info from level.dat
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]
        disp_spawnX = spawnX = data['Data']['SpawnX']
        spawnY = data['Data']['SpawnY']
        disp_spawnZ = spawnZ = data['Data']['SpawnZ']
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX/16
        chunkY = spawnZ/16
        
        ## clamp spawnY to a sane value, in-chunk value
        if spawnY < 0:
            spawnY = 0
        if spawnY > 127:
            spawnY = 127
        
        ## The filename of this chunk
        chunkFile = self.get_region_path(chunkX, chunkY)
        if chunkFile is not None:
            data = nbt.load_from_region(chunkFile, chunkX, chunkY)
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
        self.POI.append( dict(x=disp_spawnX, y=spawnY, z=disp_spawnZ,
                msg="Spawn", type="spawn", chunk=(chunkX, chunkY)))
        self.spawn = (disp_spawnX, spawnY, disp_spawnZ)


    def _get_north_rotations(self):
        # default to lower-left for now
        return 0



class RegionSet(object):
    """\
This object is the gateway to a set of regions (or dimension) from the world
we're reading from. There is one of these per set of regions on the hard drive,
but may be several per invocation of the Overviewer in the case of multi-world.
    """

    def __init__(self, regiondir):
        #self.world = worldobj
        self.regiondir = regiondir

        logging.info("Scanning regions")
        
        # This is populated below. It is a mapping from (x,y) region coords to filename
        self.regionfiles = {}
        
        for x, y, regionfile in self._iterate_regionfiles():
            # regionfile is a pathname
            self.regionfiles[(x,y)] = regionfile

        self.empty_chunk = [None,None]
        logging.debug("Done scanning regions")

    def __repr__(self):
        return "<RegionSet regiondir=%r>" % self.regiondir

    def get_region_path(self, chunkX, chunkY):
        """Returns the path to the region that contains chunk (chunkX, chunkY)
        Coords can be either be global chunk coords, or local to a region
        """
        regionfile = self.regionfiles.get((chunkX//32, chunkY//32),None)
        return regionfile
            
    def get_chunk(self,x, z):
        """Returns a dictionary representing the top-level NBT Compound for a chunk given
        its x, z coordinates. The coordinates are chunk coordinates.
        """

        regionfile = self.get_region_path(x, z)
        if regionfile is None:
            return None

        region = nbt.load_region(regionfile)
        data = region.load_chunk(x, z)
        region.close()
        if data is None:
            return None

        level = data[1]['Level']
        chunk_data = level
        chunk_data['Blocks'] = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))
        chunk_data['Data'] = numpy.frombuffer(level['Data'], dtype=numpy.uint8).reshape((16,16,64))


        skylight = numpy.frombuffer(level['SkyLight'], dtype=numpy.uint8).reshape((16,16,64))

        # this array is 2 blocks per byte, so expand it
        skylight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        # Even elements get the lower 4 bits
        skylight_expanded[:,:,::2] = skylight & 0x0F
        # Odd elements get the upper 4 bits
        skylight_expanded[:,:,1::2] = (skylight & 0xF0) >> 4
        chunk_data['SkyLight'] = skylight_expanded

        # expand just like skylight
        blocklight = numpy.frombuffer(level['BlockLight'], dtype=numpy.uint8).reshape((16,16,64))
        blocklight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        blocklight_expanded[:,:,::2] = blocklight & 0x0F
        blocklight_expanded[:,:,1::2] = (blocklight & 0xF0) >> 4
        chunk_data['BlockLight'] = blocklight_expanded
        
        #chunk_data = {}
        #chunk_data['skylight'] = chunk.get_skylight_array(level)
        #chunk_data['blocklight'] = chunk.get_blocklight_array(level)
        #chunk_data['blockarray'] = chunk.get_blockdata_array(level)
        #chunk_data['TileEntities'] = chunk.get_tileentity_data(level)
        
        return chunk_data      
    

    def rotate(self, north_direction):
        return RotatedRegionSet(self.worldobj, self.regiondir, north_direction)
    
    def iterate_chunks(self):
        """Returns an iterator over all chunk metadata in this world. Iterates over tuples
        of integers (x,z,mtime) for each chunk.  Other chunk data is not returned here.
        
        Old name: world.iterate_chunk_metadata
        """

        for (regionx, regiony), regionfile in self.regionfiles.iteritems():
            mcr = nbt.load_region(regionfile)
            for chunkx, chunky in mcr.get_chunks():
                yield chunkx+32*regionx, chunky+32*regiony, mcr.get_chunk_timestamp(chunkx, chunky)

    def get_chunk_mtime(self, x, z):
        """Returns a chunk's mtime, or False if the chunk does not exist.
        This is therefore a dual purpose method. It corrects for the given north
        direction as described in the docs for get_chunk()"""

        regionfile = self.get_region_path(x,z)
        if regionfile is None:
            return None

        data = nbt.load_region(regionfile)
        if data.chunk_exists(x,z):
            return data.get_chunk_timestamp(x,z)
        return None


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
                    logging.warning("Ignoring non region file '%s' in regionlist", f)

        else:                    
            print "regiondir is", self.regiondir
            for path in glob(self.regiondir + "/r.*.*.mcr"):
                dirpath, f = os.path.split(path)
                p = f.split(".")
                x = int(p[1])
                y = int(p[2])
                ##TODO if self.north_direction == 'upper-left':
                ##TODO     temp = x
                ##TODO     x = -y-1
                ##TODO     y = temp
                ##TODO elif self.north_direction == 'upper-right':
                ##TODO     x = -x-1
                ##TODO     y = -y-1
                ##TODO elif self.north_direction == 'lower-right':
                ##TODO     temp = x
                ##TODO     x = y
                ##TODO     y = -temp-1
                yield (x, y, join(dirpath, f))
    

class RotatedRegionSet(RegionSet):
    def __init__(self, worldobj, regiondir, north_dir):
        super(RotatedRegionSet, self).__init__(worldobj, regiondir)
        self.north_dir = north_dir
    def get_chunk(self, x, z):
        chunk_data = super(RotatedRegionSet, self).get_chunk(x,z)
        chunk_data['Blocks'] = numpy.rot90(chunk_data['Blocks'], self.north_dir)
        chunk_data['Data'] = numpy.rot90(chunk_data['Data'], self.north_dir)
        chunk_data['SkyLight'] = numpy.rot90(chunk_data['SkyLight'], self.north_dir)
        chunk_data['BlockLight'] = numpy.rot90(chunk_data['BlockLight'], self.north_dir)
        return chunk_data
         

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

