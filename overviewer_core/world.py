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
import logging
import collections

import numpy

import nbt
import cache

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

def log_other_exceptions(func):
    """A decorator that prints out any errors that are not ChunkDoesntExist
    errors. This decorates get_chunk because the C code is likely to swallow
    exceptions, so this will at least make them visible.

    """
    functools.wraps(func)
    def newfunc(*args):
        try:
            return func(*args)
        except ChunkDoesntExist:
            raise
        except Exception, e:
            logging.exception("%s raised this exception", func.func_name)
            raise
    return newfunc


class World(object):
    """Encapsulates the concept of a Minecraft "world". A Minecraft world is a
    level.dat file, a players directory with info about each player, a data
    directory with info about that world's maps, and one or more "dimension"
    directories containing a set of region files with the actual world data.

    This class deals with reading all the metadata about the world.  Reading
    the actual world data for each dimension from the region files is handled
    by a RegionSet object.

    Note that vanilla Minecraft servers and single player games have a single
    world with multiple dimensions: one for the overworld, the nether, etc.

    On Bukkit enabled servers, to support "multiworld," the server creates
    multiple Worlds, each with a single dimension.

    In this file, the World objects act as an interface for RegionSet objects.
    The RegionSet objects are what's really important and are used for reading
    block data for rendering.  A RegionSet object will always correspond to a
    set of region files, or what is colloquially referred to as a "world," or
    more accurately as a dimension.

    The only thing this class actually stores is a list of RegionSet objects
    and the parsed level.dat data

    """
    
    def __init__(self, worlddir):
        self.worlddir = worlddir

        # This list, populated below, will hold RegionSet files that are in
        # this world
        self.regionsets = []
       
        # The level.dat file defines a minecraft world, so assert that this
        # object corresponds to a world on disk
        if not os.path.exists(os.path.join(self.worlddir, "level.dat")):
            raise ValueError("level.dat not found in %s" % self.worlddir)

        # figure out chunk format is in use if not mcregion, error out early
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]['Data']
        if not ('version' in data and data['version'] == 19132):
            logging.critical("Sorry, This version of Minecraft-Overviewer only works with the new McRegion chunk format")
            raise ValueError("World at %s is not compatible with Overviewer" % self.worlddir)

        # This isn't much data, around 15 keys and values for vanilla worlds.
        self.leveldat = data


        # Scan worlddir to try to identify all region sets. Since different
        # server mods like to arrange regions differently and there does not
        # seem to be any set standard on what dimensions are in each world,
        # just scan the directory heirarchy to find a directory with .mcr
        # files.
        for root, dirs, files in os.walk(self.worlddir):
            # any .mcr files in this directory?
            mcrs = filter(lambda x: x.endswith(".mcr"), files)
            if mcrs:
                # construct a regionset object for this
                rset = RegionSet(root)
                if root == os.path.join(self.worlddir, "region"):
                    self.regionsets.insert(0, rset)
                else:
                    self.regionsets.append(rset)
        
        # TODO move a lot of the following code into the RegionSet


        try:
            # level.dat should have the LevelName attribute so we'll use that
            self.name = data['LevelName']
        except KeyError:
            # but very old ones might not? so we'll just go with the world dir name if they don't
            self.name = os.path.basename(os.path.realpath(self.worlddir))

       
        # TODO figure out where to handle regionlists
        
        self.useBiomeData = os.path.exists(os.path.join(worlddir, 'biomes'))
        if not self.useBiomeData:
            logging.info("Notice: Not using biome data for tinting")

    def get_regionsets(self):
        return self.regionsets
    def get_regionset(self, index):
        if type(index) == int:
            return self.regionsets[index]
        else: # assume a string constant
            if index == "default":
                return self.regionsets[0]
            else:
                candids = [x for x in self.regionsets if x.get_type() == index]
                if len(candids) > 0:
                    return candids[0]
                else: 
                    return None


    def get_level_dat_data(self):
        # Return a copy
        return dict(self.data)
      
    def find_true_spawn(self):
        """Returns the spawn point for this world. Since there is one spawn
        point for a world across all dimensions (RegionSets), this method makes
        sense as a member of the World class.
        
        Returns (x, y, z)
        
        """
        # The spawn Y coordinate is almost always the default of 64.  Find the
        # first air block above the stored spawn location for the true spawn
        # location

        ## read spawn info from level.dat
        data = self.data
        disp_spawnX = spawnX = data['SpawnX']
        spawnY = data['SpawnY']
        disp_spawnZ = spawnZ = data['SpawnZ']
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX//16
        chunkZ = spawnZ//16
        
        ## clamp spawnY to a sane value, in-chunk value
        if spawnY < 0:
            spawnY = 0
        if spawnY > 127:
            spawnY = 127
        
        # Open up the chunk that the spawn is in
        regionset = self.get_regionset(0)
        try:
            chunk = regionset.get_chunk(chunkX, chunkZ)
        except ChunkDoesntExist:
            return (spawnX, spawnY, spawnZ)

        blockArray = chunk['Blocks']

        ## The block for spawn *within* the chunk
        inChunkX = spawnX - (chunkX*16)
        inChunkZ = spawnZ - (chunkZ*16)

        ## find the first air block
        while (blockArray[inChunkX, inChunkZ, spawnY] != 0) and spawnY < 127:
            spawnY += 1

        return spawnX, spawnY, spawnZ

class RegionSet(object):
    """This object is the gateway to a particular Minecraft dimension within a
    world. It corresponds to a set of region files containing the actual
    world data. This object has methods for parsing and returning data from the
    chunks from its regions.

    See the docs for the World object for more information on the difference
    between Worlds and RegionSets.


    """

    def __init__(self, regiondir, cachesize=16):
        """Initialize a new RegionSet to access the region files in the given
        directory.

        regiondir is a path to a directory containing region files.

        cachesize, if specified, is the number of chunks to keep parsed and
        in-memory.

        """
        self.regiondir = os.path.normpath(regiondir)

        logging.info("Scanning regions")
        
        # This is populated below. It is a mapping from (x,y) region coords to filename
        self.regionfiles = {}
        
        for x, y, regionfile in self._iterate_regionfiles():
            # regionfile is a pathname
            self.regionfiles[(x,y)] = regionfile

        self.empty_chunk = [None,None]
        logging.debug("Done scanning regions")

        # Caching implementaiton: a simple LRU cache
        # Decorate the get_chunk method with the cache decorator
        self.get_chunk = cache.lru_cache(cachesize)(self.get_chunk)

    # Re-initialize upon unpickling
    def __getstate__(self):
        return self.regiondir
    __setstate__ = __init__

    def __repr__(self):
        return "<RegionSet regiondir=%r>" % self.regiondir

    def get_type(self):
        """Attempts to return a string describing the dimension represented by
        this regionset.  Either "nether", "end" or "overworld"
        """
        # path will be normalized in __init__
        if self.regiondir.endswith("/DIM-1/region"): 
            return "nether"
        elif self.regiondir.endswith("/DIM1/region"):
            return "end"
        elif self.regiondir.endswith("/region"):
            return "overworld"
        else:
            raise Exception("Woah, what kind of dimension is this! %r" % self.regiondir)

    @log_other_exceptions
    def get_chunk(self, x, z):
        """Returns a dictionary object representing the "Level" NBT Compound
        structure for a chunk given its x, z coordinates. The coordinates are
        chunk coordinates. Raises ChunkDoesntExist exception if the given chunk
        does not exist.

        The returned dictionary corresponds to the "Level" structure in the
        chunk file, with a few changes:
        * The "Blocks" byte string is transformed into a 16x16x128 numpy array
        * The "SkyLight" byte string is transformed into a 16x16x128 numpy
          array
        * The "BlockLight" byte string is transformed into a 16x16x128 numpy
          array
        * The "Data" byte string is transformed into a 16x16x128 numpy array

        Warning: the returned data may be cached and thus should not be
        modified, lest it affect the return values of future calls for the same
        chunk.
        """

        regionfile = self._get_region_path(x, z)
        if regionfile is None:
            raise ChunkDoesntExist("Chunk %s,%s doesn't exist (and neither does its region)" % (x,z))

        region = nbt.load_region(regionfile)
        data = region.load_chunk(x, z)
        region.close()
        if data is None:
            raise ChunkDoesntExist("Chunk %s,%s doesn't exist" % (x,z))

        level = data[1]['Level']
        chunk_data = level
        chunk_data['Blocks'] = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))

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
        
        # expand just like skylight
        blockdata = numpy.frombuffer(level['Data'], dtype=numpy.uint8).reshape((16,16,64))
        blockdata_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        blockdata_expanded[:,:,::2] = blockdata & 0x0F
        blockdata_expanded[:,:,1::2] = (blockdata & 0xF0) >> 4
        chunk_data['Data'] = blockdata_expanded
        
        return chunk_data      
    

    def rotate(self, north_direction):
        return RotatedRegionSet(self.regiondir, north_direction)
    
    def iterate_chunks(self):
        """Returns an iterator over all chunk metadata in this world. Iterates
        over tuples of integers (x,z,mtime) for each chunk.  Other chunk data
        is not returned here.
        
        """

        for (regionx, regiony), regionfile in self.regionfiles.iteritems():
            mcr = nbt.load_region(regionfile)
            for chunkx, chunky in mcr.get_chunks():
                yield chunkx+32*regionx, chunky+32*regiony, mcr.get_chunk_timestamp(chunkx, chunky)

    def get_chunk_mtime(self, x, z):
        """Returns a chunk's mtime, or False if the chunk does not exist.  This
        is therefore a dual purpose method. It corrects for the given north
        direction as described in the docs for get_chunk()
        
        """

        regionfile = self._get_region_path(x,z)
        if regionfile is None:
            return None

        data = nbt.load_region(regionfile)
        if data.chunk_exists(x,z):
            return data.get_chunk_timestamp(x,z)
        return None

    def _get_region_path(self, chunkX, chunkY):
        """Returns the path to the region that contains chunk (chunkX, chunkY)
        Coords can be either be global chunk coords, or local to a region

        """
        regionfile = self.regionfiles.get((chunkX//32, chunkY//32),None)
        return regionfile
            
    def _iterate_regionfiles(self):
        """Returns an iterator of all of the region files, along with their 
        coordinates

        Returns (regionx, regiony, filename)"""

        logging.debug("regiondir is %s", self.regiondir)

        for path in glob(self.regiondir + "/r.*.*.mcr"):
            dirpath, f = os.path.split(path)
            p = f.split(".")
            x = int(p[1])
            y = int(p[2])
            yield (x, y, path)
    
# see RegionSet.rotate.  These values are chosen so that they can be passed directly to rot90
UPPER_LEFT  = 0 ## - Return the world such that north is down the -Z axis (no rotation)
LOWER_LEFT  = 1 ## - Return the world such that north is down the -X axis (rotate 90 degrees counterclockwise)
LOWER_RIGHT = 2 ## - Return the world such that north is down the +Z axis (rotate 180 degrees)
UPPER_RIGHT = 3 ## - Return the world such that north is down the +X axis (rotate 90 degrees clockwise)

NO_ROTATION =               lambda x,z: (x,z)
ROTATE_CLOCKWISE =          lambda x,z: (-z,x)
ROTATE_COUNTERCLOCKWISE =   lambda x,z: (z,-x)
ROTATE_180 =                lambda x,z: (-x,-z)

# These take rotated coords and translate into un-rotated coords
unrotation_funcs = {
        UPPER_LEFT: NO_ROTATION,
        UPPER_RIGHT: ROTATE_CLOCKWISE,
        LOWER_LEFT: ROTATE_COUNTERCLOCKWISE,
        LOWER_RIGHT: ROTATE_180,
    }

# These translate un-rotated coordinates into rotated coordinates
rotation_funcs = {
        UPPER_LEFT: NO_ROTATION,
        UPPER_RIGHT: ROTATE_COUNTERCLOCKWISE,
        LOWER_LEFT: ROTATE_CLOCKWISE,
        LOWER_RIGHT: ROTATE_180,
    }

class RotatedRegionSet(RegionSet):
    """A regionset, only rotated such that north points in the given direction

    """
    def __init__(self, regiondir, north_dir):
        self.north_dir = north_dir
        self.unrotate = unrotation_funcs[north_dir]
        self.rotate = rotation_funcs[north_dir]

        super(RotatedRegionSet, self).__init__(regiondir)

    
    # Re-initialize upon unpickling
    def __getstate__(self):
        return (self.regiondir, self.north_dir)
    def __setstate__(self, args):
        self.__init__(args[0], args[1])

    def get_chunk(self, x, z):
        x,z = self.unrotate(x,z)
        chunk_data = super(RotatedRegionSet, self).get_chunk(x,z)
        chunk_data['Blocks'] = numpy.rot90(chunk_data['Blocks'], self.north_dir)
        chunk_data['Data'] = numpy.rot90(chunk_data['Data'], self.north_dir)
        chunk_data['SkyLight'] = numpy.rot90(chunk_data['SkyLight'], self.north_dir)
        chunk_data['BlockLight'] = numpy.rot90(chunk_data['BlockLight'], self.north_dir)
        return chunk_data

    def get_chunk_mtime(self, x, z):
        x,z = self.unrotate(x,z)
        return super(RotatedRegionSet, self).get_chunk_mtime(x, z)

    def iterate_chunks(self):
        for x,z,mtime in super(RotatedRegionSet, self).iterate_chunks():
            x,z = self.rotate(x,z)
            yield x,z,mtime

class ChunkDoesntExist(Exception):
    pass

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
