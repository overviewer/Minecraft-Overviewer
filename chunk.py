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

import numpy
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import os.path
import hashlib
import logging
import time
import math
import sys

import nbt
import textures
import world
import composite

"""
This module has routines related to rendering one particular chunk into an
image

"""

# General note about pasting transparent image objects onto an image with an
# alpha channel:
# If you use the image as its own mask, it will work fine only if the alpha
# channel is binary. If there's any translucent parts, then the alpha channel
# of the dest image will have its alpha channel modified. To prevent this:
# first use im.split() and take the third item which is the alpha channel and
# use that as the mask. Then take the image and use im.convert("RGB") to strip
# the image from its alpha channel, and use that as the source to alpha_over()

# (note that this workaround is NOT technically needed when using the
# alpha_over extension, BUT this extension may fall back to PIL's
# paste(), which DOES need the workaround.)

def get_lvldata(filename, x, y):
    """Takes a filename and chunkcoords and returns the Level struct, which contains all the
    level info"""

    d =  nbt.load_from_region(filename, x, y)
    if not d: raise NoSuchChunk(x,y)
    return d[1]['Level']

def get_blockarray(level):
    """Takes the level struct as returned from get_lvldata, and returns the
    Block array, which just contains all the block ids"""
    return numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))

def get_blockarray_fromfile(filename):
    """Same as get_blockarray except takes a filename and uses get_lvldata to
    open it. This is a shortcut"""
    level = get_lvldata(filename)
    return get_blockarray(level)

def get_skylight_array(level):
    """Returns the skylight array. This is 4 bits per block, but it is
    expanded for you so you may index it normally."""
    skylight = numpy.frombuffer(level['SkyLight'], dtype=numpy.uint8).reshape((16,16,64))
    # this array is 2 blocks per byte, so expand it
    skylight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
    # Even elements get the lower 4 bits
    skylight_expanded[:,:,::2] = skylight & 0x0F
    # Odd elements get the upper 4 bits
    skylight_expanded[:,:,1::2] = (skylight & 0xF0) >> 4
    return skylight_expanded

def get_blocklight_array(level):
    """Returns the blocklight array. This is 4 bits per block, but it
    is expanded for you so you may index it normally."""
    # expand just like get_skylight_array()
    blocklight = numpy.frombuffer(level['BlockLight'], dtype=numpy.uint8).reshape((16,16,64))
    blocklight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
    blocklight_expanded[:,:,::2] = blocklight & 0x0F
    blocklight_expanded[:,:,1::2] = (blocklight & 0xF0) >> 4
    return blocklight_expanded

def get_blockdata_array(level):
    """Returns the ancillary data from the 'Data' byte array.  Data is packed
    in a similar manner to skylight data"""
    return numpy.frombuffer(level['Data'], dtype=numpy.uint8).reshape((16,16,64))

def get_tileentity_data(level):
    """Returns the TileEntities TAG_List from chunk dat file"""
    data = level['TileEntities']
    return data

def iterate_chunkblocks(xoff,yoff):
    """Iterates over the 16x16x128 blocks of a chunk in rendering order.
    Yields (x,y,z,imgx,imgy)
    x,y,z is the block coordinate in the chunk
    imgx,imgy is the image offset in the chunk image where that block should go
    """
    for x in xrange(15,-1,-1):
        for y in xrange(16):
            imgx = xoff + x*12 + y*12
            imgy = yoff - x*6 + y*6 + 128*12 + 16*12//2
            for z in xrange(128):
                yield x,y,z,imgx,imgy
                imgy -= 12


# This set holds blocks ids that can be seen through, for occlusion calculations
transparent_blocks = set([0, 6, 8, 9, 18, 20, 37, 38, 39, 40, 44, 50, 51, 52, 53, 55,
    59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 83, 85])

# This set holds block ids that are solid blocks
solid_blocks = set([1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 35, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 56, 57, 58, 60, 
    61, 62, 64, 65, 66, 67, 71, 73, 74, 78, 79, 80, 81, 82, 84, 86, 87, 88, 89, 91])

# This set holds block ids that are fluid blocks
fluid_blocks = set([8,9,10,11])

# This set holds block ids that are not candidates for spawning mobs on
# (glass, half blocks)
nospawn_blocks = set([20,44])

def find_oldimage(chunkXY, cached, cave):
    blockid = "%d.%d" % chunkXY

    # Get the name of the existing image.
    dir1 = world.base36encode(chunkXY[0]%64)
    dir2 = world.base36encode(chunkXY[1]%64)
    cachename = '/'.join((dir1, dir2))

    oldimg = oldimg_path = None
    key = ".".join((blockid, "cave" if cave else "nocave"))
    if key in cached[cachename]:
         oldimg_path = cached[cachename][key]
         _, oldimg = os.path.split(oldimg_path)
         #logging.debug("Found cached image {0}".format(oldimg))
    return oldimg, oldimg_path

def check_cache(world, chunkXY, oldimg):
    """Returns True is oldimg is OK to use (i.e. not stale)"""
# TODO read to the region file and get the timestamp??
# TODO currently, just use the mtime on the region file
# TODO (which will cause a single chunk update to invalidate everything in the region

    chunkfile = os.path.join(world.worlddir, "region", "r.%d.%d.mcr" % (chunkXY[0]%64, chunkXY[1]%64))
    #logging.debug("checking cache  %s against %s", chunkfile, oldimg[1])
    try:
        if oldimg[1] and os.path.getmtime(chunkfile) <= os.path.getmtime(oldimg[1]):
            return True
        return False
    except OSError:
        return False

# chunkcoords should be the coordinates of a possible chunk.  it may not exist
def render_and_save(chunkcoords, cachedir, worldobj, oldimg, cave=False, queue=None):
    """Used as the entry point for the multiprocessing workers (since processes
    can't target bound methods) or to easily render and save one chunk

    chunkcoords is a tuple:  (chunkX, chunkY)

    If the chunk doesn't exist, return None. 
    Else, returns the image file location"""
    a = ChunkRenderer(chunkcoords, cachedir, worldobj, oldimg, queue)
    try:
        return a.render_and_save(cave)
    except ChunkCorrupt:
        # This should be non-fatal, but should print a warning
        pass
    except Exception, e:
        import traceback
        traceback.print_exc()
        raise
    except KeyboardInterrupt:
        print
        print "You pressed Ctrl-C. Exiting..."
        # Raise an exception that is an instance of Exception. Unlike
        # KeyboardInterrupt, this will re-raise in the parent, killing the
        # entire program, instead of this process dying and the parent waiting
        # forever for it to finish.
        raise Exception()

class ChunkCorrupt(Exception):
    pass

class NoSuchChunk(Exception):
    pass

class ChunkRenderer(object):
    def __init__(self, chunkcoords, cachedir, worldobj, oldimg, queue):
        """Make a new chunk renderer for the given chunk coordinates.
        chunkcoors should be a tuple: (chunkX, chunkY)
        
        cachedir is a directory to save the resulting chunk images to
        """
        self.queue = queue
        # derive based on worlddir and chunkcoords
        self.regionfile = os.path.join(worldobj.worlddir, "region",
                "r.%d.%d.mcr" % (chunkcoords[0] // 32, chunkcoords[1]//32))
    
        if not os.path.exists(self.regionfile):
            raise ValueError("Could not find regionfile: %s" % self.regionfile)

        ## TODO TODO all of this class

        #destdir, filename = os.path.split(self.chunkfile)
        #filename_split = filename.split(".")
        #chunkcoords = filename_split[1:3]
        
        #self.coords = map(world.base36decode, chunkcoords)
        self.blockid = "%d.%d" % chunkcoords

        # chunk coordinates (useful to converting local block coords to 
        # global block coords)
        self.chunkX = chunkcoords[0]
        self.chunkY = chunkcoords[1]



        self.world = worldobj


        # Cachedir here is the base directory of the caches. We need to go 2
        # levels deeper according to the chunk file. Get the last 2 components
        # of destdir and use that
        ##moredirs, dir2 = os.path.split(destdir)
        ##_, dir1 = os.path.split(moredirs)
        self.cachedir = os.path.join(cachedir, 
                world.base36encode(self.chunkX%64), 
                world.base36encode(self.chunkY%64))

        #logging.debug("cache location for this chunk: %s", self.cachedir)
        self.oldimg, self.oldimg_path = oldimg


        if self.world.useBiomeData:
            # make sure we've at least *tried* to load the color arrays in this process...
            textures.prepareBiomeData(self.world.worlddir)
            if not textures.grasscolor or not textures.foliagecolor:
                raise Exception("Can't find grasscolor.png or foliagecolor.png")


        if not os.path.exists(self.cachedir):
            try:
                os.makedirs(self.cachedir)
            except OSError, e:
                import errno
                if e.errno != errno.EEXIST:
                    raise

    def _load_level(self):
        """Loads and returns the level structure"""
        if not hasattr(self, "_level"):
            try:
                self._level = get_lvldata(self.regionfile, self.chunkX, self.chunkY)
            except NoSuchChunk, e:
                #logging.debug("Skipping non-existant chunk")
                raise
            except Exception, e:
                logging.warning("Error opening chunk file %s. It may be corrupt. %s", self.regionfile, e)
                raise ChunkCorrupt(str(e))
        return self._level
    level = property(_load_level)
        
    def _load_blocks(self):
        """Loads and returns the block array"""
        if not hasattr(self, "_blocks"):
            self._blocks = get_blockarray(self._load_level())
        return self._blocks
    blocks = property(_load_blocks)
    
    def _load_skylight(self):
        """Loads and returns skylight array"""
        if not hasattr(self, "_skylight"):
            self._skylight = get_skylight_array(self.level)
        return self._skylight
    skylight = property(_load_skylight)

    def _load_blocklight(self):
        """Loads and returns blocklight array"""
        if not hasattr(self, "_blocklight"):
            self._blocklight = get_blocklight_array(self.level)
        return self._blocklight
    blocklight = property(_load_blocklight)
    
    def _load_left(self):
        """Loads and sets data from lower-left chunk"""
        chunk_path = self.world.get_region_path(self.chunkX - 1, self.chunkY)
        try:
            chunk_data = get_lvldata(chunk_path, self.chunkX - 1, self.chunkY)
            self._left_skylight = get_skylight_array(chunk_data)
            self._left_blocklight = get_blocklight_array(chunk_data)
            self._left_blocks = get_blockarray(chunk_data)
        except NoSuchChunk:
            self._left_skylight = None
            self._left_blocklight = None
            self._left_blocks = None
    
    def _load_left_blocks(self):
        """Loads and returns lower-left block array"""
        if not hasattr(self, "_left_blocks"):
            self._load_left()
        return self._left_blocks
    left_blocks = property(_load_left_blocks)

    def _load_left_skylight(self):
        """Loads and returns lower-left skylight array"""
        if not hasattr(self, "_left_skylight"):
            self._load_left()
        return self._left_skylight
    left_skylight = property(_load_left_skylight)

    def _load_left_blocklight(self):
        """Loads and returns lower-left blocklight array"""
        if not hasattr(self, "_left_blocklight"):
            self._load_left()
        return self._left_blocklight
    left_blocklight = property(_load_left_blocklight)

    def _load_right(self):
        """Loads and sets data from lower-right chunk"""
        chunk_path = self.world.get_region_path(self.chunkX, self.chunkY + 1)
        try:
            chunk_data = get_lvldata(chunk_path, self.chunkX, self.chunkY + 1)
            self._right_skylight = get_skylight_array(chunk_data)
            self._right_blocklight = get_blocklight_array(chunk_data)
            self._right_blocks = get_blockarray(chunk_data)
        except NoSuchChunk:
            self._right_skylight = None
            self._right_blocklight = None
            self._right_blocks = None
    
    def _load_right_blocks(self):
        """Loads and returns lower-right block array"""
        if not hasattr(self, "_right_blocks"):
            self._load_right()
        return self._right_blocks
    right_blocks = property(_load_right_blocks)

    def _load_right_skylight(self):
        """Loads and returns lower-right skylight array"""
        if not hasattr(self, "_right_skylight"):
            self._load_right()
        return self._right_skylight
    right_skylight = property(_load_right_skylight)

    def _load_right_blocklight(self):
        """Loads and returns lower-right blocklight array"""
        if not hasattr(self, "_right_blocklight"):
            self._load_right()
        return self._right_blocklight
    right_blocklight = property(_load_right_blocklight)

    def _load_up_right(self):
        """Loads and sets data from upper-right chunk"""
        chunk_path = self.world.get_region_path(self.chunkX + 1, self.chunkY)
        try:
            chunk_data = get_lvldata(chunk_path, self.chunkX + 1, self.chunkY)
            self._up_right_skylight = get_skylight_array(chunk_data)
            self._up_right_blocklight = get_blocklight_array(chunk_data)
            self._up_right_blocks = get_blockarray(chunk_data)
        except IOError:
            self._up_right_skylight = None
            self._up_right_blocklight = None
            self._up_right_blocks = None
    
    def _load_up_right_blocks(self):
        """Loads and returns upper-right block array"""
        if not hasattr(self, "_up_right_blocks"):
            self._load_up_right()
        return self._up_right_blocks
    up_right_blocks = property(_load_up_right_blocks)
    
    def _load_up_left(self):
        """Loads and sets data from upper-left chunk"""
        chunk_path = self.world.get_region_path(self.chunkX, self.chunkY - 1)
        try:
            chunk_data = get_lvldata(chunk_path, self.chunkX, self.chunkY - 1)
            self._up_left_skylight = get_skylight_array(chunk_data)
            self._up_left_blocklight = get_blocklight_array(chunk_data)
            self._up_left_blocks = get_blockarray(chunk_data)
        except IOError:
            self._up_left_skylight = None
            self._up_left_blocklight = None
            self._up_left_blocks = None
    
    def _load_up_left_blocks(self):
        """Loads and returns lower-left block array"""
        if not hasattr(self, "_up_left_blocks"):
            self._load_up_left()
        return self._up_left_blocks
    up_left_blocks = property(_load_up_left_blocks)

    def generate_pseudo_ancildata(self,x,y,z,blockid, north_position = 0 ):
        """ Generates a pseudo ancillary data for blocks that depend of 
        what are surrounded and don't have ancillary data
        
        This uses a binary number of 4 digits to encode the info. 
        The encode is:
        
        Bit:   1   2   3   4
        Side:  x   y  -x  -y
        Values: bit = 0 -> The corresponding side block has different blockid
                bit = 1 -> The corresponding side block has same blockid
        Example: if the bit1 is 1 that means that there is a block with 
        blockid in the side of the +x direction.
        
        You can rotate the pseudo data multiplying by 2 and
        if it is > 15 subtracting 15 and adding 1. (moving bits
        in the left direction is like rotate 90 degree in anticlockwise
        direction). In this way can be used for maps with other
        north orientation.
        
        North position can have the values 0, 1, 2, 3, corresponding to 
        north in bottom-left, bottom-right, top-right and top-left of
        the screen.
        
        The rotation feature is not used anywhere yet.
        """
        
        blocks = self.blocks
        up_left_blocks = self.up_left_blocks 
        up_right_blocks = self.up_right_blocks
        left_blocks = self.left_blocks
        right_blocks = self.right_blocks

        pseudo_data = 0

        # first check if we are in the border of a chunk, next check for chunks adjacent to this 
        # and finally check for a block with same blockid. I we aren't in the border of a chunk, 
        # check for the block having the sme blockid.
        
        if (up_right_blocks != None and up_right_blocks[0,y,z] == blockid) if x == 15 else blocks[x+1,y,z] == blockid:
            pseudo_data = pseudo_data | 0b1000
        
        if (right_blocks != None and right_blocks[x,0,z] == blockid) if y == 15 else blocks[x,y + 1,z] == blockid:
            pseudo_data = pseudo_data | 0b0100
        
        if (left_blocks != None and left_blocks[15,y,z] == blockid) if x == 0 else blocks[x - 1,y,z] == blockid:
            pseudo_data = pseudo_data | 0b0010
            
        if (up_left_blocks != None and up_left_blocks[x,15,z] == blockid) if y == 0 else blocks[x,y - 1,z] == blockid:
            pseudo_data = pseudo_data | 0b0001

        # rotate the bits for other north orientations
        while north_position > 0:
            pseudo_data *= 2
            if pseudo_data > 15:
                pseudo_data -= 16
                pseudo_data +=1
            north_position -= 1
            
        return pseudo_data
        
    def _hash_blockarray(self):
        """Finds a hash of the block array"""
        if hasattr(self, "_digest"):
            return self._digest
        h = hashlib.md5()
        h.update(self.level['Blocks'])

        # If the render algorithm changes, change this line to re-generate all
        # the chunks automatically:
        h.update("1")

        digest = h.hexdigest()
        # 6 digits ought to be plenty
        self._digest = digest[:6]
        return self._digest

    def render_and_save(self, cave=False):
        """Render the chunk using chunk_render, and then save it to a file in
        the same directory as the source image. If the file already exists and
        is up to date, this method doesn't render anything.
        """
        blockid = self.blockid
        

        # Reasons for the code to get to this point:
        # 1) An old image doesn't exist
        # 2) An old image exists, but the chunk was more recently modified (the
        #    image was NOT checked if it was valid)
        # 3) An old image exists, the chunk was not modified more recently, but
        #    the image was invalid and deleted (sort of the same as (1))

        # What /should/ the image be named, go ahead and hash the block array
        try:
            dest_filename = "img.{0}.{1}.{2}.png".format(
                blockid,
                "cave" if cave else "nocave",
                self._hash_blockarray(),
                )
        except NoSuchChunk, e:
            return None

        
        dest_path = os.path.join(self.cachedir, dest_filename)
        #logging.debug("cache filename: %s", dest_path)

        if self.oldimg:
            if dest_filename == self.oldimg:
                # There is an existing file, the chunk has a newer mtime, but the
                # hashes match.
                # Before we return it, update its mtime so the next round
                # doesn't have to check the hash
                # TODO fix up hash checking
                os.utime(dest_path, None)
                logging.debug("Using cached image")
                return dest_path
            else:
                # Remove old image for this chunk. Anything already existing is
                # either corrupt or out of date
                os.unlink(self.oldimg_path)


        # Render the chunk
        img = self.chunk_render(cave=cave)
        # Save it
        try:
            img.save(dest_path)
        except:
            os.unlink(dest_path)
            raise
        # Return its location
        #raise Exception("early exit")
        return dest_path

    def calculate_darkness(self, skylight, blocklight):
        """Takes a raw blocklight and skylight, and returns a value
        between 0.0 (fully lit) and 1.0 (fully black) that can be used as
        an alpha value for a blend with a black source image. It mimics
        Minecraft lighting calculations."""
        if not self.world.night:
            # Daytime
            return 1.0 - pow(0.8, 15 - max(blocklight, skylight))
        else:
            # Nighttime
            return 1.0 - pow(0.8, 15 - max(blocklight, skylight - 11))
    
    def get_lighting_coefficient(self, x, y, z, norecurse=False):
        """Calculates the lighting coefficient for the given
        coordinate, using default lighting and peeking into
        neighboring chunks, if needed. A lighting coefficient of 1.0
        means fully black.
        
        Returns a tuple (coefficient, occluded), where occluded is
        True if the given coordinate is filled with a solid block, and
        therefore the returned coefficient is just the default."""
                
        # placeholders for later data arrays, coordinates
        blocks = None
        skylight = None
        blocklight = None
        local_x = x
        local_y = y
        local_z = z
        is_local_chunk = False
        
        # find out what chunk we're in, and translate accordingly
        if x >= 0 and y < 16:
            blocks = self.blocks
            skylight = self.skylight
            blocklight = self.blocklight
            is_local_chunk = True
        elif x < 0:
            local_x += 16
            blocks = self.left_blocks
            skylight = self.left_skylight
            blocklight = self.left_blocklight
        elif y >= 16:
            local_y -= 16
            blocks = self.right_blocks
            skylight = self.right_skylight
            blocklight = self.right_blocklight
        
        # make sure we have a correctly-ranged coordinates and enough
        # info about the chunk
        if not (blocks != None and skylight != None and blocklight != None and
                local_x >= 0 and local_x < 16 and local_y >= 0 and local_y < 16 and
                local_z >= 0 and local_z < 128):
            # we have no useful info, return default
            return (self.calculate_darkness(15, 0), False)
        
        blocktype = blocks[local_x, local_y, local_z]
        
        # special handling for half-blocks
        # (don't recurse more than once!)
        if blocktype == 44 and not norecurse:
            # average gathering variables
            averagegather = 0.0
            averagecount = 0
            
            # how bright we need before we consider a side "lit"
            threshold = self.calculate_darkness(0, 0)
            # iterate through all the sides of the block
            sides = [(x-1, y, z), (x+1, y, z), (x, y, z-1), (x, y, z+1), (x, y-1, z), (x, y+1, z)]
            
            for side in sides:
                val, occ = self.get_lighting_coefficient(*side, norecurse=True)
                if (not occ) and (val < threshold):
                    averagegather += val
                    averagecount += 1
            
            # if at least one side was lit, return the average
            if averagecount > 0:
                return (averagegather / averagecount, False)
        
        # calculate the return...
        occluded = not (blocktype in transparent_blocks)
        
        # only calculate the non-default coefficient if we're not occluded
        if (blocktype == 10) or (blocktype == 11):
            # lava blocks should always be lit!
            coefficient = 0.0
        elif occluded:
            coefficient = self.calculate_darkness(15, 0)
        else:
            coefficient = self.calculate_darkness(skylight[local_x, local_y, local_z], blocklight[local_x, local_y, local_z])
        
        # only say we're occluded if the point is in the CURRENT
        # chunk, so that we don't get obvious inter-chunk dependencies
        # (we want this here so we still have the default coefficient
        # for occluded blocks, even when we don't report them as
        # occluded)
        if not is_local_chunk:
            occluded = False
        
        return (coefficient, occluded)
        
    def chunk_render(self, img=None, xoff=0, yoff=0, cave=False):
        """Renders a chunk with the given parameters, and returns the image.
        If img is given, the chunk is rendered to that image object. Otherwise,
        a new one is created. xoff and yoff are offsets in the image.
        
        For cave mode, all blocks that have any direct sunlight are not
        rendered, and blocks are drawn with a color tint depending on their
        depth."""
        blocks = self.blocks
        pseudo_ancildata_blocks = set([85])
        
        left_blocks = self.left_blocks
        right_blocks = self.right_blocks
        
        if cave:
            # Cave mode. Actually go through and 0 out all blocks that are not in a
            # cave, so that it only renders caves.

            # Places where the skylight is not 0 (there's some amount of skylight
            # touching it) change it to something that won't get rendered, AND
            # won't get counted as "transparent".
            blocks = blocks.copy()
            blocks[self.skylight != 0] = 21

        blockData = get_blockdata_array(self.level)
        blockData_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        # Even elements get the lower 4 bits
        blockData_expanded[:,:,::2] = blockData & 0x0F
        # Odd elements get the upper 4 bits
        blockData_expanded[:,:,1::2] = blockData >> 4

        tileEntities = get_tileentity_data(self.level)

        if self.world.useBiomeData:
            biomeColorData = textures.getBiomeData(self.world.worlddir,
                self.chunkX, self.chunkY)
        # in the 8x8 block of biome data, what chunk is this?l
        startX = (self.chunkX - int(math.floor(self.chunkX/8)*8))
        startY = (self.chunkY - int(math.floor(self.chunkY/8)*8))

        # Each block is 24x24
        # The next block on the X axis adds 12px to x and subtracts 6px from y in the image
        # The next block on the Y axis adds 12px to x and adds 6px to y in the image
        # The next block up on the Z axis subtracts 12 from y axis in the image

        # Since there are 16x16x128 blocks in a chunk, the image will be 384x1728
        # (height is 128*12 high, plus the size of the horizontal plane: 16*12)
        if not img:
            img = Image.new("RGBA", (384, 1728), (38,92,255,0))

        for x,y,z,imgx,imgy in iterate_chunkblocks(xoff,yoff):
            blockid = blocks[x,y,z]

            # the following blocks don't have textures that can be pre-computed from the blockid
            # alone.  additional data is required.
            # TODO torches, redstone torches, crops, ladders, stairs, 
            #      levers, doors, buttons, and signs all need to be handled here (and in textures.py)

            ## minecart track, crops, ladder, doors, etc.
            if blockid in textures.special_blocks:
             # also handle furnaces here, since one side has a different texture than the other
                ancilData = blockData_expanded[x,y,z]
                try:
                    if blockid in pseudo_ancildata_blocks:
                        
                        pseudo_ancilData = self.generate_pseudo_ancildata(x,y,z,blockid)
                        ancilData = pseudo_ancilData
                    t = textures.specialblockmap[(blockid, ancilData)]
                except KeyError:
                    t = None

            else:
                t = textures.blockmap[blockid]

            if not t:
                continue
            
            if self.world.useBiomeData:
                if blockid == 2: #grass
                    index = biomeColorData[ ((startY*16)+y) * 128 + (startX*16) + x]
                    c = textures.grasscolor[index]

                    # only tint the top texture
                    t = textures.prepareGrassTexture(c)
                elif blockid == 18: # leaves
                    index = biomeColorData[ ((startY*16)+y) * 128 + (startX*16) + x]
                    c = textures.foliagecolor[index]
                    
                    t = textures.prepareLeafTexture(c)




            # Check if this block is occluded
            if cave and (
                    x == 0 and y != 15 and z != 127
            ):
                # If it's on the x face, only render if there's a
                # transparent block in the y+1 direction OR the z-1
                # direction
                if (
                    blocks[x,y+1,z] not in transparent_blocks and
                    blocks[x,y,z+1] not in transparent_blocks
                ):
                    continue
            elif cave and (
                    y == 15 and x != 0 and z != 127
            ):
                # If it's on the facing y face, only render if there's
                # a transparent block in the x-1 direction OR the z-1
                # direction
                if (
                    blocks[x-1,y,z] not in transparent_blocks and
                    blocks[x,y,z+1] not in transparent_blocks
                ):
                    continue
            elif cave and (
                    y == 15 and x == 0 and z != 127
            ):
                # If it's on the facing edge, only render if what's
                # above it is transparent
                if (
                    blocks[x,y,z+1] not in transparent_blocks
                ):
                    continue
            elif (left_blocks == None and right_blocks == None):
                    # Normal block or not cave mode, check sides for
                    # transparentcy or render if it's a border chunk.

                if (
                    x != 0 and y != 15 and z != 127 and
                    blocks[x-1,y,z] not in transparent_blocks and
                    blocks[x,y+1,z] not in transparent_blocks and
                    blocks[x,y,z+1] not in transparent_blocks
                    ):
                        continue

            elif (left_blocks != None and right_blocks == None):

                if (
                    # If it has the left face covered check for 
                    # transparent blocks in left face
                    y != 15 and z != 127 and
                    (left_blocks[15,y,z] if x == 0 else blocks[x - 1,y,z]) not in transparent_blocks and
                    blocks[x,y+1,z] not in transparent_blocks and
                    blocks[x,y,z+1] not in transparent_blocks
                    ):
                        continue

            elif (left_blocks == None and right_blocks != None):

                if (
                    # If it has the right face covered check for
                    # transparent blocks in right face
                    x != 0 and z != 127 and
                    blocks[x-1,y,z] not in transparent_blocks and
                    (right_blocks[x,0,z] if y == 15 else blocks[x,y + 1,z]) not in transparent_blocks and
                    blocks[x,y,z+1] not in transparent_blocks
                    ):
                        continue
                
            elif (
                # If it's a interior chunk check for transparent blocks
                # in the adjacent chunks.
                z != 127 and 
                (left_blocks[15,y,z] if x == 0 else blocks[x - 1,y,z]) not in transparent_blocks and
                (right_blocks[x,0,z] if y == 15 else blocks[x,y + 1,z]) not in transparent_blocks and
                blocks[x,y,z+1] not in transparent_blocks
                # Don't render if all sides aren't transparent
                ):
                    continue

            # Draw the actual block on the image. For cave images,
            # tint the block with a color proportional to its depth
            if cave:
                # no lighting for cave -- depth is probably more useful
                composite.alpha_over(img, Image.blend(t[0],depth_colors[z],0.3), (imgx, imgy), t[1])
            else:
                if not self.world.lighting:
                    # no lighting at all
                    composite.alpha_over(img, t[0], (imgx, imgy), t[1])
                elif blockid in transparent_blocks:
                    # transparent means draw the whole
                    # block shaded with the current
                    # block's light
                    black_coeff, _ = self.get_lighting_coefficient(x, y, z)
                    if self.world.spawn and black_coeff > 0.8 and blockid in solid_blocks and not (
                        blockid in nospawn_blocks or (
                            z != 127 and (blocks[x,y,z+1] in solid_blocks or blocks[x,y,z+1] in fluid_blocks)
                        )
                    ):
                        composite.alpha_over(img, Image.blend(t[0], red_color, black_coeff), (imgx, imgy), t[1])
                    else:
                        composite.alpha_over(img, Image.blend(t[0], black_color, black_coeff), (imgx, imgy), t[1])
                else:
                    # draw each face lit appropriately,
                    # but first just draw the block
                    composite.alpha_over(img, t[0], (imgx, imgy), t[1])
                    
                    # top face
                    black_coeff, face_occlude = self.get_lighting_coefficient(x, y, z + 1)
                    # Use red instead of black for spawnable blocks
                    if self.world.spawn and black_coeff > 0.8 and blockid in solid_blocks and not (
                        blockid in nospawn_blocks or (
                            z != 127 and (blocks[x,y,z+1] in solid_blocks or blocks[x,y,z+1] in fluid_blocks)
                        )
                    ):
                        over_color = red_color
                    else:
                        over_color = black_color

                    if not face_occlude:
                        composite.alpha_over(img, over_color, (imgx, imgy), ImageEnhance.Brightness(facemasks[0]).enhance(black_coeff))
                    
                    # left face
                    black_coeff, face_occlude = self.get_lighting_coefficient(x - 1, y, z)
                    if not face_occlude:
                        composite.alpha_over(img, over_color, (imgx, imgy), ImageEnhance.Brightness(facemasks[1]).enhance(black_coeff))

                    # right face
                    black_coeff, face_occlude = self.get_lighting_coefficient(x, y + 1, z)
                    if not face_occlude:
                        composite.alpha_over(img, over_color, (imgx, imgy), ImageEnhance.Brightness(facemasks[2]).enhance(black_coeff))

            # Draw edge lines
            if blockid in (44,): # step block
               increment = 6
            elif blockid in (78,): # snow
               increment = 9
            else:
               increment = 0

            if blockid not in transparent_blocks or blockid in (78,): #special case snow so the outline is still drawn
                draw = ImageDraw.Draw(img)
                if x != 15 and blocks[x+1,y,z] == 0:
                    draw.line(((imgx+12,imgy+increment), (imgx+22,imgy+5+increment)), fill=(0,0,0), width=1)
                if y != 0 and blocks[x,y-1,z] == 0:
                    draw.line(((imgx,imgy+6+increment), (imgx+12,imgy+increment)), fill=(0,0,0), width=1)


        for entity in tileEntities:
            if entity['id'] == 'Sign':
                msg=' \n'.join([entity['Text1'], entity['Text2'], entity['Text3'], entity['Text4']])
                if msg.strip():
                    # convert the blockID coordinates from local chunk
                    # coordinates to global world coordinates
                    newPOI = dict(type="sign",
                                    x= entity['x'],
                                    y= entity['y'],
                                    z= entity['z'],
                                    msg=msg,
                                    chunk= (self.chunkX, self.chunkY),
                                  )
                    self.queue.put(["newpoi", newPOI])


        # check to see if there are any signs in the persistentData list that are from this chunk.
        # if so, remove them from the persistentData list (since they're have been added to the world.POI
        # list above.
        self.queue.put(['removePOI', (self.chunkX, self.chunkY)])

        return img

# Render 3 blending masks for lighting
# first is top (+Z), second is left (-X), third is right (+Y)
def generate_facemasks():
    white = Image.new("L", (24,24), 255)
    
    top = Image.new("L", (24,24), 0)
    left = Image.new("L", (24,24), 0)
    whole = Image.new("L", (24,24), 0)
    
    toppart = textures.transform_image(white)
    leftpart = textures.transform_image_side(white)
    
    # using the real PIL paste here (not alpha_over) because there is
    # no alpha channel (and it's mode "L")
    top.paste(toppart, (0,0))
    left.paste(leftpart, (0,6))
    right = left.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Manually touch up 6 pixels that leave a gap, like in
    # textures._build_block()
    for x,y in [(13,23), (17,21), (21,19)]:
        right.putpixel((x,y), 255)
    for x,y in [(3,4), (7,2), (11,0)]:
        top.putpixel((x,y), 255)
    
    return (top, left, right)
facemasks = generate_facemasks()
black_color = Image.new("RGB", (24,24), (0,0,0))
red_color = Image.new("RGB", (24,24), (229,36,38))

# Render 128 different color images for color coded depth blending in cave mode
def generate_depthcolors():
    depth_colors = []
    r = 255
    g = 0
    b = 0
    for z in range(128):
        img = Image.new("RGB", (24,24), (r,g,b))
        depth_colors.append(img)
        if z < 32:
            g += 7
        elif z < 64:
            r -= 7
        elif z < 96:
            b += 7
        else:
            g -= 7

    return depth_colors
depth_colors = generate_depthcolors()
