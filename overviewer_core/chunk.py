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
import logging
import time
import math
import sys

import nbt
import textures
import world
import composite
import c_overviewer

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

def get_lvldata(world, filename, x, y, retries=2):
    """Takes a filename and chunkcoords and returns the Level struct, which contains all the
    level info"""
    
    # non existent region file doesn't mean corrupt chunk.
    if filename == None:
        raise NoSuchChunk
    
    try:
        d =  world.load_from_region(filename, x, y)
    except Exception, e:
        if retries > 0:
            # wait a little bit, and try again (up to `retries` times)
            time.sleep(1)
            #make sure we reload region info
            world.reload_region(filename)
            return get_lvldata(world, filename, x, y, retries=retries-1)
        else:
            logging.warning("Error opening chunk (%i, %i) in %s. It may be corrupt. %s", x, y, filename, e)
            raise ChunkCorrupt(str(e))
    
    if not d: raise NoSuchChunk(x,y)
    return d

def get_blockarray(level):
    """Takes the level struct as returned from get_lvldata, and returns the
    Block array, which just contains all the block ids"""
    return level['Blocks']

def get_blockarray_fromfile(filename, north_direction='lower-left'):
    """Same as get_blockarray except takes a filename. This is a shortcut"""
    d = nbt.load_from_region(filename, x, y, north_direction)
    level = d[1]['Level']
    chunk_data = level
    rots = 0
    if self.north_direction == 'upper-left':
        rots = 1
    elif self.north_direction == 'upper-right':
        rots = 2
    elif self.north_direction == 'lower-right':
        rots = 3

    chunk_data['Blocks'] = numpy.rot90(numpy.frombuffer(
            level['Blocks'], dtype=numpy.uint8).reshape((16,16,128)),
            rots)
    return get_blockarray(chunk_data)

def get_skylight_array(level):
    """Returns the skylight array. This is 4 bits per block, but it is
    expanded for you so you may index it normally."""
    skylight = level['SkyLight']
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
    blocklight = level['BlockLight']
    blocklight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
    blocklight_expanded[:,:,::2] = blocklight & 0x0F
    blocklight_expanded[:,:,1::2] = (blocklight & 0xF0) >> 4
    return blocklight_expanded

def get_blockdata_array(level):
    """Returns the ancillary data from the 'Data' byte array.  Data is packed
    in a similar manner to skylight data"""
    return level['Data']

def get_tileentity_data(level):
    """Returns the TileEntities TAG_List from chunk dat file"""
    data = level['TileEntities']
    return data

def get_entity_data(level):
    """Returns the Entities TAG_List from chunk dat file"""
    data = level['Entities']
    return data

# This set holds blocks ids that can be seen through, for occlusion calculations
transparent_blocks = set([ 0,  6,  8,  9, 18, 20, 26, 27, 28, 29, 30, 31, 32, 33,
                          34, 37, 38, 39, 40, 44, 50, 51, 52, 53, 55, 59, 63, 64,
                          65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79,
                          81, 83, 85, 90, 92, 93, 94, 96, 101, 102, 104, 105,
                          106, 107, 108, 109])

# This set holds block ids that are solid blocks
solid_blocks = set([1, 2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 35, 41, 42, 43, 44, 45, 46, 47, 48, 49, 53, 54, 56, 57, 58, 60, 
    61, 62, 67, 73, 74, 78, 79, 80, 81, 82, 84, 86, 87, 88, 89, 91])

# This set holds block ids that are fluid blocks
fluid_blocks = set([8,9,10,11])

# This set holds block ids that are not candidates for spawning mobs on
# (glass, slabs, stairs, fluids, ice, pistons, webs,TNT, wheat, cactus, iron bars, glass planes, fences, fence gate, cake, bed, repeaters, trapdoor)
nospawn_blocks = set([20,26, 29, 30, 33, 34, 44, 46, 53, 59, 67, 79, 81, 85, 92, 93, 94, 96, 107, 109, 101, 102]).union(fluid_blocks)

class ChunkCorrupt(Exception):
    pass

class NoSuchChunk(Exception):
    pass

class ChunkRenderer(object):
    def __init__(self, chunkcoords, worldobj, rendermode, queue):
        """Make a new chunk renderer for the given chunk coordinates.
        chunkcoors should be a tuple: (chunkX, chunkY)
        
        cachedir is a directory to save the resulting chunk images to
        """
        self.queue = queue
        
        self.regionfile = worldobj.get_region_path(*chunkcoords)    
        #if not os.path.exists(self.regionfile):
        #    raise ValueError("Could not find regionfile: %s" % self.regionfile)

        ## TODO TODO all of this class

        #destdir, filename = os.path.split(self.chunkfile)
        #filename_split = filename.split(".")
        #chunkcoords = filename_split[1:3]
        
        #self.coords = map(world.base36decode, chunkcoords)
        #self.blockid = "%d.%d" % chunkcoords

        # chunk coordinates (useful to converting local block coords to 
        # global block coords)
        self.chunkX = chunkcoords[0]
        self.chunkY = chunkcoords[1]

        self.world = worldobj
        self.rendermode = rendermode

    def _load_level(self):
        """Loads and returns the level structure"""
        if not hasattr(self, "_level"):
            try:
                self._level = get_lvldata(self.world,self.regionfile, self.chunkX, self.chunkY)
            except NoSuchChunk, e:
                logging.debug("Skipping non-existant chunk")
                raise
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
            chunk_data = get_lvldata(self.world,chunk_path, self.chunkX - 1, self.chunkY)
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
            chunk_data = get_lvldata(self.world,chunk_path, self.chunkX, self.chunkY + 1)
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
            chunk_data = get_lvldata(self.world,chunk_path, self.chunkX + 1, self.chunkY)
            self._up_right_skylight = get_skylight_array(chunk_data)
            self._up_right_blocklight = get_blocklight_array(chunk_data)
            self._up_right_blocks = get_blockarray(chunk_data)
        except NoSuchChunk:
            self._up_right_skylight = None
            self._up_right_blocklight = None
            self._up_right_blocks = None
    
    def _load_up_right_blocks(self):
        """Loads and returns upper-right block array"""
        if not hasattr(self, "_up_right_blocks"):
            self._load_up_right()
        return self._up_right_blocks
    up_right_blocks = property(_load_up_right_blocks)

    def _load_up_right_skylight(self):
        """Loads and returns lower-right skylight array"""
        if not hasattr(self, "_up_right_skylight"):
            self._load_up_right()
        return self._up_right_skylight
    up_right_skylight = property(_load_up_right_skylight)

    def _load_up_right_blocklight(self):
        """Loads and returns lower-right blocklight array"""
        if not hasattr(self, "_up_right_blocklight"):
            self._load_up_right()
        return self._up_right_blocklight
    up_right_blocklight = property(_load_up_right_blocklight)

    def _load_up_left(self):
        """Loads and sets data from upper-left chunk"""
        chunk_path = self.world.get_region_path(self.chunkX, self.chunkY - 1)
        try:
            chunk_data = get_lvldata(self.world,chunk_path, self.chunkX, self.chunkY - 1)
            self._up_left_skylight = get_skylight_array(chunk_data)
            self._up_left_blocklight = get_blocklight_array(chunk_data)
            self._up_left_blocks = get_blockarray(chunk_data)
        except NoSuchChunk:
            self._up_left_skylight = None
            self._up_left_blocklight = None
            self._up_left_blocks = None
    
    def _load_up_left_blocks(self):
        """Loads and returns lower-left block array"""
        if not hasattr(self, "_up_left_blocks"):
            self._load_up_left()
        return self._up_left_blocks
    up_left_blocks = property(_load_up_left_blocks)

    def _load_up_left_skylight(self):
        """Loads and returns lower-right skylight array"""
        if not hasattr(self, "_up_left_skylight"):
            self._load_up_left()
        return self._up_left_skylight
    up_left_skylight = property(_load_up_left_skylight)

    def _load_up_left_blocklight(self):
        """Loads and returns lower-left blocklight array"""
        if not hasattr(self, "_up_left_blocklight"):
            self._load_up_left()
        return self._up_left_blocklight
    up_left_blocklight = property(_load_up_left_blocklight)

    def chunk_render(self, img=None, xoff=0, yoff=0, cave=False):
        """Renders a chunk with the given parameters, and returns the image.
        If img is given, the chunk is rendered to that image object. Otherwise,
        a new one is created. xoff and yoff are offsets in the image.
        
        For cave mode, all blocks that have any direct sunlight are not
        rendered, and blocks are drawn with a color tint depending on their
        depth."""
        
        blockData = get_blockdata_array(self.level)
        blockData_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        # Even elements get the lower 4 bits
        blockData_expanded[:,:,::2] = blockData & 0x0F
        # Odd elements get the upper 4 bits
        blockData_expanded[:,:,1::2] = blockData >> 4


        # Each block is 24x24
        # The next block on the X axis adds 12px to x and subtracts 6px from y in the image
        # The next block on the Y axis adds 12px to x and adds 6px to y in the image
        # The next block up on the Z axis subtracts 12 from y axis in the image

        # Since there are 16x16x128 blocks in a chunk, the image will be 384x1728
        # (height is 128*12 high, plus the size of the horizontal plane: 16*12)
        if not img:
            img = Image.new("RGBA", (384, 1728), (38,92,255,0))

        c_overviewer.render_loop(self, img, xoff, yoff, blockData_expanded)

        tileEntities = get_tileentity_data(self.level)
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
                    if self.queue:
                        self.queue.put(["newpoi", newPOI])

		# we're going to look for animals here
#        entities = get_entity_data(self.level)
#        for entity in entities:
#			if entity['id'] == 'Cow':
#				newPOI = dict(type="cow",
#								x= entity['Pos'][0], y= entity['Pos'][1], z= entity['Pos'][2], msg='moo', chunk=(self.chunkX, self.chunkY),
#							   )
#			    if self.queue:
#					self.queue.put(["cow", newPOI])

        # check to see if there are any signs in the persistentData list that are from this chunk.
        # if so, remove them from the persistentData list (since they're have been added to the world.POI
        # list above.
        if self.queue:
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
    
    # special fix for chunk boundary stipple
    for x,y in [(13,11), (17,9), (21,7)]:
        right.putpixel((x,y), 0)
    
    return (top, left, right)
facemasks = generate_facemasks()
black_color = Image.new("RGB", (24,24), (0,0,0))
white_color = Image.new("RGB", (24,24), (255,255,255))

# Render 128 different color images for color coded depth blending in cave mode
def generate_depthcolors():
    depth_colors = []
    r = 255
    g = 0
    b = 0
    for z in range(128):
        depth_colors.append(r)
        depth_colors.append(g)
        depth_colors.append(b)
        
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
