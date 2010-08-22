import numpy
from PIL import Image
from itertools import izip, count

import nbt
import textures
from textures import texturemap as txtarray

# General note about pasting transparent image objects onto an image with an
# alpha channel:
# If you use the image as its own mask, it will work fine only if the alpha
# channel is binary. If there's any translucent parts, then the alpha channel
# of the dest image will have its alpha channel modified. To prevent this:
# first use im.split() and take the third item which is the alpha channel and
# use that as the mask. Then take the image and use im.convert("RGB") to strip
# the image from its alpha channel, and use that as the source to paste()

def get_lvldata(filename):
    """Takes a filename and returns the Level struct, which contains all the
    level info"""
    return nbt.load(filename)[1]['Level']

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
    """Returns the skylight array. Remember this is 4 bits per block, so divide
    the z component by 2 when accessing the array. and mask off the top or
    bottom 4 bits if it's odd or even respectively
    """
    return numpy.frombuffer(level['SkyLight'], dtype=numpy.uint8).reshape((16,16,64))

# This set holds blocks ids that can be seen through
transparent_blocks = set([0, 8, 9, 18, 20, 37, 38, 39, 40, 50, 51, 52, 53, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 79, 83, 85])

def chunk_render(chunkfile, img=None, xoff=0, yoff=0, cave=False):
    level = get_lvldata(chunkfile)
    blocks = get_blockarray(level)
    if cave:
        skylight = get_skylight_array(level)
        # Cave mode. Actually go through and 0 out all blocks that are not in a
        # cave, so that it only renders caves.

        # 1st task: this array is 2 blocks per byte, expand it so we can just
        # do a bitwise and on the arrays
        skylight_expanded = numpy.empty((16,16,128), dtype=numpy.uint8)
        # Even elements get the lower 4 bits
        skylight_expanded[:,:,::2] = skylight & 0x0F
        # Odd elements get the upper 4 bits
        skylight_expanded[:,:,1::2] = skylight >> 4

        # Places where the skylight is not 0 (there's some amount of skylight
        # touching it) change it to something that won't get rendered, AND
        # won't get counted as "transparent".
        blocks = blocks.copy()
        blocks[skylight_expanded != 0] = 21

        # Don't render


    # Each block is 24x24
    # The next block on the X axis adds 12px to x and subtracts 6px from y in the image
    # The next block on the Y axis adds 12px to x and adds 6px to y in the image
    # The next block up on the Z axis subtracts 12 from y axis in the image

    # Since there are 16x16x128 blocks in a chunk, the image will be 384x1728
    # (height is 128*24 high, plus the size of the horizontal plane: 16*12)
    if not img:
        img = Image.new("RGBA", (384, 1728))

    for x in xrange(15,-1,-1):
        for y in xrange(16):
            imgx = xoff + x*12 + y*12
            imgy = yoff - x*6 + y*6 + 128*12 + 16*12//2
            for z in xrange(128):
                try:

                    blockid = blocks[x,y,z]
                    t = textures.blockmap[blockid]
                    if not t:
                        continue

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
                            y == 15 and x == 0
                    ):
                        # If it's on the facing edge, only render if what's
                        # above it is transparent
                        if (
                            blocks[x,y,z+1] not in transparent_blocks
                        ):
                            continue
                    elif (
                            # Normal block or not cave mode, check sides for
                            # transparentcy or render unconditionally if it's
                            # on a shown face
                            x != 0 and y != 15 and z != 127 and
                            blocks[x-1,y,z] not in transparent_blocks and
                            blocks[x,y+1,z] not in transparent_blocks and
                            blocks[x,y,z+1] not in transparent_blocks
                    ):
                        # Don't render if all sides aren't transparent and
                        # we're not on the edge
                        continue

                    img.paste(t[0], (imgx, imgy), t[1])

                finally:
                    # Do this no mater how the above block exits
                    imgy -= 12

    return img
