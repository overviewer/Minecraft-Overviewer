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

import sys
import os
import os.path
import zipfile
from cStringIO import StringIO
import math

import numpy
from PIL import Image, ImageEnhance, ImageOps, ImageDraw

import util
import composite

def _find_file(filename, mode="rb"):
    """Searches for the given file and returns an open handle to it.
    This searches the following locations in this order:
    
    * The program dir (same dir as this file)
    * On Darwin, in /Applications/Minecraft
    * Inside minecraft.jar, which is looked for at these locations

      * On Windows, at %APPDATA%/.minecraft/bin/minecraft.jar
      * On Darwin, at $HOME/Library/Application Support/minecraft/bin/minecraft.jar
      * at $HOME/.minecraft/bin/minecraft.jar

    * The current working directory
    * The program dir / textures

    """
    programdir = util.get_program_path()
    path = os.path.join(programdir, filename)
    if os.path.exists(path):
        return open(path, mode)

    if sys.platform == "darwin":
        path = os.path.join("/Applications/Minecraft", filename)
        if os.path.exists(path):
            return open(path, mode)

    # Find minecraft.jar.
    jarpaths = []
    if "APPDATA" in os.environ:
        jarpaths.append( os.path.join(os.environ['APPDATA'], ".minecraft",
            "bin", "minecraft.jar"))
    if "HOME" in os.environ:
        jarpaths.append(os.path.join(os.environ['HOME'], "Library",
                "Application Support", "minecraft","bin","minecraft.jar"))
        jarpaths.append(os.path.join(os.environ['HOME'], ".minecraft", "bin",
                "minecraft.jar"))
    jarpaths.append(programdir)
    jarpaths.append(os.getcwd())

    for jarpath in jarpaths:
        if os.path.exists(jarpath):
            try:
                jar = zipfile.ZipFile(jarpath)
                return jar.open(filename)
            except (KeyError, IOError):
                pass

    path = filename
    if os.path.exists(path):
        return open(path, mode)

    path = os.path.join(programdir, "textures", filename)
    if os.path.exists(path):
        return open(path, mode)

    raise IOError("Could not find the file {0}. Is Minecraft installed? If so, I couldn't find the minecraft.jar file.".format(filename))

def _load_image(filename):
    """Returns an image object"""
    fileobj = _find_file(filename)
    buffer = StringIO(fileobj.read())
    return Image.open(buffer)

def _get_terrain_image():
    return _load_image("terrain.png")

def _split_terrain(terrain):
    """Builds and returns a length 256 array of each 16x16 chunk of texture"""
    textures = []
    (terrain_width, terrain_height) = terrain.size
    texture_resolution = terrain_width / 16
    for y in xrange(16):
        for x in xrange(16):
            left = x*texture_resolution
            upper = y*texture_resolution
            right = left+texture_resolution
            lower = upper+texture_resolution
            region = terrain.transform(
                      (16, 16),
                      Image.EXTENT,
                      (left,upper,right,lower),
                      Image.BICUBIC)
            textures.append(region)

    return textures

# This maps terainids to 16x16 images
terrain_images = _split_terrain(_get_terrain_image())

def transform_image(img, blockID=None):
    """Takes a PIL image and rotates it left 45 degrees and shrinks the y axis
    by a factor of 2. Returns the resulting image, which will be 24x12 pixels

    """

    # Resize to 17x17, since the diagonal is approximately 24 pixels, a nice
    # even number that can be split in half twice
    img = img.resize((17, 17), Image.ANTIALIAS)

    # Build the Affine transformation matrix for this perspective
    transform = numpy.matrix(numpy.identity(3))
    # Translate up and left, since rotations are about the origin
    transform *= numpy.matrix([[1,0,8.5],[0,1,8.5],[0,0,1]])
    # Rotate 45 degrees
    ratio = math.cos(math.pi/4)
    #transform *= numpy.matrix("[0.707,-0.707,0;0.707,0.707,0;0,0,1]")
    transform *= numpy.matrix([[ratio,-ratio,0],[ratio,ratio,0],[0,0,1]])
    # Translate back down and right
    transform *= numpy.matrix([[1,0,-12],[0,1,-12],[0,0,1]])
    # scale the image down by a factor of 2
    transform *= numpy.matrix("[1,0,0;0,2,0;0,0,1]")

    transform = numpy.array(transform)[:2,:].ravel().tolist()

    newimg = img.transform((24,12), Image.AFFINE, transform)
    return newimg

def transform_image_side(img, blockID=None):
    """Takes an image and shears it for the left side of the cube (reflect for
    the right side)"""

    if blockID in (44,): # step block
        # make the top half transparent
        # (don't just crop img, since we want the size of
        # img to be unchanged
        mask = img.crop((0,8,16,16))
        n = Image.new(img.mode, img.size, (38,92,255,0))
        composite.alpha_over(n, mask,(0,0,16,8), mask)
        img = n
    if blockID in (78,): # snow
        # make the top three quarters transparent
        mask = img.crop((0,12,16,16))
        n = Image.new(img.mode, img.size, (38,92,255,0))
        composite.alpha_over(n, mask,(0,12,16,16), mask)
        img = n

    # Size of the cube side before shear
    img = img.resize((12,12), Image.ANTIALIAS)

    # Apply shear
    transform = numpy.matrix(numpy.identity(3))
    transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

    transform = numpy.array(transform)[:2,:].ravel().tolist()

    newimg = img.transform((12,18), Image.AFFINE, transform)
    return newimg

def transform_image_slope(img, blockID=None):
    """Takes an image and shears it in the shape of a slope going up
    in the -y direction (reflect for +x direction). Used for minetracks"""

    # Take the same size as trasform_image_side
    img = img.resize((12,12), Image.ANTIALIAS)

    # Apply shear
    transform = numpy.matrix(numpy.identity(3))
    transform *= numpy.matrix("[0.75,-0.5,3;0.25,0.5,-3;0,0,1]")
    transform = numpy.array(transform)[:2,:].ravel().tolist()
    
    newimg = img.transform((24,24), Image.AFFINE, transform)
    
    return newimg
    

def _build_block(top, side, blockID=None):
    """From a top texture and a side texture, build a block image.
    top and side should be 16x16 image objects. Returns a 24x24 image

    """
    img = Image.new("RGBA", (24,24), (38,92,255,0))

    top = transform_image(top, blockID)

    if not side:
        composite.alpha_over(img, top, (0,0), top)
        return img

    side = transform_image_side(side, blockID)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)

    ## special case for non-block things
    if blockID in (37,38,6,39,40,83,30): ## flowers, sapling, mushrooms, reeds, web
        #
        # instead of pasting these blocks at the cube edges, place them in the middle:
        # and omit the top
        composite.alpha_over(img, side, (6,3), side)
        composite.alpha_over(img, otherside, (6,3), otherside)
        return img


    if blockID in (81,): # cacti!
        composite.alpha_over(img, side, (1,6), side)
        composite.alpha_over(img, otherside, (11,6), otherside)
        composite.alpha_over(img, top, (0,0), top)
    elif blockID in (44,): # half step
        # shift each texture down 6 pixels
        composite.alpha_over(img, side, (0,12), side)
        composite.alpha_over(img, otherside, (12,12), otherside)
        composite.alpha_over(img, top, (0,6), top)
    elif blockID in (78,): # snow
        # shift each texture down 9 pixels
        composite.alpha_over(img, side, (0,6), side)
        composite.alpha_over(img, otherside, (12,6), otherside)
        composite.alpha_over(img, top, (0,9), top)
    else:
        composite.alpha_over(img, side, (0,6), side)
        composite.alpha_over(img, otherside, (12,6), otherside)
        composite.alpha_over(img, top, (0,0), top)

    # Manually touch up 6 pixels that leave a gap because of how the
    # shearing works out. This makes the blocks perfectly tessellate-able
    for x,y in [(13,23), (17,21), (21,19)]:
        # Copy a pixel to x,y from x-1,y
        img.putpixel((x,y), img.getpixel((x-1,y)))
    for x,y in [(3,4), (7,2), (11,0)]:
        # Copy a pixel to x,y from x+1,y
        img.putpixel((x,y), img.getpixel((x+1,y)))

    return img


def _build_full_block(top, side1, side2, side3, side4, bottom=None, blockID=None):
    """From a top texture, a bottom texture and 4 different side textures,
    build a full block with four differnts faces. All images should be 16x16 
    image objects. Returns a 24x24 image. Can be used to render any block.
    
    side1 is in the -y face of the cube     (top left, east)
    side2 is in the +x                      (top right, south)
    side3 is in the -x                      (bottom left, north)
    side4 is in the +y                      (bottom right, west)
    
    A non transparent block uses top, side 3 and side 4.
    
    If top is a tuple then first member is the top image and the second
    member is an increment (integer) from 0 to 12. This increment will
    used to crop the side images to look like a block and to paste all
    the images increment pixels lower. Using increment = 6 will create
    a half-block.
    
    NOTE: this method uses the top of the texture image (as done in 
    minecraft with beds)
    
    """
    
    increment = 0
    if isinstance(top, tuple):
        increment = top[1]
        crop_height = int(increment * 16./12.)
        top = top[0]
        if side1 != None:
            side1 = side1.copy()
            ImageDraw.Draw(side1).rectangle((0, 16 - crop_height,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
        if side2 != None:
            side2 = side2.copy()
            ImageDraw.Draw(side2).rectangle((0, 16 - crop_height,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
        if side3 != None:
            side3 = side3.copy()
            ImageDraw.Draw(side3).rectangle((0, 16 - crop_height,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
        if side4 != None:
            side4 = side4.copy()
            ImageDraw.Draw(side4).rectangle((0, 16 - crop_height,16,16),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), (38,92,255,0))
    
    # first back sides
    if side1 != None :
        side1 = transform_image_side(side1, blockID)
        side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Darken this side.
        sidealpha = side1.split()[3]
        side1 = ImageEnhance.Brightness(side1).enhance(0.9)
        side1.putalpha(sidealpha)        
        
        composite.alpha_over(img, side1, (0,0 + increment), side1)

        
    if side2 != None :
        side2 = transform_image_side(side2, blockID)

        # Darken this side.
        sidealpha2 = side2.split()[3]
        side2 = ImageEnhance.Brightness(side2).enhance(0.8)
        side2.putalpha(sidealpha2)

        composite.alpha_over(img, side2, (12,0 + increment), side2)

    if bottom != None :
        bottom = transform_image(bottom, blockID)
        composite.alpha_over(img, bottom, (0,12), top)
        
    # front sides
    if side3 != None :
        side3 = transform_image_side(side3, blockID)
        
        # Darken this side
        sidealpha = side3.split()[3]
        side3 = ImageEnhance.Brightness(side3).enhance(0.9)
        side3.putalpha(sidealpha)
        
        composite.alpha_over(img, side3, (0,6 + increment), side3)
        
    if side4 != None :
        side4 = transform_image_side(side4, blockID)
        side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken this side
        sidealpha = side4.split()[3]
        side4 = ImageEnhance.Brightness(side4).enhance(0.8)
        side4.putalpha(sidealpha)
        
        composite.alpha_over(img, side4, (12,6 + increment), side4)

    if top != None :
        top = transform_image(top, blockID)
        composite.alpha_over(img, top, (0, increment), top)

    return img


def _build_blockimages():
    """Returns a mapping from blockid to an image of that block in perspective
    The values of the mapping are actually (image in RGB mode, alpha channel).
    This is not appropriate for all block types, only block types that are
    proper cubes"""

    # Top textures of all block types. The number here is the index in the
    # texture array (terrain_images), which comes from terrain.png's cells, left to right top to
    # bottom.
       #        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    topids = [ -1,  1,  0,  2, 16,  4, -1, 17,205,205,237,237, 18, 19, 32, 33,
       #       16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
               34, -1, 52, 48, 49,160,144, -1,176, 74, -1, -1, -1, -1, 11, -1,
       #       32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
               -1, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  9,  4, 
       #       48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
               36, 37, -1, -1, 65, -1, -1, -1, 50, 24, -1, -1, 86, -1, -1, -1,
       #       64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
               -1, -1, -1, -1, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, 67,
       #       80  81  82  83  84  85  86  87  88  89  90  91
               66, 69, 72, 73, 75, -1,102,103,104,105,-1, 102 # clay?
        ]

    # NOTE: For non-block textures, the sideid is ignored, but can't be -1

    # And side textures of all block types
       #         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    sideids = [ -1,  1,  3,  2, 16,  4, -1, 17,205,205,237,237, 18, 19, 32, 33,
       #        16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
                34, -1, 52, 48, 49,160,144, -1,192, 74, -1, -1,- 1, -1, 11, -1,
       #        32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
                -1, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  8, 35,
       #        48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
                36, 37, -1, -1, 65, -1, -1,101, 50, 24, -1, -1, 86, -1, -1, -1,
       #        64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
                -1, -1, -1, -1, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, 67,
       #        80  81  82  83  84  85  86  87  88  89  90  91
                66, 70, 72, 73, 74,-1 ,118,103,104,105, -1, 118
        ]

    # This maps block id to the texture that goes on the side of the block
    if len(topids) != len(sideids):
        raise Exception("mismatched lengths")

    allimages = []
    for toptextureid, sidetextureid,blockID in zip(topids, sideids,range(len(topids))):
        if toptextureid == -1 or sidetextureid == -1:
            allimages.append(None)
            continue

        toptexture = terrain_images[toptextureid]
        sidetexture = terrain_images[sidetextureid]

        ## _build_block needs to know about the block ID, not just the textures
        ## of the block or the texture ID
        img = _build_block(toptexture, sidetexture, blockID)

        allimages.append((img.convert("RGB"), img.split()[3]))

    # Future block types:
    while len(allimages) < 256:
        allimages.append(None)
    return allimages
blockmap = _build_blockimages()

def load_water():
    """Evidentially, the water and lava textures are not loaded from any files
    in the jar (that I can tell). They must be generated on the fly. While
    terrain.png does have some water and lava cells, not all texture packs
    include them. So I load them here from a couple pngs included.

    This mutates the blockmap global list with the new water and lava blocks.
    Block 9, standing water, is given a block with only the top face showing.
    Block 8, flowing water, is given a full 3 sided cube."""

    watertexture = _load_image("water.png")
    w1 = _build_block(watertexture, None)
    blockmap[9] = w1.convert("RGB"), w1
    w2 = _build_block(watertexture, watertexture)
    blockmap[8] = w2.convert("RGB"), w2

    lavatexture = _load_image("lava.png")
    lavablock = _build_block(lavatexture, lavatexture)
    blockmap[10] = lavablock.convert("RGB"), lavablock
    blockmap[11] = blockmap[10]
load_water()


def generate_special_texture(blockID, data):
    """Generates a special texture, such as a correctly facing minecraft track"""
    #print "%s has ancillary data: %X" %(blockID, data)
    # TODO ladders, stairs, levers, buttons, and signs
    # all need to behandled here (and in chunkpy)
    
    if blockID == 2: # grass
        # data & 0x10 means SNOW sides
        side_img = terrain_images[3]
        if data & 0x10:
            side_img = terrain_images[68]
        img = _build_block(terrain_images[0], side_img, 2)
        if not data & 0x10:
            colored = tintTexture(biome_grass_texture, (115, 175, 71))
            composite.alpha_over(img, colored, (0, 0), colored)
        return (img.convert("RGB"), img.split()[3])


    if blockID == 6: # saplings
        # The bottom two bits are used fo the sapling type, the top two
        # bits are used as a grow-counter for the tree.

        if data & 0x3 == 0: # usual saplings
            toptexture = terrain_images[15]
            sidetexture = terrain_images[15]

        if data & 0x3 == 1: # spruce sapling
            toptexture = terrain_images[63]
            sidetexture = terrain_images[63]
        
        if data & 0x3 == 2: # birch sapling
            toptexture = terrain_images[79]
            sidetexture = terrain_images[79]
        
        if data & 0x3 == 3: # unused usual sapling
            toptexture = terrain_images[15]
            sidetexture = terrain_images[15]

        img = _build_block(toptexture, sidetexture, blockID)
        return (img.convert("RGB"),img.split()[3])


    if blockID == 9: # spring water, flowing water and waterfall water

        watertexture = _load_image("water.png")
        
        if (data & 0b10000) == 16:
            top = watertexture
            
        else: top = None

        if (data & 0b0001) == 1:
            side1 = watertexture    # top left
        else: side1 = None
        
        if (data & 0b1000) == 8:
            side2 = watertexture    # top right           
        else: side2 = None
        
        if (data & 0b0010) == 2:
            side3 = watertexture    # bottom left    
        else: side3 = None
        
        if (data & 0b0100) == 4:
            side4 = watertexture    # bottom right
        else: side4 = None
        
        img = _build_full_block(top,side1,side2,side3,side4)
        
        return (img.convert("RGB"),img.split()[3])


    if blockID == 17: # wood: normal, birch and pines
        top = terrain_images[21]
        if data == 0:
            side = terrain_images[20]
            img = _build_block(top, side, 17)
            return (img.convert("RGB"), img.split()[3])
        if data == 1:
            side = terrain_images[116]
            img = _build_block(top, side, 17)
            return (img.convert("RGB"), img.split()[3])
        if data == 2:
            side = terrain_images[117]
            img = _build_block(top, side, 17)
            return (img.convert("RGB"), img.split()[3])


    if blockID == 18: # leaves
        t = tintTexture(terrain_images[52], (37, 118, 25))
        img = _build_block(t, t, 18)
        return (img.convert("RGB"), img.split()[3])


    if blockID == 26: # bed
        increment = 5
        left_face = None
        right_face = None
        if data & 0x8 == 0x8: # head of the bed
            top = terrain_images[135]
            if data & 0x00 == 0x00: # head pointing to West
                top = top.copy().rotate(270)
                left_face = terrain_images[151]
                right_face = terrain_images[152]
            if data & 0x01 == 0x01: # ... North
                top = top.rotate(270)
                left_face = terrain_images[152]
                right_face = terrain_images[151]
            if data & 0x02 == 0x02: # East
                top = top.rotate(180)
                left_face = terrain_images[151].transpose(Image.FLIP_LEFT_RIGHT)
                right_face = None
            if data & 0x03 == 0x03: # South
                right_face = None
                right_face = terrain_images[151].transpose(Image.FLIP_LEFT_RIGHT)

        else: # foot of the bed
            top = terrain_images[134]
            if data & 0x00 == 0x00: # head pointing to West
                top = top.rotate(270)
                left_face = terrain_images[150]
                right_face = None
            if data & 0x01 == 0x01: # ... North
                top = top.rotate(270)
                left_face = None
                right_face = terrain_images[150]
            if data & 0x02 == 0x02: # East
                top = top.rotate(180)
                left_face = terrain_images[150].transpose(Image.FLIP_LEFT_RIGHT)
                right_face = terrain_images[149].transpose(Image.FLIP_LEFT_RIGHT)
            if data & 0x03 == 0x03: # South
                left_face = terrain_images[149]
                right_face = terrain_images[150].transpose(Image.FLIP_LEFT_RIGHT)

        top = (top, increment)
        img = _build_full_block(top, None, None, left_face, right_face)

        return (img.convert("RGB"), img.split()[3])


    if blockID == 35: # wool
        if data == 0: # white
            top = side = terrain_images[64]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 1: # orange
            top = side = terrain_images[210]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 2: # magenta
            top = side = terrain_images[194]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 3: # light blue
            top = side = terrain_images[178]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 4: # yellow
            top = side = terrain_images[162]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 5: # light green
            top = side = terrain_images[146]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 6: # pink
            top = side = terrain_images[130]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 7: # grey
            top = side = terrain_images[114]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 8: # light grey
            top = side = terrain_images[225]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 9: # cyan
            top = side = terrain_images[209]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 10: # purple
            top = side = terrain_images[193]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 11: # blue
            top = side = terrain_images[177]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 12: # brown
            top = side = terrain_images[161]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 13: # dark green
            top = side = terrain_images[145]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 14: # red
            top = side = terrain_images[129]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])
        if data == 15: # black
            top = side = terrain_images[113]
            img = _build_block(top, side, 35)
            return (img.convert("RGB"), img.split()[3])


    if blockID in (43,44): # slab and double-slab
        
        if data == 0: # stone slab
            top = terrain_images[6]
            side = terrain_images[5]
            img = _build_block(top, side, blockID)
            return (img.convert("RGB"), img.split()[3])
            
        if data == 1: # stone slab
            top = terrain_images[176]
            side = terrain_images[192]
            img = _build_block(top, side, blockID)
            return (img.convert("RGB"), img.split()[3])
            
        if data == 2: # wooden slab
            top = side = terrain_images[4]
            img = _build_block(top, side, blockID)
            return (img.convert("RGB"), img.split()[3])
            
        if data == 3: # cobblestone slab
            top = side = terrain_images[16]
            img = _build_block(top, side, blockID)
            return (img.convert("RGB"), img.split()[3])


    if blockID in (50,75,76): # torch, off redstone torch, on redstone torch
    
        # choose the proper texture
        if blockID == 50: # torch
            small = terrain_images[80]
        elif blockID == 75: # off redstone torch
            small = terrain_images[115]
        else: # on redstone torch
            small = terrain_images[99]
        
        # compose a torch bigger than the normal
        # (better for doing transformations)
        torch = Image.new("RGBA", (16,16), (38,92,255,0))
        composite.alpha_over(torch,small,(-4,-3))
        composite.alpha_over(torch,small,(-5,-2))
        composite.alpha_over(torch,small,(-3,-2))
        
        # angle of inclination of the texture
        rotation = 15
        
        if data == 1: # pointing south
            torch = torch.rotate(-rotation, Image.NEAREST) # nearest filter is more nitid.
            img = _build_full_block(None, None, None, torch, None, None, blockID)
            
        elif data == 2: # pointing north
            torch = torch.rotate(rotation, Image.NEAREST)
            img = _build_full_block(None, None, torch, None, None, None, blockID)
            
        elif data == 3: # pointing west
            torch = torch.rotate(rotation, Image.NEAREST)
            img = _build_full_block(None, torch, None, None, None, None, blockID)
            
        elif data == 4: # pointing east
            torch = torch.rotate(-rotation, Image.NEAREST)
            img = _build_full_block(None, None, None, None, torch, None, blockID)
            
        elif data == 5: # standing on the floor
            # compose a "3d torch".
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            
            small_crop = small.crop((2,2,14,14))
            slice = small_crop.copy()
            ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
            ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
            
            composite.alpha_over(img, slice, (6,4))
            composite.alpha_over(img, small_crop, (5,5))
            composite.alpha_over(img, small_crop, (6,5))
            composite.alpha_over(img, slice, (6,6))

        return (img.convert("RGB"), img.split()[3])


    if blockID == 51: # fire
        firetexture = _load_image("fire.png")
        side1 = transform_image_side(firetexture)
        side2 = transform_image_side(firetexture).transpose(Image.FLIP_LEFT_RIGHT)
        
        img = Image.new("RGBA", (24,24), (38,92,255,0))

        composite.alpha_over(img, side1, (12,0), side1)
        composite.alpha_over(img, side2, (0,0), side2)

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        
        return (img.convert("RGB"), img.split()[3])


    if blockID in (53,67): # wooden and cobblestone stairs.
        
        if blockID == 53: # wooden
            texture = terrain_images[4]
            
        elif blockID == 67: # cobblestone
            texture = terrain_images[16]
        
        side = texture.copy()
        half_block_u = texture.copy() # up, down, left, right
        half_block_d = texture.copy()
        half_block_l = texture.copy()
        half_block_r = texture.copy()

        # generate needed geometries
        ImageDraw.Draw(side).rectangle((0,0,7,6),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(half_block_u).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(half_block_d).rectangle((0,0,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(half_block_l).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(half_block_r).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
        
        if data == 0: # ascending south
            img = _build_full_block(half_block_r, None, None, half_block_d, side.transpose(Image.FLIP_LEFT_RIGHT))
            tmp1 = transform_image_side(half_block_u)
            
            # Darken the vertical part of the second step
            sidealpha = tmp1.split()[3]
            # darken it a bit more than usual, looks better
            tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.8)
            tmp1.putalpha(sidealpha)
            
            composite.alpha_over(img, tmp1, (6,3))
            tmp2 = transform_image(half_block_l)
            composite.alpha_over(img, tmp2, (0,6))
            
        elif data == 1: # ascending north
            img = Image.new("RGBA", (24,24), (38,92,255,0)) # first paste the texture in the back
            tmp1 = transform_image(half_block_r)
            composite.alpha_over(img, tmp1, (0,6))
            tmp2 = _build_full_block(half_block_l, None, None, texture, side)
            composite.alpha_over(img, tmp2)
        
        elif data == 2: # ascending west
            img = Image.new("RGBA", (24,24), (38,92,255,0)) # first paste the texture in the back
            tmp1 = transform_image(half_block_u)
            composite.alpha_over(img, tmp1, (0,6))
            tmp2 = _build_full_block(half_block_d, None, None, side, texture)
            composite.alpha_over(img, tmp2)
            
        elif data == 3: # ascending east
            img = _build_full_block(half_block_u, None, None, side.transpose(Image.FLIP_LEFT_RIGHT), half_block_d)
            tmp1 = transform_image_side(half_block_u).transpose(Image.FLIP_LEFT_RIGHT)
            
            # Darken the vertical part of the second step
            sidealpha = tmp1.split()[3]
            # darken it a bit more than usual, looks better
            tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.7)
            tmp1.putalpha(sidealpha)
            
            composite.alpha_over(img, tmp1, (6,3))
            tmp2 = transform_image(half_block_d)
            composite.alpha_over(img, tmp2, (0,6))
            
            # touch up a (horrible) pixel
            img.putpixel((18,3),(0,0,0,0))
            
        return (img.convert("RGB"), img.split()[3])

    if blockID == 54: # chests
        # First to bits of the pseudo data store if it's a single chest
        # or it's a double chest, first half or second half.
        # The to last bits store the orientation.

        top = terrain_images[25]
        side = terrain_images[26]

        if data & 12 == 0: # single chest
            front = terrain_images[27]
            back = terrain_images[26]

        elif data & 12 == 4: # double, first half
            front = terrain_images[41]
            back = terrain_images[57]

        elif data & 12 == 8: # double, second half
            front = terrain_images[42]
            back = terrain_images[58]

        else: # just in case
            front = terrain_images[25]
            side = terrain_images[25]
            back = terrain_images[25]

        if data & 3 == 0: # facing west
            img = _build_full_block(top, None, None, side, front)

        elif data & 3 == 1: # north
            img = _build_full_block(top, None, None, front, side)

        elif data & 3 == 2: # east
            img = _build_full_block(top, None, None, side, back)

        elif data & 3 == 3: # south
            img = _build_full_block(top, None, None, back, side)
            
        else:
            img = _build_full_block(top, None, None, back, side)

        return (img.convert("RGB"), img.split()[3])


    if blockID == 55: # redstone wire
        
        if data & 0b1000000 == 64: # powered redstone wire
            redstone_wire_t = terrain_images[165]
            redstone_wire_t = tintTexture(redstone_wire_t,(255,0,0))

            redstone_cross_t = terrain_images[164]
            redstone_cross_t = tintTexture(redstone_cross_t,(255,0,0))

            
        else: # unpowered redstone wire
            redstone_wire_t = terrain_images[165]
            redstone_wire_t = tintTexture(redstone_wire_t,(48,0,0))
            
            redstone_cross_t = terrain_images[164]
            redstone_cross_t = tintTexture(redstone_cross_t,(48,0,0))

        # generate an image per redstone direction
        branch_top_left = redstone_cross_t.copy()
        ImageDraw.Draw(branch_top_left).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_top_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_top_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        
        branch_top_right = redstone_cross_t.copy()
        ImageDraw.Draw(branch_top_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_top_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_top_right).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        
        branch_bottom_right = redstone_cross_t.copy()
        ImageDraw.Draw(branch_bottom_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_bottom_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_bottom_right).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        branch_bottom_left = redstone_cross_t.copy()
        ImageDraw.Draw(branch_bottom_left).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_bottom_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(branch_bottom_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
                
        # generate the bottom texture
        if data & 0b111111 == 0:
            bottom = redstone_cross_t.copy()
        
        elif data & 0b1111 == 10: #= 0b1010 redstone wire in the x direction
            bottom = redstone_wire_t.copy()
            
        elif data & 0b1111 == 5: #= 0b0101 redstone wire in the y direction
            bottom = redstone_wire_t.copy().rotate(90)
        
        else:
            bottom = Image.new("RGBA", (16,16), (38,92,255,0))
            if (data & 0b0001) == 1:
                composite.alpha_over(bottom,branch_top_left)
                
            if (data & 0b1000) == 8:
                composite.alpha_over(bottom,branch_top_right)
                
            if (data & 0b0010) == 2:
                composite.alpha_over(bottom,branch_bottom_left)
                
            if (data & 0b0100) == 4:
                composite.alpha_over(bottom,branch_bottom_right)

        # check for going up redstone wire
        if data & 0b100000 == 32:
            side1 = redstone_wire_t.rotate(90)
        else:
            side1 = None
            
        if data & 0b010000 == 16:
            side2 = redstone_wire_t.rotate(90)
        else:
            side2 = None
            
        img = _build_full_block(None,side1,side2,None,None,bottom)

        return (img.convert("RGB"),img.split()[3])


    if blockID == 58: # crafting table
        top = terrain_images[43]
        side3 = terrain_images[43+16]
        side4 = terrain_images[43+16+1]
        
        img = _build_full_block(top, None, None, side3, side4, None, 58)
        return (img.convert("RGB"), img.split()[3])


    if blockID == 59: # crops
        raw_crop = terrain_images[88+data]
        crop1 = transform_image(raw_crop, blockID)
        crop2 = transform_image_side(raw_crop, blockID)
        crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        composite.alpha_over(img, crop1, (0,12), crop1)
        composite.alpha_over(img, crop2, (6,3), crop2)
        composite.alpha_over(img, crop3, (6,3), crop3)
        return (img.convert("RGB"), img.split()[3])


    if blockID in (61, 62, 23): #furnace and burning furnace
        top = terrain_images[62]
        side = terrain_images[45]

        if blockID == 61:
            front = terrain_images[44]

        elif blockID == 62:
            front = terrain_images[45+16]

        elif blockID == 23:
            front = terrain_images[46]

        if data == 3: # pointing west
            img = _build_full_block(top, None, None, side, front)

        elif data == 4: # pointing north
            img = _build_full_block(top, None, None, front, side)

        else: # in any other direction the front can't be seen
            img = _build_full_block(top, None, None, side, side)

        return (img.convert("RGB"), img.split()[3])


    if blockID in (64,71): #wooden door, or iron door
        if data & 0x8 == 0x8: # top of the door
            raw_door = terrain_images[81 if blockID == 64 else 82]
        else: # bottom of the door
            raw_door = terrain_images[97 if blockID == 64 else 98]
        
        # if you want to render all doors as closed, then force
        # force swung to be False
        if data & 0x4 == 0x4:
            swung=True
        else:
            swung=False

        # mask out the high bits to figure out the orientation 
        img = Image.new("RGBA", (24,24), (38,92,255,0))
        if (data & 0x03) == 0:
            if not swung:
                tex = transform_image_side(raw_door)
                composite.alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (0,0), tex)
        
        if (data & 0x03) == 1:
            if not swung:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (0,0), tex)
            else:
                tex = transform_image_side(raw_door)
                composite.alpha_over(img, tex, (12,0), tex)

        if (data & 0x03) == 2:
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                composite.alpha_over(img, tex, (12,0), tex)
            else:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (12,6), tex)

        if (data & 0x03) == 3:
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT)).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (12,6), tex)
            else:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                composite.alpha_over(img, tex, (0,6), tex)
        
        return (img.convert("RGB"), img.split()[3])


    if blockID == 65: # ladder
        raw_texture = terrain_images[83]
        #print "ladder is facing: %d" % data
        if data == 5:
            # normally this ladder would be obsured by the block it's attached to
            # but since ladders can apparently be placed on transparent blocks, we 
            # have to render this thing anyway.  same for data == 2
            tex = transform_image_side(raw_texture)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, tex, (0,6), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 2:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, tex, (12,6), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 3:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, tex, (0,0), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 4:
            tex = transform_image_side(raw_texture)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, tex, (12,0), tex)
            return (img.convert("RGB"), img.split()[3])


    if blockID in (27, 28, 66): # minetrack:
    
        if blockID == 27: # powered rail
            if data & 0x8 == 0: # unpowered
                raw_straight = terrain_images[163]
                raw_corner = terrain_images[112]    # they don't exist but make the code
                                                    # much simplier
            elif data & 0x8 == 0x8: # powered
                raw_straight = terrain_images[179]
                raw_corner = terrain_images[112]    # leave corners for code simplicity
            # filter the 'powered' bit
            data = data & 0x7
            
        elif blockID == 28: # detector rail
            raw_straight = terrain_images[195]
            raw_corner = terrain_images[112]    # leave corners for code simplicity

        elif blockID == 66: # normal rail
            raw_straight = terrain_images[128]
            raw_corner = terrain_images[112]

        ## use transform_image to scale and shear
        if data == 0:
            track = transform_image(raw_straight, blockID)
        elif data == 6:
            track = transform_image(raw_corner, blockID)
        elif data == 7:
            track = transform_image(raw_corner.rotate(270), blockID)
        elif data == 8:
            # flip
            track = transform_image(raw_corner.transpose(Image.FLIP_TOP_BOTTOM).rotate(90), 
                    blockID)
        elif data == 9:
            track = transform_image(raw_corner.transpose(Image.FLIP_TOP_BOTTOM), 
                    blockID)
        elif data == 1:
            track = transform_image(raw_straight.rotate(90), blockID)
            
        #slopes
        elif data == 2: # slope going up in +x direction
            track = transform_image_slope(raw_straight,blockID)
            track = track.transpose(Image.FLIP_LEFT_RIGHT)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, track, (2,0), track)
            # the 2 pixels move is needed to fit with the adjacent tracks
            return (img.convert("RGB"), img.split()[3])

        elif data == 3: # slope going up in -x direction
            # tracks are sprites, in this case we are seeing the "side" of 
            # the sprite, so draw a line to make it beautiful.
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            ImageDraw.Draw(img).line([(11,11),(23,17)],fill=(164,164,164))
            # grey from track texture (exterior grey).
            # the track doesn't start from image corners, be carefull drawing the line!
            return (img.convert("RGB"), img.split()[3])
            
        elif data == 4: # slope going up in -y direction
            track = transform_image_slope(raw_straight,blockID)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            composite.alpha_over(img, track, (0,0), track)
            return (img.convert("RGB"), img.split()[3])
            
        elif data == 5: # slope going up in +y direction
            # same as "data == 3"
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            ImageDraw.Draw(img).line([(1,17),(12,11)],fill=(164,164,164))
            return (img.convert("RGB"), img.split()[3])
            

        else: # just in case
            track = transform_image(raw_straight, blockID)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        composite.alpha_over(img, track, (0,12), track)

        return (img.convert("RGB"), img.split()[3])



    if blockID == 85: # fences
        # create needed images for Big stick fence
        raw_texture = terrain_images[4]
        raw_fence_top = Image.new("RGBA", (16,16), (38,92,255,0))
        raw_fence_side = Image.new("RGBA", (16,16), (38,92,255,0))
        fence_top_mask = Image.new("RGBA", (16,16), (38,92,255,0))
        fence_side_mask = Image.new("RGBA", (16,16), (38,92,255,0))
        
        # generate the masks images for textures of the fence
        ImageDraw.Draw(fence_top_mask).rectangle((6,6,9,9),outline=(0,0,0),fill=(0,0,0))
        ImageDraw.Draw(fence_side_mask).rectangle((6,1,9,15),outline=(0,0,0),fill=(0,0,0))

        # create textures top and side for fence big stick
        composite.alpha_over(raw_fence_top,raw_texture,(0,0),fence_top_mask)
        composite.alpha_over(raw_fence_side,raw_texture,(0,0),fence_side_mask)

        # Create the sides and the top of the big stick
        fence_side = transform_image_side(raw_fence_side,85)
        fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
        fence_top = transform_image(raw_fence_top,85)

        # Darken the sides slightly. These methods also affect the alpha layer,
        # so save them first (we don't want to "darken" the alpha layer making
        # the block transparent)
        sidealpha = fence_side.split()[3]
        fence_side = ImageEnhance.Brightness(fence_side).enhance(0.9)
        fence_side.putalpha(sidealpha)
        othersidealpha = fence_other_side.split()[3]
        fence_other_side = ImageEnhance.Brightness(fence_other_side).enhance(0.8)
        fence_other_side.putalpha(othersidealpha)

        # Compose the fence big stick
        fence_big = Image.new("RGBA", (24,24), (38,92,255,0))
        composite.alpha_over(fence_big,fence_side, (5,4),fence_side)
        composite.alpha_over(fence_big,fence_other_side, (7,4),fence_other_side)
        composite.alpha_over(fence_big,fence_top, (0,1),fence_top)
        
        # Now render the small sticks.
        # Create needed images
        raw_fence_small_side = Image.new("RGBA", (16,16), (38,92,255,0))
        fence_small_side_mask = Image.new("RGBA", (16,16), (38,92,255,0))
        
        # Generate mask
        ImageDraw.Draw(fence_small_side_mask).rectangle((10,1,15,3),outline=(0,0,0),fill=(0,0,0))
        ImageDraw.Draw(fence_small_side_mask).rectangle((10,7,15,9),outline=(0,0,0),fill=(0,0,0))
        
         # create the texture for the side of small sticks fence
        composite.alpha_over(raw_fence_small_side,raw_texture,(0,0),fence_small_side_mask)
        
        # Create the sides and the top of the small sticks
        fence_small_side = transform_image_side(raw_fence_small_side,85)
        fence_small_other_side = fence_small_side.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Darken the sides slightly. These methods also affect the alpha layer,
        # so save them first (we don't want to "darken" the alpha layer making
        # the block transparent)
        sidealpha = fence_small_other_side.split()[3]
        fence_small_other_side = ImageEnhance.Brightness(fence_small_other_side).enhance(0.9)
        fence_small_other_side.putalpha(sidealpha)
        sidealpha = fence_small_side.split()[3]
        fence_small_side = ImageEnhance.Brightness(fence_small_side).enhance(0.9)
        fence_small_side.putalpha(sidealpha)


       # Create img to compose the fence

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        # Position of fence small sticks in img.
        # These postitions are strange because the small sticks of the 
        # fence are at the very left and at the very right of the 16x16 images
        pos_top_left = (-2,0)
        pos_top_right = (14,0)
        pos_bottom_right = (6,4)
        pos_bottom_left = (6,4)
        
        # +x axis points top right direction
        # +y axis points bottom right direction
        # First compose small sticks in the back of the image, 
        # then big stick and thecn small sticks in the front.

        if (data & 0b0001) == 1:
            composite.alpha_over(img,fence_small_side, pos_top_left,fence_small_side)                # top left
        if (data & 0b1000) == 8:
            composite.alpha_over(img,fence_small_other_side, pos_top_right,fence_small_other_side)    # top right
            
        composite.alpha_over(img,fence_big,(0,0),fence_big)
            
        if (data & 0b0010) == 2:
            composite.alpha_over(img,fence_small_other_side, pos_bottom_left,fence_small_other_side)      # bottom left    
        if (data & 0b0100) == 4:
            composite.alpha_over(img,fence_small_side, pos_bottom_right,fence_small_side)                  # bottom right
            
        return (img.convert("RGB"),img.split()[3])


    if blockID in (86,91): # pumpkins, jack-o-lantern
        top = terrain_images[102]
        frontID = 119 if blockID == 86 else 120
        front = terrain_images[frontID]
        side = terrain_images[118]

        if data == 0: # pointing west
            img = _build_full_block(top, None, None, side, front)

        elif data == 1: # pointing north
            img = _build_full_block(top, None, None, front, side)

        else: # in any other direction the front can't be seen
            img = _build_full_block(top, None, None, side, side)

        return (img.convert("RGB"), img.split()[3])


    if blockID == 90: # portal
        portaltexture = _load_image("portal.png")
        img = Image.new("RGBA", (24,24), (38,92,255,0))

        side = transform_image_side(portaltexture)
        otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

        if data in (1,4):
            composite.alpha_over(img, side, (5,4), side)

        if data in (2,8):
            composite.alpha_over(img, otherside, (5,4), otherside)

        return (img.convert("RGB"), img.split()[3])


    if blockID == 92: # cake! (without bites, at the moment)
    
        top = terrain_images[121]
        side = terrain_images[122]
        top = transform_image(top, blockID)
        side = transform_image_side(side, blockID)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
        
        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)
        
        img = Image.new("RGBA", (24,24), (38,92,255,0))
        
        composite.alpha_over(img, side, (1,12), side)
        composite.alpha_over(img, otherside, (11,13), otherside) # workaround, fixes a hole
        composite.alpha_over(img, otherside, (12,12), otherside)
        composite.alpha_over(img, top, (0,6), top)

        return (img.convert("RGB"), img.split()[3])


    if blockID in (93, 94): # redstone repeaters, ON and OFF
        # NOTE: this function uses the redstone torches generated above,
        # this must run after the function of the torches.

        top = terrain_images[131] if blockID == 93 else terrain_images[147]
        side = terrain_images[5]
        increment = 9
        
        if (data & 0x3) == 0: # pointing east
            pass
        
        if (data & 0x3) == 1: # pointing south
            top = top.rotate(270)

        if (data & 0x3) == 2: # pointing west
            top = top.rotate(180)

        if (data & 0x3) == 3: # pointing north
            top = top.rotate(90)

        img = _build_full_block( (top, increment), None, None, side, side)
        
        # paste redstone torches everywhere!
        t = specialblockmap[(75,5)] if blockID == 93 else specialblockmap[(76,5)]
        torch = t[0].copy()        # textures are stored as tuples (RGB,A)
        torch.putalpha(t[1])
        
        # the torch is too tall for the repeater, crop the bottom.
        ImageDraw.Draw(torch).rectangle((0,16,24,24),outline=(0,0,0,0),fill=(0,0,0,0))
        
        # touch up the 3d effect with big rectangles, just in case, for other texture packs
        ImageDraw.Draw(torch).rectangle((0,24,10,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(torch).rectangle((12,15,24,24),outline=(0,0,0,0),fill=(0,0,0,0))
        
        # torch positions for every redstone torch orientation.
        #
        # This is a horrible list of torch orientations. I tried to 
        # obtain these orientations by rotating the positions for one
        # orientation, but pixel rounding is horrible and messes the
        # torches.

        if (data & 0x3) == 0: # pointing east
            if (data & 0xC) == 0: # one tick delay
                moving_torch = (1,1)
                static_torch = (-3,-1)
                
            elif (data & 0xC) == 4: # two ticks delay
                moving_torch = (2,2)
                static_torch = (-3,-1)
                
            elif (data & 0xC) == 8: # three ticks delay
                moving_torch = (3,2)
                static_torch = (-3,-1)
                
            elif (data & 0xC) == 12: # four ticks delay
                moving_torch = (4,3)
                static_torch = (-3,-1)
        
        elif (data & 0x3) == 1: # pointing south
            if (data & 0xC) == 0: # one tick delay
                moving_torch = (1,1)
                static_torch = (5,-1)
                
            elif (data & 0xC) == 4: # two ticks delay
                moving_torch = (2,0)
                static_torch = (5,-1)
                
            elif (data & 0xC) == 8: # three ticks delay
                moving_torch = (3,0)
                static_torch = (5,-1)
                
            elif (data & 0xC) == 12: # four ticks delay
                moving_torch = (4,-1)
                static_torch = (5,-1)

        elif (data & 0x3) == 2: # pointing west
            if (data & 0xC) == 0: # one tick delay
                moving_torch = (1,1)
                static_torch = (5,3)
                
            elif (data & 0xC) == 4: # two ticks delay
                moving_torch = (0,0)
                static_torch = (5,3)
                
            elif (data & 0xC) == 8: # three ticks delay
                moving_torch = (-1,0)
                static_torch = (5,3)
                
            elif (data & 0xC) == 12: # four ticks delay
                moving_torch = (-2,-1)
                static_torch = (5,3)

        elif (data & 0x3) == 3: # pointing north
            if (data & 0xC) == 0: # one tick delay
                moving_torch = (1,1)
                static_torch = (-3,3)
                
            elif (data & 0xC) == 4: # two ticks delay
                moving_torch = (2,0)
                static_torch = (-3,3)
                
            elif (data & 0xC) == 8: # three ticks delay
                moving_torch = (3,0)
                static_torch = (-3,3)
                
            elif (data & 0xC) == 12: # four ticks delay
                moving_torch = (4,-1)
                static_torch = (-3,3)
        
        # this paste order it's ok for east and south orientation
        # but it's wrong for north and west orientations. But using the
        # default texture pack the torches are small enough to no overlap.
        composite.alpha_over(img, torch, static_torch, torch) 
        composite.alpha_over(img, torch, moving_torch, torch)

        return (img.convert("RGB"), img.split()[3])


    return None

def tintTexture(im, c):
    # apparently converting to grayscale drops the alpha channel?
    i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
    i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
    return i

# generate biome (still grayscale) leaf, grass textures
biome_grass_texture = _build_block(terrain_images[0], terrain_images[38], 2)
biome_leaf_texture = _build_block(terrain_images[52], terrain_images[52], 18)


currentBiomeFile = None
currentBiomeData = None
grasscolor = None
foliagecolor = None

def prepareBiomeData(worlddir):
    global grasscolor, foliagecolor
    
    # skip if the color files are already loaded
    if grasscolor and foliagecolor:
        return
    
    biomeDir = os.path.join(worlddir, "biomes")
    if not os.path.exists(biomeDir):
        raise Exception("biomes not found")

    # try to find the biome color images.  If _find_file can't locate them
    # then try looking in the EXTRACTEDBIOMES folder
    try:
        grasscolor = list(Image.open(_find_file("grasscolor.png")).getdata())
        foliagecolor = list(Image.open(_find_file("foliagecolor.png")).getdata())
    except IOError:
        try:
            grasscolor = list(Image.open(os.path.join(biomeDir,"grasscolor.png")).getdata())
            foliagecolor = list(Image.open(os.path.join(biomeDir,"foliagecolor.png")).getdata())
        except:
            # clear anything that managed to get set
            grasscolor = None
            foliagecolor = None

def getBiomeData(worlddir, chunkX, chunkY):
    '''Opens the worlddir and reads in the biome color information
    from the .biome files.  See also:
    http://www.minecraftforum.net/viewtopic.php?f=25&t=80902
    '''

    global currentBiomeFile, currentBiomeData

    biomeFile = "b.%d.%d.biome" % (chunkX // 32, chunkY // 32)
    if biomeFile == currentBiomeFile:
        return currentBiomeData

    try:
        with open(os.path.join(worlddir, "biomes", biomeFile), "rb") as f:
            rawdata = f.read()
            # make sure the file size is correct
            if not len(rawdata) == 512 * 512 * 2:
                raise Exception("Biome file %s is not valid." % (biomeFile,))
            data = numpy.frombuffer(rawdata, dtype=numpy.dtype(">u2"))
    except IOError:
        data = None
        pass # no biome data   

    currentBiomeFile = biomeFile
    currentBiomeData = data
    return data

# This set holds block ids that require special pre-computing.  These are typically
# things that require ancillary data to render properly (i.e. ladder plus orientation)
# A good source of information is:
#  http://www.minecraftwiki.net/wiki/Data_values
# (when adding new blocks here and in generate_special_textures,
# please, if possible, keep the ascending order of blockid value)

special_blocks = set([ 2,  6,  9, 17, 18, 26, 23, 27, 28, 35, 43, 44, 50,
                      51, 53, 54, 55, 58, 59, 61, 62, 64, 65, 66, 67, 71,
                      75, 76, 85, 86, 90, 91, 92, 93, 94])

# this is a map of special blockIDs to a list of all 
# possible values for ancillary data that it might have.

special_map = {}

special_map[6] = range(16)  # saplings: usual, spruce, birch and future ones (rendered as usual saplings)
special_map[9] = range(32)  # water: spring,flowing, waterfall, and others (unknown) ancildata values, uses pseudo data
special_map[17] = range(4)  # wood: normal, birch and pine
special_map[26] = range(12) # bed, orientation
special_map[23] = range(6)  # dispensers, orientation
special_map[27] = range(14) # powered rail, orientation/slope and powered/unpowered
special_map[28] = range(6) # detector rail, orientation/slope
special_map[35] = range(16) # wool, colored and white
special_map[43] = range(4)  # stone, sandstone, wooden and cobblestone double-slab
special_map[44] = range(4)  # stone, sandstone, wooden and cobblestone slab
special_map[50] = (1,2,3,4,5) # torch, position in the block
special_map[51] = range(16) # fire, position in the block (not implemented)
special_map[53] = range(4)  # wooden stairs, orientation
special_map[54] = range(12) # chests, orientation and type (single or double), uses pseudo data
special_map[55] = range(128) # redstone wire, all the possible combinations, uses pseudo data
special_map[58] = (0,)      # crafting table
special_map[59] = range(8)  # crops, grow from 0 to 7
special_map[61] = range(6)  # furnace, orientation
special_map[62] = range(6)  # burning furnace, orientation
special_map[64] = range(16) # wooden door, open/close and orientation
special_map[65] = (2,3,4,5) # ladder, orientation
special_map[66] = range(10) # minecrart tracks, orientation, slope
special_map[67] = range(4)  # cobblestone stairs, orientation
special_map[71] = range(16) # iron door, open/close and orientation
special_map[75] = (1,2,3,4,5) # off redstone torch, orientation
special_map[76] = (1,2,3,4,5) # on redstone torch, orientation
special_map[85] = range(17) # fences, all the possible combination, uses pseudo data
special_map[86] = range(5)  # pumpkin, orientation
special_map[90] = (1,2,4,8) # portal, in 2 orientations, 4 cases, uses pseudo data
special_map[91] = range(5)  # jack-o-lantern, orientation
special_map[92] = range(6) # cake!
special_map[93] = range(16) # OFF redstone repeater, orientation and delay (delay not implemented)
special_map[94] = range(16) # ON redstone repeater, orientation and delay (delay not implemented)

# grass and leaves are graysacle in terrain.png
# we treat them as special so we can manually tint them
# it is unknown how the specific tint (biomes) is calculated
# also, 0x10 means SNOW sides
special_map[2] = range(11) + [0x10,]  # grass, grass has not ancildata but is
                                      # used in the mod WildGrass, and this
                                      # small fix shows the map as expected,
                                      # and is harmless for normal maps
special_map[18] = range(16) # leaves, birch, normal or pine leaves (not implemented)


specialblockmap = {}

for blockID in special_blocks:
    for data in special_map[blockID]:
        specialblockmap[(blockID, data)] = generate_special_texture(blockID, data)
