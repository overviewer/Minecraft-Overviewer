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
from PIL import Image, ImageEnhance

import util

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

    if blockID in (81,): # cacti
        # Resize to 15x15, since the cactus texture is a little smaller than the other textures
        img = img.resize((15, 15), Image.BILINEAR)

    else:
        # Resize to 17x17, since the diagonal is approximately 24 pixels, a nice
        # even number that can be split in half twice
        img = img.resize((17, 17), Image.BILINEAR)

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
        n.paste(mask,(0,0,16,8), mask)
        img = n
    if blockID in (78,): # snow
        # make the top three quarters transparent
        mask = img.crop((0,12,16,16))
        n = Image.new(img.mode, img.size, (38,92,255,0))
        n.paste(mask,(0,12,16,16), mask)
        img = n

    # Size of the cube side before shear
    img = img.resize((12,12))

    # Apply shear
    transform = numpy.matrix(numpy.identity(3))
    transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

    transform = numpy.array(transform)[:2,:].ravel().tolist()

    newimg = img.transform((12,18), Image.AFFINE, transform)
    return newimg


def _build_block(top, side, blockID=None):
    """From a top texture and a side texture, build a block image.
    top and side should be 16x16 image objects. Returns a 24x24 image

    """
    img = Image.new("RGBA", (24,24), (38,92,255,0))

    top = transform_image(top, blockID)

    if not side:
        img.paste(top, (0,0), top)
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
    # TODO once torches are handled by generate_special_texture, remove
    # them from this list
    if blockID in (37,38,6,39,40,50,83): ## flowers, sapling, mushrooms,  regular torch, reeds
        # instead of pasting these blocks at the cube edges, place them in the middle:
        # and omit the top
        img.paste(side, (6,3), side)
        img.paste(otherside, (6,3), otherside)
        return img


    if blockID in (81,): # cacti!
        img.paste(side, (2,6), side)
        img.paste(otherside, (10,6), otherside)
        img.paste(top, (0,2), top)
    elif blockID in (44,): # half step
        # shift each texture down 6 pixels
        img.paste(side, (0,12), side)
        img.paste(otherside, (12,12), otherside)
        img.paste(top, (0,6), top)
    elif blockID in (78,): # snow
        # shift each texture down 9 pixels
        img.paste(side, (0,6), side)
        img.paste(otherside, (12,6), otherside)
        img.paste(top, (0,9), top)
    else:
        img.paste(side, (0,6), side)
        img.paste(otherside, (12,6), otherside)
        img.paste(top, (0,0), top)

    # Manually touch up 6 pixels that leave a gap because of how the
    # shearing works out. This makes the blocks perfectly tessellate-able
    for x,y in [(13,23), (17,21), (21,19)]:
        # Copy a pixel to x,y from x-1,y
        img.putpixel((x,y), img.getpixel((x-1,y)))
    for x,y in [(3,4), (7,2), (11,0)]:
        # Copy a pixel to x,y from x+1,y
        img.putpixel((x,y), img.getpixel((x+1,y)))

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
    topids = [ -1,  1,  0,  2, 16,  4, 15, 17,205,205,237,237, 18, 19, 32, 33,
       #       16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
               34, 21, 52, 48, 49, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, # Cloths are left out
       #       32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
               -1, -1, -1, 64, 64, 13, 12, 29, 28, 23, 22,  6,  6,  7,  8, 35, # Gold/iron blocks? Doublestep? TNT from above?
       #       48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
               36, 37, 80, -1, 65,  4, 25,101, 98, 24, 43, -1, 86,  1,  1, -1, # Torch from above? leaving out fire. Redstone wire? Crops/furnaces handled elsewhere. sign post
       #       64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
               -1, -1, -1, 16, -1, -1, -1, -1, -1, 51, 51, -1, -1,  1, 66, 67, # door,ladder left out. Minecart rail orientation
       #       80  81  82  83  84
               66, 69, 72, 73, 74 # clay?
        ]

    # NOTE: For non-block textures, the sideid is ignored, but can't be -1

    # And side textures of all block types
       #         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    sideids = [ -1,  1,  3,  2, 16,  4, 15, 17,205,205,237,237, 18, 19, 32, 33,
       #        16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
                34, 20, 52, 48, 49, -1, -1, -1, -1, -1, -1, -1,- 1, -1, -1, -1,
       #        32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
                -1, -1, -1, 64, 64, 13, 12, 29, 28, 23, 22,  5,  5,  7,  8, 35,
       #        48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
                36, 37, 80, -1, 65,  4, 25,101, 98, 24, 43, -1, 86, 44, 61, -1,
       #        64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
                -1, -1, -1, 16, -1, -1, -1, -1, -1, 51, 51, -1, -1,  1, 66, 67,
       #        80  81  82  83  84
                66, 69, 72, 73, 74
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
    # TODO torches, redstone torches, crops, ladders, stairs, 
    #      levers, doors, buttons, and signs all need to be handled here (and in chunkpy)
    if blockID == 66: # minetrack:
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
        else:
            # TODO render carts that slop up or down
            track = transform_image(raw_straight, blockID)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        img.paste(track, (0,12), track)

        return (img.convert("RGB"), img.split()[3])
    if blockID == 59: # crops
        raw_crop = terrain_images[88+data]
        crop1 = transform_image(raw_crop, blockID)
        crop2 = transform_image_side(raw_crop, blockID)
        crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        img.paste(crop1, (0,12), crop1)
        img.paste(crop2, (6,3), crop2)
        img.paste(crop3, (6,3), crop3)
        return (img.convert("RGB"), img.split()[3])

    if blockID == 61: #furnace
        top = transform_image(terrain_images[1])
        side1 = transform_image_side(terrain_images[45])
        side2 = transform_image_side(terrain_images[44]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        img.paste(side1, (0,6), side1)
        img.paste(side2, (12,6), side2)
        img.paste(top, (0,0), top)
        return (img.convert("RGB"), img.split()[3])
    
    if blockID == 62: # lit furnace
        top = transform_image(terrain_images[1])
        side1 = transform_image_side(terrain_images[45])
        side2 = transform_image_side(terrain_images[45+16]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        img.paste(side1, (0,6), side1)
        img.paste(side2, (12,6), side2)
        img.paste(top, (0,0), top)
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
            img.paste(tex, (0,6), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 2:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            img.paste(tex, (12,6), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 3:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            img.paste(tex, (0,0), tex)
            return (img.convert("RGB"), img.split()[3])
        if data == 4:
            tex = transform_image_side(raw_texture)
            img = Image.new("RGBA", (24,24), (38,92,255,0))
            img.paste(tex, (12,0), tex)
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
                img.paste(tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                img.paste(tex, (0,0), tex)
        
        if (data & 0x03) == 1:
            if not swung:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                img.paste(tex, (0,0), tex)
            else:
                tex = transform_image_side(raw_door)
                img.paste(tex, (12,0), tex)

        if (data & 0x03) == 2:
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                img.paste(tex, (12,0), tex)
            else:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                img.paste(tex, (12,6), tex)

        if (data & 0x03) == 3:
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT)).transpose(Image.FLIP_LEFT_RIGHT)
                img.paste(tex, (12,6), tex)
            else:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                img.paste(tex, (0,6), tex)
        
        return (img.convert("RGB"), img.split()[3])
    

    return None


# This set holds block ids that require special pre-computing.  These are typically
# things that require ancillary data to render properly (i.e. ladder plus orientation)
special_blocks = set([66,59,61,62, 65,64,71])

# this is a map of special blockIDs to a list of all 
# possible values for ancillary data that it might have.
special_map = {}
special_map[66] = range(10) # minecrart tracks
special_map[59] = range(8)  # crops
special_map[61] = (0,)      # furnace
special_map[62] = (0,)      # burning furnace
special_map[65] = (2,3,4,5) # ladder
special_map[64] = range(16) # wooden door
special_map[71] = range(16) # iron door

specialblockmap = {}

for blockID in special_blocks:
    for data in special_map[blockID]:
        specialblockmap[(blockID, data)] = generate_special_texture(blockID, data)
