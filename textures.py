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
        composite.alpha_over(n, mask,(0,0,16,8), mask)
        img = n
    if blockID in (78,): # snow
        # make the top three quarters transparent
        mask = img.crop((0,12,16,16))
        n = Image.new(img.mode, img.size, (38,92,255,0))
        composite.alpha_over(n, mask,(0,12,16,16), mask)
        img = n

    # Size of the cube side before shear
    img = img.resize((12,12))

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
    img = img.resize((12,12))

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
    # TODO once torches are handled by generate_special_texture, remove
    # them from this list
    if blockID in (37,38,6,39,40,50,83): ## flowers, sapling, mushrooms,  regular torch, reeds
        # instead of pasting these blocks at the cube edges, place them in the middle:
        # and omit the top
        composite.alpha_over(img, side, (6,3), side)
        composite.alpha_over(img, otherside, (6,3), otherside)
        return img


    if blockID in (81,): # cacti!
        composite.alpha_over(img, side, (2,6), side)
        composite.alpha_over(img, otherside, (10,6), otherside)
        composite.alpha_over(img, top, (0,2), top)
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
               34, -1, 52, 48, 49,160,144, -1,176, 74, -1, -1, -1, -1, -1, -1, # Cloths are left out, sandstone (it has top, side, and bottom wich is ignored here), note block
       #       32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
               -1, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  8, 35, # Gold/iron blocks? Doublestep? TNT from above?
       #       48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
               36, 37, 80, -1, 65,  4, 25, -1, 98, 24, 43, -1, 86, -1, -1, -1, # Torch from above? leaving out fire. Redstone wire? Crops/furnaces handled elsewhere. sign post
       #       64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
               -1, -1, -1, 16, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, 67, # door,ladder left out. Minecart rail orientation
       #       80  81  82  83  84  85  86  87  88  89  90  91
               66, 69, 72, 73, 74, -1,102,103,104,105,-1, 102 # clay?
        ]

    # NOTE: For non-block textures, the sideid is ignored, but can't be -1

    # And side textures of all block types
       #         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    sideids = [ -1,  1,  3,  2, 16,  4, 15, 17,205,205,237,237, 18, 19, 32, 33,
       #        16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
                34, -1, 52, 48, 49,160,144, -1,192, 74, -1, -1,- 1, -1, -1, -1,
       #        32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
                -1, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  8, 35,
       #        48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
                36, 37, 80, -1, 65,  4, 25,101, 98, 24, 43, -1, 86, -1, -1, -1,
       #        64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
                -1, -1, -1, 16, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, 67,
       #        80  81  82  83  84  85  86  87  88  89  90  91
                66, 69, 72, 73, 74,-1 ,118,103,104,105, -1, 118
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

    if blockID == 61: #furnace
        top = transform_image(terrain_images[62])
        side1 = transform_image_side(terrain_images[45])
        side2 = transform_image_side(terrain_images[44]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
        return (img.convert("RGB"), img.split()[3])

    if blockID in (86,91): # jack-o-lantern
        top = transform_image(terrain_images[102])
        frontID = 119 if blockID == 86 else 120
        side1 = transform_image_side(terrain_images[frontID])
        side2 = transform_image_side(terrain_images[118]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
        return (img.convert("RGB"), img.split()[3])
    
    if blockID == 62: # lit furnace
        top = transform_image(terrain_images[62])
        side1 = transform_image_side(terrain_images[45])
        side2 = transform_image_side(terrain_images[45+16]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        
        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
        return (img.convert("RGB"), img.split()[3])

    if blockID == 23: # dispenser
        top = transform_image(terrain_images[62])
        side1 = transform_image_side(terrain_images[46])
        side2 = transform_image_side(terrain_images[45]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))
        
        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
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
    
    if blockID == 2: # grass
        top = transform_image(tintTexture(terrain_images[0],(115,175,71)))
        side1 = transform_image_side(terrain_images[3])
        side2 = transform_image_side(terrain_images[3]).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
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
    
    if blockID == 18: # leaves
        t = tintTexture(terrain_images[52], (37, 118, 25))
        top = transform_image(t)
        side1 = transform_image_side(t)
        side2 = transform_image_side(t).transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), (38,92,255,0))

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        composite.alpha_over(img, top, (0,0), top)
        return (img.convert("RGB"), img.split()[3])
        
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
        composite.alpha_over(fence_big,fence_side, (4,4),fence_side)
        composite.alpha_over(fence_big,fence_other_side, (8,4),fence_other_side)
        composite.alpha_over(fence_big,fence_top, (-1,1),fence_top)
        
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

    return None

def tintTexture(im, c):
    # apparently converting to grayscale drops the alpha channel?
    i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
    i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
    return i

grassSide1 = transform_image_side(terrain_images[3])
grassSide2 = transform_image_side(terrain_images[3]).transpose(Image.FLIP_LEFT_RIGHT)
def prepareGrassTexture(color):
    top = transform_image(tintTexture(terrain_images[0],color))
    img = Image.new("RGBA", (24,24), (38,92,255,0))

    img.paste(grassSide1, (0,6), grassSide1)
    img.paste(grassSide2, (12,6), grassSide2)
    img.paste(top, (0,0), top)
    return (img.convert("RGB"), img.split()[3])


def prepareLeafTexture(color):
    t = tintTexture(terrain_images[52], color)
    top = transform_image(t)
    side1 = transform_image_side(t)
    side2 = transform_image_side(t).transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), (38,92,255,0))

    img.paste(side1, (0,6), side1)
    img.paste(side2, (12,6), side2)
    img.paste(top, (0,0), top)
    return (img.convert("RGB"), img.split()[3])



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

    currentBiomeFile = biomeFile

    f = open(os.path.join(worlddir, "biomes", biomeFile), "rb")
    rawdata = f.read()
    f.close()

    data = numpy.frombuffer(rawdata, dtype=numpy.dtype(">u2"))

    currentBiomeData = data
    return data

# This set holds block ids that require special pre-computing.  These are typically
# things that require ancillary data to render properly (i.e. ladder plus orientation)

special_blocks = set([66,59,61,62, 65,64,71,91,86,2,18,85,17,23,35,51,43,44])

# this is a map of special blockIDs to a list of all 
# possible values for ancillary data that it might have.
special_map = {}
special_map[66] = range(10) # minecrart tracks
special_map[59] = range(8)  # crops
special_map[61] = range(6)  # furnace
special_map[62] = range(6)  # burning furnace
special_map[65] = (2,3,4,5) # ladder
special_map[64] = range(16) # wooden door
special_map[71] = range(16) # iron door
special_map[91] = range(5)  # jack-o-lantern
special_map[86] = range(5)  # pumpkin
special_map[85] = range(17) # fences
special_map[17] = range(4)  # wood: normal, birch and pine
special_map[23] = range(6)  # dispensers
special_map[35] = range(16) # wool, colored and white
special_map[51] = range(16) # fire
special_map[43] = range(4)  # stone, sandstone, wooden and cobblestone double-slab
special_map[44] = range(4)  # stone, sandstone, wooden and cobblestone slab

# apparently pumpkins and jack-o-lanterns have ancillary data, but it's unknown
# what that data represents.  For now, assume that the range for data is 0 to 5
# like torches
special_map[2] = (0,)       # grass
special_map[18] = range(16) # leaves
# grass and leaves are now graysacle in terrain.png
# we treat them as special so we can manually tint them
# it is unknown how the specific tint (biomes) is calculated

# leaves have ancilary data, but its meaning is unknown (age perhaps?)

specialblockmap = {}

for blockID in special_blocks:
    for data in special_map[blockID]:
        specialblockmap[(blockID, data)] = generate_special_texture(blockID, data)
