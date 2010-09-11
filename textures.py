import sys
import os
import os.path
import zipfile
from cStringIO import StringIO
import math

import numpy
from PIL import Image, ImageEnhance

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
    programdir = os.path.dirname(__file__)
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
                "Application Support", "minecraft"))
        jarpaths.append(os.path.join(os.environ['HOME'], ".minecraft", "bin",
                "minecraft.jar"))

    for jarpath in jarpaths:
        if os.path.exists(jarpath):
            jar = zipfile.ZipFile(jarpath)
            try:
                return jar.open(filename)
            except KeyError:
                pass

    path = filename
    if os.path.exists(path):
        return open(path, mode)

    path = os.path.join(programdir, "textures", filename)
    if os.path.exists(path):
        return open(path, mode)

    raise IOError("Could not find the file {0}".format(filename))

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
    for y in xrange(16):
        for x in xrange(16):
            left = x*16
            upper = y*16
            right = left+16
            lower = upper+16
            region = terrain.crop((left,upper,right,lower))
            textures.append(region)

    return textures

# This maps terainids to 16x16 images
terrain_images = _split_terrain(_get_terrain_image())

def _transform_image(img):
    """Takes a PIL image and rotates it left 45 degrees and shrinks the y axis
    by a factor of 2. Returns the resulting image, which will be 24x12 pixels

    """

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

def _transform_image_side(img):
    """Takes an image and shears it for the left side of the cube (reflect for
    the right side)"""

    # Size of the cube side before shear
    img = img.resize((12,12))

    # Apply shear
    transform = numpy.matrix(numpy.identity(3))
    transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

    transform = numpy.array(transform)[:2,:].ravel().tolist()

    newimg = img.transform((12,18), Image.AFFINE, transform)
    return newimg


def _build_block(top, side):
    """From a top texture and a side texture, build a block image.
    top and side should be 16x16 image objects. Returns a 24x24 image

    """
    img = Image.new("RGBA", (24,24))

    top = _transform_image(top)

    if not side:
        img.paste(top, (0,0), top)
        return img

    side = _transform_image_side(side)

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
    topids = [-1,1,0,2,16,4,15,17,205,205,237,237,18,19,32,33,
        34,21,52,48,49,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, # Cloths are left out
        -1,-1,-1,64,64,13,12,29,28,23,22,6,6,7,8,35, # Gold/iron blocks? Doublestep? TNT from above?
        36,37,-1,-1,65,4,25,101,98,24,43,-1,86,1,1,-1, # Torch from above? leaving out fire. Redstone wire? Crops left out. sign post
        -1,-1,-1,16,-1,-1,-1,-1,-1,51,51,-1,-1,1,66,67, # door,ladder left out. Minecart rail orientation
        66,69,72,-1,74 # clay?
        ]

    # And side textures of all block types
    sideids = [-1,1,3,2,16,4,15,17,205,205,237,237,18,19,32,33,
        34,20,52,48,49,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,64,64,13,12,29,28,23,22,6,6,7,8,35,
        36,37,-1,-1,65,4,25,101,98,24,43,-1,86,1,1,-1,
        -1,-1,-1,16,-1,-1,-1,-1,-1,51,51,-1,-1,1,66,67,
        66,69,72,-1,74
        ]

    # This maps block id to the texture that goes on the side of the block
    allimages = []
    for toptextureid, sidetextureid in zip(topids, sideids):
        if toptextureid == -1 or sidetextureid == -1:
            allimages.append(None)
            continue

        toptexture = terrain_images[toptextureid]
        sidetexture = terrain_images[sidetextureid]

        img = _build_block(toptexture, sidetexture)

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
