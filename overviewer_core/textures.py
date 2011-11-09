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
import imp
import os
import os.path
import zipfile
from cStringIO import StringIO
import math
from random import randint
import numpy
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import logging

import util
import composite

_find_file_local_path = None
def _find_file(filename, mode="rb", verbose=False):
    """Searches for the given file and returns an open handle to it.
    This searches the following locations in this order:
    
    * the textures_path given in the config file (if present)
      this can be either a directory or a zip file (texture pack)
    * The program dir (same dir as overviewer.py)
    * The overviewer_core/data/textures dir
    * On Darwin, in /Applications/Minecraft
    * Inside minecraft.jar, which is looked for at these locations

      * On Windows, at %APPDATA%/.minecraft/bin/minecraft.jar
      * On Darwin, at $HOME/Library/Application Support/minecraft/bin/minecraft.jar
      * at $HOME/.minecraft/bin/minecraft.jar
    
    In all of these, files are searched for in '.', 'misc/', and 'environment/'.

    """
    
    # a list of subdirectories to search for a given file,
    # after the obvious '.'
    search_dirs = ['misc', 'environment']
    search_zip_paths = [filename,] + [d + '/' + filename for d in search_dirs]
    def search_dir(base):
        """Search the given base dir for filename, in search_dirs."""
        for path in [os.path.join(base, d, filename) for d in ['',] + search_dirs]:
            if os.path.isfile(path):
                return path
        return None
    
    if _find_file_local_path:
        if os.path.isdir(_find_file_local_path):
            path = search_dir(_find_file_local_path)
            if path:
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)
        elif os.path.isfile(_find_file_local_path):
            try:
                pack = zipfile.ZipFile(_find_file_local_path)
                for packfilename in search_zip_paths:
                    try:
                        pack.getinfo(packfilename)
                        if verbose: logging.info("Found %s in '%s'", packfilename, _find_file_local_path)
                        return pack.open(packfilename)
                    except (KeyError, IOError):
                        pass
            except (zipfile.BadZipfile, IOError):
                pass
    
    programdir = util.get_program_path()
    path = search_dir(programdir)
    if path:
        if verbose: logging.info("Found %s in '%s'", filename, path)
        return open(path, mode)
    
    path = search_dir(os.path.join(programdir, "overviewer_core", "data", "textures"))
    if path:
        if verbose: logging.info("Found %s in '%s'", filename, path)
        return open(path, mode)
    elif hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        # windows special case, when the package dir doesn't exist
        path = search_dir(os.path.join(programdir, "textures"))
        if path:
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)

    if sys.platform == "darwin":
        path = search_dir("/Applications/Minecraft")
        if path:
            if verbose: logging.info("Found %s in '%s'", filename, path)
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
    jarpaths.append(os.path.join(programdir,"minecraft.jar"))
    jarpaths.append(os.path.join(os.getcwd(), "minecraft.jar"))
    if _find_file_local_path:
        jarpaths.append(os.path.join(_find_file_local_path, "minecraft.jar"))

    for jarpath in jarpaths:
        if os.path.isfile(jarpath):
            jar = zipfile.ZipFile(jarpath)
            for jarfilename in search_zip_paths:
                try:
                    jar.getinfo(jarfilename)
                    if verbose: logging.info("Found %s in '%s'", jarfilename, jarpath)
                    return jar.open(jarfilename)
                except (KeyError, IOError), e:
                    pass
        elif os.path.isdir(jarpath):
            path = search_dir(jarpath)
            if path:
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, 'rb')

    raise IOError("Could not find the file `{0}'. You can either place it in the same place as overviewer.py, use --textures-path, or install the Minecraft client.".format(filename))

def _load_image(filename):
    """Returns an image object"""
    fileobj = _find_file(filename)
    buffer = StringIO(fileobj.read())
    return Image.open(buffer).convert("RGBA")

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
        n = Image.new(img.mode, img.size, bgcolor)
        composite.alpha_over(n, mask,(0,0,16,8), mask)
        img = n
    if blockID in (78,): # snow
        # make the top three quarters transparent
        mask = img.crop((0,12,16,16))
        n = Image.new(img.mode, img.size, bgcolor)
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


def transform_image_angle(img, angle, blockID=None):
    """Takes an image an shears it in arbitrary angle with the axis of
    rotation being vertical.
    
    WARNING! Don't use angle = pi/2 (or multiplies), it will return
    a blank image (or maybe garbage).
    
    NOTE: angle is in the image not in game, so for the left side of a
    block angle = 30 degree.
    """
    
    # Take the same size as trasform_image_side
    img = img.resize((12,12), Image.ANTIALIAS)

    # some values
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)

    # function_x and function_y are used to keep the result image in the 
    # same position, and constant_x and constant_y are the coordinates
    # for the center for angle = 0.
    constant_x = 6.
    constant_y = 6.
    function_x = 6.*(1-cos_angle)
    function_y = -6*sin_angle
    big_term = ( (sin_angle * (function_x + constant_x)) - cos_angle* (function_y + constant_y))/cos_angle

    # The numpy array is not really used, but is helpful to 
    # see the matrix used for the transformation.
    transform = numpy.array([[1./cos_angle, 0, -(function_x + constant_x)/cos_angle],
                             [-sin_angle/(cos_angle), 1., big_term ],
                             [0, 0, 1.]])

    transform = tuple(transform[0]) + tuple(transform[1])

    newimg = img.transform((24,24), Image.AFFINE, transform)

    return newimg


def _build_block(top, side, blockID=None):
    """From a top texture and a side texture, build a block image.
    top and side should be 16x16 image objects. Returns a 24x24 image

    """
    img = Image.new("RGBA", (24,24), bgcolor)
    
    original_texture = top.copy()
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

    ## special case for tall-grass, fern, dead shrub, and pumpkin/melon stem
    if blockID in (31,32,104,105):
        front = original_texture.resize((14,11), Image.ANTIALIAS)
        composite.alpha_over(img, front, (5,9))
        return img

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
        composite.alpha_over(img, top, (0,0), top)
        composite.alpha_over(img, side, (0,6), side)
        composite.alpha_over(img, otherside, (12,6), otherside)

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
    
    If top is a tuple then first item is the top image and the second
    item is an increment (integer) from 0 to 16 (pixels in the
    original minecraft texture). This increment will be used to crop the
    side images and to paste the top image increment pixels lower, so if
    you use an increment of 8, it willll draw a half-block.
    
    NOTE: this method uses the bottom of the texture image (as done in 
    minecraft with beds and cackes)
    
    """
    
    increment = 0
    if isinstance(top, tuple):
        increment = int(round((top[1] / 16.)*12.)) # range increment in the block height in pixels (half texture size)
        crop_height = increment
        top = top[0]
        if side1 != None:
            side1 = side1.copy()
            ImageDraw.Draw(side1).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
        if side2 != None:
            side2 = side2.copy()
            ImageDraw.Draw(side2).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
        if side3 != None:
            side3 = side3.copy()
            ImageDraw.Draw(side3).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
        if side4 != None:
            side4 = side4.copy()
            ImageDraw.Draw(side4).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), bgcolor)
    
    # first back sides
    if side1 != None :
        side1 = transform_image_side(side1, blockID)
        side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Darken this side.
        sidealpha = side1.split()[3]
        side1 = ImageEnhance.Brightness(side1).enhance(0.9)
        side1.putalpha(sidealpha)        
        
        composite.alpha_over(img, side1, (0,0), side1)

        
    if side2 != None :
        side2 = transform_image_side(side2, blockID)

        # Darken this side.
        sidealpha2 = side2.split()[3]
        side2 = ImageEnhance.Brightness(side2).enhance(0.8)
        side2.putalpha(sidealpha2)

        composite.alpha_over(img, side2, (12,0), side2)

    if bottom != None :
        bottom = transform_image(bottom, blockID)
        composite.alpha_over(img, bottom, (0,12), bottom)
        
    # front sides
    if side3 != None :
        side3 = transform_image_side(side3, blockID)
        
        # Darken this side
        sidealpha = side3.split()[3]
        side3 = ImageEnhance.Brightness(side3).enhance(0.9)
        side3.putalpha(sidealpha)
        
        composite.alpha_over(img, side3, (0,6), side3)
        
    if side4 != None :
        side4 = transform_image_side(side4, blockID)
        side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken this side
        sidealpha = side4.split()[3]
        side4 = ImageEnhance.Brightness(side4).enhance(0.8)
        side4.putalpha(sidealpha)
        
        composite.alpha_over(img, side4, (12,6), side4)

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
               34, -1, 52, 48, -1,160,144, -1,176, 74, -1, -1, -1, -1, 11, -1,
       #       32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
               55, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  9,  4, 
       #       48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
               36, 37, -1, -1, 65, -1, -1, -1, 50, 24, -1, -1, 86, -1, -1, -1,
       #       64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
               -1, -1, -1, -1, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, -1,
       #       80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
               66, 69, 72, 73, 75, -1,102,103,104,105,-1, 102, -1, -1, -1, -1,
       #       96  97  98  99  100  101  102  103 
               -1, -1, -1, -1, -1,   -1,  -1, 137,
        ]

    # NOTE: For non-block textures, the sideid is ignored, but can't be -1

    # And side textures of all block types
       #         0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
    sideids = [ -1,  1,  3,  2, 16,  4, -1, 17,205,205,237,237, 18, 19, 32, 33,
       #        16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
                34, -1, 52, 48, -1,160,144, -1,192, 74, -1, -1,- 1, -1, 11, -1,
       #        32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47
                55, -1, -1, -1, -1, 13, 12, 29, 28, 23, 22, -1, -1,  7,  8, 35,
       #        48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
                36, 37, -1, -1, 65, -1, -1,101, 50, 24, -1, -1, 86, -1, -1, -1,
       #        64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79
                -1, -1, -1, -1, -1, -1, -1, -1, -1, 51, 51, -1, -1, -1, 66, -1,
       #        80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95
                66, 70, 72, 73, 74,-1 ,118,103,104,105, -1, 118,-1, -1, -1, -1,
       #        96  97  98  99  100  101  102  103 
                -1, -1, -1, -1, -1,   -1,  -1, 136,
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

        allimages.append(generate_texture_tuple(img, blockID))

    # Future block types:
    while len(allimages) < 256:
        allimages.append(None)
    return allimages

def load_water():
    """Evidentially, the water and lava textures are not loaded from any files
    in the jar (that I can tell). They must be generated on the fly. While
    terrain.png does have some water and lava cells, not all texture packs
    include them. So I load them here from a couple pngs included.

    This mutates the blockmap global list with the new water and lava blocks.
    Block 9, standing water, is given a block with only the top face showing.
    Block 8, flowing water, is given a full 3 sided cube."""

    try:
        # try the MCPatcher water first, in case it's present
        watertexture = _load_image("custom_water_still.png")
        watertexture = watertexture.crop((0, 0, watertexture.size[0], watertexture.size[0]))
    except IOError:
        watertexture = _load_image("water.png")
    w1 = _build_block(watertexture, None)
    blockmap[9] = generate_texture_tuple(w1,9)
    w2 = _build_block(watertexture, watertexture)
    blockmap[8] = generate_texture_tuple(w2,8)
    
    try:
        # try the MCPatcher lava first, in case it's present
        lavatexture = _load_image("custom_lava_still.png")
        lavatexture = lavatexture.crop((0, 0, lavatexture.size[0], lavatexture.size[0]))
    except IOError:
        lavatexture = _load_image("lava.png")
    lavablock = _build_block(lavatexture, lavatexture)
    blockmap[10] = generate_texture_tuple(lavablock,10)
    blockmap[11] = blockmap[10]

def generate_opaque_mask(img):
    """ Takes the alpha channel of the image and generates a mask
    (used for lighting the block) that deprecates values of alpha
    smallers than 50, and sets every other value to 255. """
    
    alpha = img.split()[3]
    return alpha.point(lambda a: int(min(a, 25.5) * 10))

def generate_texture_tuple(img, blockid):
    """ This takes an image and returns the needed tuple for the
    blockmap list and specialblockmap dictionary."""
    return (img.convert("RGB"), img.split()[3], generate_opaque_mask(img))

def generate_special_texture(blockID, data):
    """Generates a special texture, such as a correctly facing minecraft track"""

    data = convert_data(blockID, data)

    # blocks need to be handled here (and in chunk.py)
    
    if blockID == 2: # grass
        # data & 0x10 means SNOW sides
        side_img = terrain_images[3]
        if data & 0x10:
            side_img = terrain_images[68]
        img = _build_block(terrain_images[0], side_img, 2)
        if not data & 0x10:
            global biome_grass_texture
            composite.alpha_over(img, biome_grass_texture, (0, 0), biome_grass_texture)
        return generate_texture_tuple(img, blockID)


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
        return generate_texture_tuple(img, blockID)


    if blockID == 9 or blockID == 20 or blockID == 79: # spring water, flowing water and waterfall water, AND glass, AND ice
        # water,glass and ice share the way to be rendered
        if blockID == 9:
            try:
                texture = _load_image("custom_water_still.png")
                texture = texture.crop((0, 0, texture.size[0], texture.size[0]))
            except IOError:
                texture = _load_image("water.png")
        elif blockID == 20:
            texture = terrain_images[49]
        else:
            texture = terrain_images[67]
        
        if (data & 0b10000) == 16:
            top = texture
            
        else: top = None

        if (data & 0b0001) == 1:
            side1 = texture    # top left
        else: side1 = None
        
        if (data & 0b1000) == 8:
            side2 = texture    # top right           
        else: side2 = None
        
        if (data & 0b0010) == 2:
            side3 = texture    # bottom left    
        else: side3 = None
        
        if (data & 0b0100) == 4:
            side4 = texture    # bottom right
        else: side4 = None
        
        # if nothing shown do not draw at all
        if top == side3 == side4 == None:
            return None
        
        img = _build_full_block(top,None,None,side3,side4)
        return generate_texture_tuple(img, blockID)


    if blockID == 17: # wood: normal, birch and pines
        top = terrain_images[21]
        if data == 0:
            side = terrain_images[20]
            img = _build_block(top, side, 17)
        if data == 1:
            side = terrain_images[116]
            img = _build_block(top, side, 17)
        if data == 2:
            side = terrain_images[117]
            img = _build_block(top, side, 17)
        
        return generate_texture_tuple(img, blockID)


    if blockID == 18: # leaves
        t = terrain_images[52]
        if data == 1:
            # pine!
            t = terrain_images[132]
        img = _build_block(t, t, 18)
        return generate_texture_tuple(img, blockID)


    if blockID == 26: # bed
        increment = 8
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

        return generate_texture_tuple(img, blockID)


    if blockID == 31: # tall grass
        if data == 0: # dead shrub
            texture = terrain_images[55]
        elif data == 1: # tall grass
            texture = terrain_images[39]
        elif data == 2: # fern
            texture = terrain_images[56]
        
        img = _build_block(texture, texture, blockID)
        return generate_texture_tuple(img,31)


    if blockID in (29,33): # sticky and normal body piston.
        if blockID == 29: # sticky
            piston_t = terrain_images[106].copy()
        else: # normal
            piston_t = terrain_images[107].copy()
        
        # other textures
        side_t = terrain_images[108].copy()
        back_t = terrain_images[109].copy()
        interior_t = terrain_images[110].copy()
        
        if data & 0x08 == 0x08: # pushed out, non full blocks, tricky stuff
            # remove piston texture from piston body
            ImageDraw.Draw(side_t).rectangle((0, 0,16,3),outline=(0,0,0,0),fill=(0,0,0,0))
            
            if data & 0x07 == 0x0: # down
                side_t = side_t.rotate(180)
                img = _build_full_block(back_t ,None ,None ,side_t, side_t)

            elif data & 0x07 == 0x1: # up
                img = _build_full_block((interior_t, 4) ,None ,None ,side_t, side_t)

            elif data & 0x07 == 0x2: # east
                img = _build_full_block(side_t , None, None ,side_t.rotate(90), back_t)

            elif data & 0x07 == 0x3: # west
                img = _build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), None)
                temp = transform_image_side(interior_t, blockID)
                temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, temp, (9,5), temp)

            elif data & 0x07 == 0x4: # north
                img = _build_full_block(side_t.rotate(90) ,None ,None , None, side_t.rotate(270))
                temp = transform_image_side(interior_t, blockID)
                composite.alpha_over(img, temp, (3,5), temp)

            elif data & 0x07 == 0x5: # south
                img = _build_full_block(side_t.rotate(270) ,None , None ,back_t, side_t.rotate(90))

        else: # pushed in, normal full blocks, easy stuff
            if data & 0x07 == 0x0: # down
                side_t = side_t.rotate(180)
                img = _build_full_block(back_t ,None ,None ,side_t, side_t)
            elif data & 0x07 == 0x1: # up
                img = _build_full_block(piston_t ,None ,None ,side_t, side_t)
            elif data & 0x07 == 0x2: # east 
                img = _build_full_block(side_t ,None ,None ,side_t.rotate(90), back_t)
            elif data & 0x07 == 0x3: # west
                img = _build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
            elif data & 0x07 == 0x4: # north
                img = _build_full_block(side_t.rotate(90) ,None ,None ,piston_t, side_t.rotate(270))
            elif data & 0x07 == 0x5: # south
                img = _build_full_block(side_t.rotate(270) ,None ,None ,back_t, side_t.rotate(90))
            

        return generate_texture_tuple(img, blockID)


    if blockID == 34: # piston extension (sticky and normal)
        if (data & 0x8) == 0x8: # sticky
            piston_t = terrain_images[106].copy()
        else: # normal
            piston_t = terrain_images[107].copy()

        # other textures
        side_t = terrain_images[108].copy()
        back_t = terrain_images[107].copy()
        # crop piston body
        ImageDraw.Draw(side_t).rectangle((0, 4,16,16),outline=(0,0,0,0),fill=(0,0,0,0))

        # generate the horizontal piston extension stick
        h_stick = Image.new("RGBA", (24,24), bgcolor)
        temp = transform_image_side(side_t, blockID)
        composite.alpha_over(h_stick, temp, (1,7), temp)
        temp = transform_image(side_t.rotate(90))
        composite.alpha_over(h_stick, temp, (1,1), temp)
        # Darken it
        sidealpha = h_stick.split()[3]
        h_stick = ImageEnhance.Brightness(h_stick).enhance(0.85)
        h_stick.putalpha(sidealpha)

        # generate the vertical piston extension stick
        v_stick = Image.new("RGBA", (24,24), bgcolor)
        temp = transform_image_side(side_t.rotate(90), blockID)
        composite.alpha_over(v_stick, temp, (12,6), temp)
        temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(v_stick, temp, (1,6), temp)
        # Darken it
        sidealpha = v_stick.split()[3]
        v_stick = ImageEnhance.Brightness(v_stick).enhance(0.85)
        v_stick.putalpha(sidealpha)

        # Piston orientation is stored in the 3 first bits
        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = _build_full_block((back_t, 12) ,None ,None ,side_t, side_t)
            composite.alpha_over(img, v_stick, (0,-3), v_stick)
        elif data & 0x07 == 0x1: # up
            img = Image.new("RGBA", (24,24), bgcolor)
            img2 = _build_full_block(piston_t ,None ,None ,side_t, side_t)
            composite.alpha_over(img, v_stick, (0,4), v_stick)
            composite.alpha_over(img, img2, (0,0), img2)
        elif data & 0x07 == 0x2: # east 
            img = _build_full_block(side_t ,None ,None ,side_t.rotate(90), None)
            temp = transform_image_side(back_t, blockID).transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, temp, (2,2), temp)
            composite.alpha_over(img, h_stick, (6,3), h_stick)
        elif data & 0x07 == 0x3: # west
            img = Image.new("RGBA", (24,24), bgcolor)
            img2 = _build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
            composite.alpha_over(img, h_stick, (0,0), h_stick)
            composite.alpha_over(img, img2, (0,0), img2)            
        elif data & 0x07 == 0x4: # north
            img = _build_full_block(side_t.rotate(90) ,None ,None , piston_t, side_t.rotate(270))
            composite.alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (0,0), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
        elif data & 0x07 == 0x5: # south
            img = Image.new("RGBA", (24,24), bgcolor)
            img2 = _build_full_block(side_t.rotate(270) ,None ,None ,None, side_t.rotate(90))
            temp = transform_image_side(back_t, blockID)
            composite.alpha_over(img2, temp, (10,2), temp)
            composite.alpha_over(img, img2, (0,0), img2)
            composite.alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (-3,2), h_stick.transpose(Image.FLIP_LEFT_RIGHT))

        return generate_texture_tuple(img, blockID)


    if blockID == 35: # wool
        if data == 0: # white
            top = side = terrain_images[64]
        elif data == 1: # orange
            top = side = terrain_images[210]
        elif data == 2: # magenta
            top = side = terrain_images[194]
        elif data == 3: # light blue
            top = side = terrain_images[178]
        elif data == 4: # yellow
            top = side = terrain_images[162]
        elif data == 5: # light green
            top = side = terrain_images[146]
        elif data == 6: # pink
            top = side = terrain_images[130]
        elif data == 7: # grey
            top = side = terrain_images[114]
        elif data == 8: # light grey
            top = side = terrain_images[225]
        elif data == 9: # cyan
            top = side = terrain_images[209]
        elif data == 10: # purple
            top = side = terrain_images[193]
        elif data == 11: # blue
            top = side = terrain_images[177]
        elif data == 12: # brown
            top = side = terrain_images[161]
        elif data == 13: # dark green
            top = side = terrain_images[145]
        elif data == 14: # red
            top = side = terrain_images[129]
        elif data == 15: # black
            top = side = terrain_images[113]

        img = _build_block(top, side, 35)
        return generate_texture_tuple(img, blockID)


    if blockID in (43,44): # slab and double-slab
        
        if data == 0: # stone slab
            top = terrain_images[6]
            side = terrain_images[5]
        elif data == 1: # stone slab
            top = terrain_images[176]
            side = terrain_images[192]
        elif data == 2: # wooden slab
            top = side = terrain_images[4]
        elif data == 3: # cobblestone slab
            top = side = terrain_images[16]
        elif data == 4: # brick?
            top = side = terrain_images[7]
        elif data == 5: # stone brick?
            top = side = terrain_images[54]

        img = _build_block(top, side, blockID)
        return generate_texture_tuple(img, blockID)


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
        torch = Image.new("RGBA", (16,16), bgcolor)
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
            img = Image.new("RGBA", (24,24), bgcolor)
            
            small_crop = small.crop((2,2,14,14))
            slice = small_crop.copy()
            ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
            ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
            
            composite.alpha_over(img, slice, (7,5))
            composite.alpha_over(img, small_crop, (6,6))
            composite.alpha_over(img, small_crop, (7,6))
            composite.alpha_over(img, slice, (7,7))

        return generate_texture_tuple(img, blockID)


    if blockID == 51: # fire
        firetexture = _load_image("fire.png")
        side1 = transform_image_side(firetexture)
        side2 = transform_image_side(firetexture).transpose(Image.FLIP_LEFT_RIGHT)
        
        img = Image.new("RGBA", (24,24), bgcolor)

        composite.alpha_over(img, side1, (12,0), side1)
        composite.alpha_over(img, side2, (0,0), side2)

        composite.alpha_over(img, side1, (0,6), side1)
        composite.alpha_over(img, side2, (12,6), side2)
        
        return generate_texture_tuple(img, blockID)


    if blockID in (53,67, 108, 109): # wooden, stone brick, and cobblestone stairs.
        
        if blockID == 53: # wooden
            texture = terrain_images[4]
        elif blockID == 67: # cobblestone
            texture = terrain_images[16]
        elif blockID == 108: # red brick stairs
            texture = terrain_images[7]
        elif blockID == 109: # stone brick stairs
            texture = terrain_images[54]
        
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
            img = Image.new("RGBA", (24,24), bgcolor) # first paste the texture in the back
            tmp1 = transform_image(half_block_r)
            composite.alpha_over(img, tmp1, (0,6))
            tmp2 = _build_full_block(half_block_l, None, None, texture, side)
            composite.alpha_over(img, tmp2)
        
        elif data == 2: # ascending west
            img = Image.new("RGBA", (24,24), bgcolor) # first paste the texture in the back
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
            
        return generate_texture_tuple(img, blockID)

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

        return generate_texture_tuple(img, blockID)


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
            bottom = Image.new("RGBA", (16,16), bgcolor)
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

        return generate_texture_tuple(img, blockID)


    if blockID == 58: # crafting table
        top = terrain_images[43]
        side3 = terrain_images[43+16]
        side4 = terrain_images[43+16+1]
        
        img = _build_full_block(top, None, None, side3, side4, None, 58)
        return generate_texture_tuple(img, blockID)


    if blockID == 59: # crops
        raw_crop = terrain_images[88+data]
        crop1 = transform_image(raw_crop, blockID)
        crop2 = transform_image_side(raw_crop, blockID)
        crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

        img = Image.new("RGBA", (24,24), bgcolor)
        composite.alpha_over(img, crop1, (0,12), crop1)
        composite.alpha_over(img, crop2, (6,3), crop2)
        composite.alpha_over(img, crop3, (6,3), crop3)
        return generate_texture_tuple(img, blockID)


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

        return generate_texture_tuple(img, blockID)


    if blockID == 63: # singposts
        
        texture = terrain_images[4].copy()
        # cut the planks to the size of a signpost
        ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        # If the signpost is looking directly to the image, draw some 
        # random dots, they will look as text.
        if data in (0,1,2,3,4,5,15):
            for i in range(15):
                x = randint(4,11)
                y = randint(3,7)
                texture.putpixel((x,y),(0,0,0,255))

        # Minecraft uses wood texture for the signpost stick
        texture_stick = terrain_images[20]
        texture_stick = texture_stick.resize((12,12), Image.ANTIALIAS)
        ImageDraw.Draw(texture_stick).rectangle((2,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))

        img = Image.new("RGBA", (24,24), bgcolor)

        #         W                N      ~90       E                   S        ~270
        angles = (330.,345.,0.,15.,30.,55.,95.,120.,150.,165.,180.,195.,210.,230.,265.,310.)
        angle = math.radians(angles[data])
        post = transform_image_angle(texture, angle)

        # choose the position of the "3D effect"
        incrementx = 0
        if data in (1,6,7,8,9,14):
            incrementx = -1
        elif data in (3,4,5,11,12,13):
            incrementx = +1

        composite.alpha_over(img, texture_stick,(11, 8),texture_stick)
        # post2 is a brighter signpost pasted with a small sift,
        # gives to the signpost some 3D effect.
        post2 = ImageEnhance.Brightness(post).enhance(1.2)
        composite.alpha_over(img, post2,(incrementx, -3),post2)
        composite.alpha_over(img, post, (0,-2), post)

        return generate_texture_tuple(img, blockID)


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
        img = Image.new("RGBA", (24,24), bgcolor)
        if (data & 0x03) == 0: # northeast corner
            if not swung:
                tex = transform_image_side(raw_door)
                composite.alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (0,0), tex)
        
        if (data & 0x03) == 1: # southeast corner
            if not swung:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (0,0), tex)
            else:
                tex = transform_image_side(raw_door)
                composite.alpha_over(img, tex, (12,0), tex)

        if (data & 0x03) == 2: # southwest corner
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                composite.alpha_over(img, tex, (12,0), tex)
            else:
                tex = transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (12,6), tex)

        if (data & 0x03) == 3: # northwest corner
            if not swung:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT)).transpose(Image.FLIP_LEFT_RIGHT)
                composite.alpha_over(img, tex, (12,6), tex)
            else:
                tex = transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                composite.alpha_over(img, tex, (0,6), tex)
        
        return generate_texture_tuple(img, blockID)


    if blockID == 65: # ladder
        img = Image.new("RGBA", (24,24), bgcolor)
        raw_texture = terrain_images[83]
        #print "ladder is facing: %d" % data
        if data == 5:
            # normally this ladder would be obsured by the block it's attached to
            # but since ladders can apparently be placed on transparent blocks, we 
            # have to render this thing anyway.  same for data == 2
            tex = transform_image_side(raw_texture)
            composite.alpha_over(img, tex, (0,6), tex)
            return generate_texture_tuple(img, blockID)
        if data == 2:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, tex, (12,6), tex)
            return generate_texture_tuple(img, blockID)
        if data == 3:
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, tex, (0,0), tex)
            return generate_texture_tuple(img, blockID)
        if data == 4:
            tex = transform_image_side(raw_texture)
            composite.alpha_over(img, tex, (12,0), tex)
            return generate_texture_tuple(img, blockID)


    if blockID in (27, 28, 66): # minetrack:
        img = Image.new("RGBA", (24,24), bgcolor)
        
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
            composite.alpha_over(img, track, (0,12), track)
        elif data == 6:
            track = transform_image(raw_corner, blockID)
            composite.alpha_over(img, track, (0,12), track)
        elif data == 7:
            track = transform_image(raw_corner.rotate(270), blockID)
            composite.alpha_over(img, track, (0,12), track)
        elif data == 8:
            # flip
            track = transform_image(raw_corner.transpose(Image.FLIP_TOP_BOTTOM).rotate(90), 
                    blockID)
            composite.alpha_over(img, track, (0,12), track)
        elif data == 9:
            track = transform_image(raw_corner.transpose(Image.FLIP_TOP_BOTTOM), 
                    blockID)
            composite.alpha_over(img, track, (0,12), track)
        elif data == 1:
            track = transform_image(raw_straight.rotate(90), blockID)
            composite.alpha_over(img, track, (0,12), track)
            
        #slopes
        elif data == 2: # slope going up in +x direction
            track = transform_image_slope(raw_straight,blockID)
            track = track.transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, track, (2,0), track)
            # the 2 pixels move is needed to fit with the adjacent tracks

        elif data == 3: # slope going up in -x direction
            # tracks are sprites, in this case we are seeing the "side" of 
            # the sprite, so draw a line to make it beautiful.
            ImageDraw.Draw(img).line([(11,11),(23,17)],fill=(164,164,164))
            # grey from track texture (exterior grey).
            # the track doesn't start from image corners, be carefull drawing the line!
        elif data == 4: # slope going up in -y direction
            track = transform_image_slope(raw_straight,blockID)
            composite.alpha_over(img, track, (0,0), track)

        elif data == 5: # slope going up in +y direction
            # same as "data == 3"
            ImageDraw.Draw(img).line([(1,17),(12,11)],fill=(164,164,164))

        return generate_texture_tuple(img, blockID)


    if blockID == 68: # wall sign
        texture = terrain_images[4].copy()
        # cut the planks to the size of a signpost
        ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        # draw some random black dots, they will look as text
        """ don't draw text at the moment, they are used in blank for decoration
        
        if data in (3,4):
            for i in range(15):
                x = randint(4,11)
                y = randint(3,7)
                texture.putpixel((x,y),(0,0,0,255))
        """
        
        img = Image.new("RGBA", (24,24), bgcolor)

        incrementx = 0
        if data == 2:  # east
            incrementx = +1
            sign = _build_full_block(None, None, None, None, texture)
        elif data == 3:  # west
            incrementx = -1
            sign = _build_full_block(None, texture, None, None, None)
        elif data == 4:  # north
            incrementx = +1
            sign = _build_full_block(None, None, texture, None, None)
        elif data == 5:  # south
            incrementx = -1
            sign = _build_full_block(None, None, None, texture, None)

        sign2 = ImageEnhance.Brightness(sign).enhance(1.2)
        composite.alpha_over(img, sign2,(incrementx, 2),sign2)
        composite.alpha_over(img, sign, (0,3), sign)

        return generate_texture_tuple(img, blockID)

    if blockID == 70 or blockID == 72: # wooden and stone pressure plates
        if blockID == 70: # stone
            t = terrain_images[1].copy()
        else: # wooden
            t = terrain_images[4].copy()
        
        # cut out the outside border, pressure plates are smaller
        # than a normal block
        ImageDraw.Draw(t).rectangle((0,0,15,15),outline=(0,0,0,0))
        
        # create the textures and a darker version to make a 3d by 
        # pasting them with an offstet of 1 pixel
        img = Image.new("RGBA", (24,24), bgcolor)
        
        top = transform_image(t, blockID)
        
        alpha = top.split()[3]
        topd = ImageEnhance.Brightness(top).enhance(0.8)
        topd.putalpha(alpha)
        
        #show it 3d or 2d if unpressed or pressed
        if data == 0:
            composite.alpha_over(img,topd, (0,12),topd)
            composite.alpha_over(img,top, (0,11),top)
        elif data == 1:
            composite.alpha_over(img,top, (0,12),top)
        
        return generate_texture_tuple(img, blockID)

    if blockID == 85: # fences
        # create needed images for Big stick fence

        fence_top = terrain_images[4].copy()
        fence_side = terrain_images[4].copy()
        
        # generate the textures of the fence
        ImageDraw.Draw(fence_top).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_top).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_top).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_top).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        ImageDraw.Draw(fence_side).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_side).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        # Create the sides and the top of the big stick
        fence_side = transform_image_side(fence_side,85)
        fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
        fence_top = transform_image(fence_top,85)

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
        fence_big = Image.new("RGBA", (24,24), bgcolor)
        composite.alpha_over(fence_big,fence_side, (5,4),fence_side)
        composite.alpha_over(fence_big,fence_other_side, (7,4),fence_other_side)
        composite.alpha_over(fence_big,fence_top, (0,0),fence_top)
        
        # Now render the small sticks.
        # Create needed images
        fence_small_side = terrain_images[4].copy()
        
        # Generate mask
        ImageDraw.Draw(fence_small_side).rectangle((0,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_small_side).rectangle((0,4,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_small_side).rectangle((0,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_small_side).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(fence_small_side).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

        # Create the sides and the top of the small sticks
        fence_small_side = transform_image_side(fence_small_side,85)
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
        img = Image.new("RGBA", (24,24), bgcolor)

        # Position of fence small sticks in img.
        # These postitions are strange because the small sticks of the 
        # fence are at the very left and at the very right of the 16x16 images
        pos_top_left = (2,3)
        pos_top_right = (10,3)
        pos_bottom_right = (10,7)
        pos_bottom_left = (2,7)
        
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
            
        return generate_texture_tuple(img, blockID)


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

        return generate_texture_tuple(img, blockID)


    if blockID == 90: # portal
        portaltexture = _load_image("portal.png")
        img = Image.new("RGBA", (24,24), bgcolor)

        side = transform_image_side(portaltexture)
        otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

        if data in (1,4):
            composite.alpha_over(img, side, (5,4), side)

        if data in (2,8):
            composite.alpha_over(img, otherside, (5,4), otherside)

        return generate_texture_tuple(img, blockID)


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
        
        img = Image.new("RGBA", (24,24), bgcolor)
        
        composite.alpha_over(img, side, (1,6), side)
        composite.alpha_over(img, otherside, (11,7), otherside) # workaround, fixes a hole
        composite.alpha_over(img, otherside, (12,6), otherside)
        composite.alpha_over(img, top, (0,6), top)

        return generate_texture_tuple(img, blockID)


    if blockID in (93, 94): # redstone repeaters (diodes), ON and OFF
        # generate the diode
        top = terrain_images[131] if blockID == 93 else terrain_images[147]
        side = terrain_images[5]
        increment = 13
        
        if (data & 0x3) == 0: # pointing east
            pass
        
        if (data & 0x3) == 1: # pointing south
            top = top.rotate(270)

        if (data & 0x3) == 2: # pointing west
            top = top.rotate(180)

        if (data & 0x3) == 3: # pointing north
            top = top.rotate(90)

        img = _build_full_block( (top, increment), None, None, side, side)

        # compose a "3d" redstone torch
        t = terrain_images[115].copy() if blockID == 93 else terrain_images[99].copy()
        torch = Image.new("RGBA", (24,24), bgcolor)
        
        t_crop = t.crop((2,2,14,14))
        slice = t_crop.copy()
        ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
        
        composite.alpha_over(torch, slice, (6,4))
        composite.alpha_over(torch, t_crop, (5,5))
        composite.alpha_over(torch, t_crop, (6,5))
        composite.alpha_over(torch, slice, (6,6))
        
        # paste redstone torches everywhere!
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
                moving_torch = (0,2)
                static_torch = (5,-1)
                
            elif (data & 0xC) == 8: # three ticks delay
                moving_torch = (-1,2)
                static_torch = (5,-1)
                
            elif (data & 0xC) == 12: # four ticks delay
                moving_torch = (-2,3)
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

        return generate_texture_tuple(img, blockID)
        
        
    if blockID == 96: # trapdoor
        texture = terrain_images[84]
        if data & 0x4 == 0x4: # opened trapdoor
            if data & 0x3 == 0: # west
                img = _build_full_block(None, None, None, None, texture)
            if data & 0x3 == 1: # east
                img = _build_full_block(None, texture, None, None, None)
            if data & 0x3 == 2: # south
                img = _build_full_block(None, None, texture, None, None)
            if data & 0x3 == 3: # north
                img = _build_full_block(None, None, None, texture, None)
            
        elif data & 0x4 == 0: # closed trapdoor
            img = _build_full_block((texture, 12), None, None, texture, texture)
        
        return generate_texture_tuple(img, blockID)

    if blockID == 98: # normal, mossy and cracked stone brick
        if data == 0: # normal
            t = terrain_images[54]
        elif data == 1: # mossy
            t = terrain_images[100]
        else: # cracked
            t = terrain_images[101]

        img = _build_full_block(t, None, None, t, t)

        return generate_texture_tuple(img, blockID)

    if blockID == 99 or blockID == 100: # huge brown and red mushroom
        if blockID == 99: # brown
            cap = terrain_images[126]
        else: # red
            cap = terrain_images[125]
        stem = terrain_images[141]
        porous = terrain_images[142]
        
        if data == 0: # fleshy piece
            img = _build_full_block(porous, None, None, porous, porous)

        if data == 1: # north-east corner
            img = _build_full_block(cap, None, None, cap, porous)

        if data == 2: # east side
            img = _build_full_block(cap, None, None, porous, porous)

        if data == 3: # south-east corner
            img = _build_full_block(cap, None, None, porous, cap)

        if data == 4: # north side
            img = _build_full_block(cap, None, None, cap, porous)

        if data == 5: # top piece
            img = _build_full_block(cap, None, None, porous, porous)

        if data == 6: # south side
            img = _build_full_block(cap, None, None, cap, porous)

        if data == 7: # north-west corner
            img = _build_full_block(cap, None, None, cap, cap)

        if data == 8: # west side
            img = _build_full_block(cap, None, None, porous, cap)

        if data == 9: # south-west corner
            img = _build_full_block(cap, None, None, porous, cap)

        if data == 10: # stem
            img = _build_full_block(porous, None, None, stem, stem)

        return generate_texture_tuple(img, blockID)

    if blockID == 101 or blockID == 102: # iron bars and glass panes
        if blockID == 101:
            # iron bars
            t = terrain_images[85]
        else:
            # glass panes
            t = terrain_images[49]
        left = t.copy()
        right = t.copy()

        # generate the four small pieces of the glass pane
        ImageDraw.Draw(right).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(left).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        
        up_left = transform_image_side(left)
        up_right = transform_image_side(right).transpose(Image.FLIP_TOP_BOTTOM)
        dw_right = transform_image_side(right)
        dw_left = transform_image_side(left).transpose(Image.FLIP_TOP_BOTTOM)

        # Create img to compose the texture
        img = Image.new("RGBA", (24,24), bgcolor)

        # +x axis points top right direction
        # +y axis points bottom right direction
        # First compose things in the back of the image, 
        # then things in the front.

        if (data & 0b0001) == 1 or data == 0:
            composite.alpha_over(img,up_left, (6,3),up_left)    # top left
        if (data & 0b1000) == 8 or data == 0:
            composite.alpha_over(img,up_right, (6,3),up_right)  # top right
        if (data & 0b0010) == 2 or data == 0:
            composite.alpha_over(img,dw_left, (6,3),dw_left)    # bottom left    
        if (data & 0b0100) == 4 or data == 0:
            composite.alpha_over(img,dw_right, (6,3),dw_right)  # bottom right

        return generate_texture_tuple(img, blockID)

    if blockID == 104 or blockID == 105: # pumpkin and melon stems.
        # the ancildata value indicates how much of the texture
        # is shown.
        if data & 48 == 0:
            # not fully grown stem or no pumpkin/melon touching it,
            # straight up stem
            t = terrain_images[111].copy()
            img = Image.new("RGBA", (16,16), bgcolor)
            composite.alpha_over(img, t, (0, int(16 - 16*((data + 1)/8.))), t)
            img = _build_block(img, img, blockID)
            if data & 7 == 7:
                # fully grown stem gets brown color!
                # there is a conditional in rendermode-normal to not
                # tint the data value 7
                img = tintTexture(img, (211,169,116))
            return generate_texture_tuple(img, blockID)
        
        else: # fully grown, and a pumpking/melon touching it,
              # corner stem
            pass
            
            
    if blockID == 106: # vine
        img = Image.new("RGBA", (24,24), bgcolor)
        raw_texture = terrain_images[143]
        # print "vine is facing: %d" % data
        if data == 2:   # south
            tex = transform_image_side(raw_texture)
            composite.alpha_over(img, tex, (0,6), tex)
            return generate_texture_tuple(img, blockID)
        if data == 1:	# east
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, tex, (12,6), tex)
            return generate_texture_tuple(img, blockID)
        if data == 4:	# west
            tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, tex, (0,0), tex)
            return generate_texture_tuple(img, blockID)
        if data == 8:	# north
            tex = transform_image_side(raw_texture)
            composite.alpha_over(img, tex, (12,0), tex)
            return generate_texture_tuple(img, blockID)
    
    if blockID == 107:
        # create the closed gate side
        gate_side = terrain_images[4].copy()
        gate_side_draw = ImageDraw.Draw(gate_side)
        gate_side_draw.rectangle((7,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
        gate_side_draw.rectangle((7,4,9,6),outline=(0,0,0,0),fill=(0,0,0,0))
        gate_side_draw.rectangle((7,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
        gate_side_draw.rectangle((0,12,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
        gate_side_draw.rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
        gate_side_draw.rectangle((14,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        
        # darken the sides slightly, as with the fences
        sidealpha = gate_side.split()[3]
        gate_side = ImageEnhance.Brightness(gate_side).enhance(0.9)
        gate_side.putalpha(sidealpha)
        
        # create the other sides
        mirror_gate_side = transform_image_side(gate_side.transpose(Image.FLIP_LEFT_RIGHT), blockID)
        gate_side = transform_image_side(gate_side, blockID)
        gate_other_side = gate_side.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_gate_other_side = mirror_gate_side.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Create img to compose the fence gate
        img = Image.new("RGBA", (24,24), bgcolor)
        
        if data & 0x4:
            # opened
            data = data & 0x3
            if data == 0:
                composite.alpha_over(img, gate_side, (2,8), gate_side)
                composite.alpha_over(img, gate_side, (13,3), gate_side)
            elif data == 1:
                composite.alpha_over(img, gate_other_side, (-1,3), gate_other_side)
                composite.alpha_over(img, gate_other_side, (10,8), gate_other_side)
            elif data == 2:
                composite.alpha_over(img, mirror_gate_side, (-1,7), mirror_gate_side)
                composite.alpha_over(img, mirror_gate_side, (10,2), mirror_gate_side)
            elif data == 3:
                composite.alpha_over(img, mirror_gate_other_side, (2,1), mirror_gate_other_side)
                composite.alpha_over(img, mirror_gate_other_side, (13,7), mirror_gate_other_side)
        else:
            # closed
            
            # positions for pasting the fence sides, as with fences
            pos_top_left = (2,3)
            pos_top_right = (10,3)
            pos_bottom_right = (10,7)
            pos_bottom_left = (2,7)
            
            if data == 0 or data == 2:
                composite.alpha_over(img, gate_other_side, pos_top_right, gate_other_side)
                composite.alpha_over(img, mirror_gate_other_side, pos_bottom_left, mirror_gate_other_side)
            elif data == 1 or data == 3:
                composite.alpha_over(img, gate_side, pos_top_left, gate_side)
                composite.alpha_over(img, mirror_gate_side, pos_bottom_right, mirror_gate_side)
        
        return generate_texture_tuple(img, blockID)

    return None

def convert_data(blockID, data):
    if blockID == 26: # bed
        #Masked to not clobber block head/foot info
        if _north == 'upper-left':
            if (data & 0b0011) == 0: data = data & 0b1100 | 1
            elif (data & 0b0011) == 1: data = data & 0b1100 | 2
            elif (data & 0b0011) == 2: data = data & 0b1100 | 3
            elif (data & 0b0011) == 3: data = data & 0b1100 | 0
        elif _north == 'upper-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 2
            elif (data & 0b0011) == 1: data = data & 0b1100 | 3
            elif (data & 0b0011) == 2: data = data & 0b1100 | 0
            elif (data & 0b0011) == 3: data = data & 0b1100 | 1
        elif _north == 'lower-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 3
            elif (data & 0b0011) == 1: data = data & 0b1100 | 0
            elif (data & 0b0011) == 2: data = data & 0b1100 | 1
            elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    if blockID in (29, 33, 34): # sticky piston, piston, piston extension
        #Masked to not clobber block head/foot info
        if _north == 'upper-left':
            if (data & 0b0111) == 2: data = data & 0b1000 | 5
            elif (data & 0b0111) == 3: data = data & 0b1000 | 4
            elif (data & 0b0111) == 4: data = data & 0b1000 | 2
            elif (data & 0b0111) == 5: data = data & 0b1000 | 3
        elif _north == 'upper-right':
            if (data & 0b0111) == 2: data = data & 0b1000 | 3
            elif (data & 0b0111) == 3: data = data & 0b1000 | 2
            elif (data & 0b0111) == 4: data = data & 0b1000 | 5
            elif (data & 0b0111) == 5: data = data & 0b1000 | 4
        elif _north == 'lower-right':
            if (data & 0b0111) == 2: data = data & 0b1000 | 4
            elif (data & 0b0111) == 3: data = data & 0b1000 | 5
            elif (data & 0b0111) == 4: data = data & 0b1000 | 3
            elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    if blockID in (27, 28, 66): # minetrack:
        #Masked to not clobber powered rail on/off info
        #Ascending and flat straight
        if _north == 'upper-left':
            if (data & 0b0111) == 0: data = data & 0b1000 | 1
            elif (data & 0b0111) == 1: data = data & 0b1000 | 0
            elif (data & 0b0111) == 2: data = data & 0b1000 | 5
            elif (data & 0b0111) == 3: data = data & 0b1000 | 4
            elif (data & 0b0111) == 4: data = data & 0b1000 | 2
            elif (data & 0b0111) == 5: data = data & 0b1000 | 3
        elif _north == 'upper-right':
            if (data & 0b0111) == 2: data = data & 0b1000 | 3
            elif (data & 0b0111) == 3: data = data & 0b1000 | 2
            elif (data & 0b0111) == 4: data = data & 0b1000 | 5
            elif (data & 0b0111) == 5: data = data & 0b1000 | 4
        elif _north == 'lower-right':
            if (data & 0b0111) == 0: data = data & 0b1000 | 1
            elif (data & 0b0111) == 1: data = data & 0b1000 | 0
            elif (data & 0b0111) == 2: data = data & 0b1000 | 4
            elif (data & 0b0111) == 3: data = data & 0b1000 | 5
            elif (data & 0b0111) == 4: data = data & 0b1000 | 3
            elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    if blockID == 66: # normal minetrack only
        #Corners
        if _north == 'upper-left':
            if data == 6: data = 7
            elif data == 7: data = 8
            elif data == 8: data = 6
            elif data == 9: data = 9
        elif _north == 'upper-right':
            if data == 6: data = 8
            elif data == 7: data = 9
            elif data == 8: data = 6
            elif data == 9: data = 7
        elif _north == 'lower-right':
            if data == 6: data = 9
            elif data == 7: data = 6
            elif data == 8: data = 8
            elif data == 9: data = 7
    if blockID in (50, 75, 76): # torch, off/on redstone torch
        if _north == 'upper-left':
            if data == 1: data = 3
            elif data == 2: data = 4
            elif data == 3: data = 2
            elif data == 4: data = 1
        elif _north == 'upper-right':
            if data == 1: data = 2
            elif data == 2: data = 1
            elif data == 3: data = 4
            elif data == 4: data = 3
        elif _north == 'lower-right':
            if data == 1: data = 4
            elif data == 2: data = 3
            elif data == 3: data = 1
            elif data == 4: data = 2
    if blockID in (53,67,108,109): # wooden and cobblestone stairs.
        if _north == 'upper-left':
            if data == 0: data = 2
            elif data == 1: data = 3
            elif data == 2: data = 1
            elif data == 3: data = 0
        elif _north == 'upper-right':
            if data == 0: data = 1
            elif data == 1: data = 0
            elif data == 2: data = 3
            elif data == 3: data = 2
        elif _north == 'lower-right':
            if data == 0: data = 3
            elif data == 1: data = 2
            elif data == 2: data = 0
            elif data == 3: data = 1
    if blockID in (61, 62, 23): # furnace and burning furnace
        if _north == 'upper-left':
            if data == 2: data = 5
            elif data == 3: data = 4
            elif data == 4: data = 2
            elif data == 5: data = 3
        elif _north == 'upper-right':
            if data == 2: data = 3
            elif data == 3: data = 2
            elif data == 4: data = 5
            elif data == 5: data = 4
        elif _north == 'lower-right':
            if data == 2: data = 4
            elif data == 3: data = 5
            elif data == 4: data = 3
            elif data == 5: data = 2
    if blockID == 63: # signposts
        if _north == 'upper-left':
            data = (data + 4) % 16
        elif _north == 'upper-right':
            data = (data + 8) % 16
        elif _north == 'lower-right':
            data = (data + 12) % 16
    if blockID in (64,71): # wooden/iron door
        #Masked to not clobber block top/bottom & swung info
        if _north == 'upper-left':
            if (data & 0b0011) == 0: data = data & 0b1100 | 1
            elif (data & 0b0011) == 1: data = data & 0b1100 | 2
            elif (data & 0b0011) == 2: data = data & 0b1100 | 3
            elif (data & 0b0011) == 3: data = data & 0b1100 | 0
        elif _north == 'upper-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 2
            elif (data & 0b0011) == 1: data = data & 0b1100 | 3
            elif (data & 0b0011) == 2: data = data & 0b1100 | 0
            elif (data & 0b0011) == 3: data = data & 0b1100 | 1
        elif _north == 'lower-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 3
            elif (data & 0b0011) == 1: data = data & 0b1100 | 0
            elif (data & 0b0011) == 2: data = data & 0b1100 | 1
            elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    if blockID == 65: # ladder
        if _north == 'upper-left':
            if data == 2: data = 5
            elif data == 3: data = 4
            elif data == 4: data = 2
            elif data == 5: data = 3
        elif _north == 'upper-right':
            if data == 2: data = 3
            elif data == 3: data = 2
            elif data == 4: data = 5
            elif data == 5: data = 4
        elif _north == 'lower-right':
            if data == 2: data = 4
            elif data == 3: data = 5
            elif data == 4: data = 3
            elif data == 5: data = 2
    if blockID == 68: # wall sign
        if _north == 'upper-left':
            if data == 2: data = 5
            elif data == 3: data = 4
            elif data == 4: data = 2
            elif data == 5: data = 3
        elif _north == 'upper-right':
            if data == 2: data = 3
            elif data == 3: data = 2
            elif data == 4: data = 5
            elif data == 5: data = 4
        elif _north == 'lower-right':
            if data == 2: data = 4
            elif data == 3: data = 5
            elif data == 4: data = 3
            elif data == 5: data = 2
    if blockID in (86,91): # pumpkins, jack-o-lantern
        if _north == 'upper-left':
            if data == 0: data = 1
            elif data == 1: data = 2
            elif data == 2: data = 3
            elif data == 3: data = 0
        elif _north == 'upper-right':
            if data == 0: data = 2
            elif data == 1: data = 3
            elif data == 2: data = 0
            elif data == 3: data = 1
        elif _north == 'lower-right':
            if data == 0: data = 3
            elif data == 1: data = 0
            elif data == 2: data = 1
            elif data == 3: data = 2
    if blockID in (93, 94): # redstone repeaters, ON and OFF
        #Masked to not clobber delay info
        if _north == 'upper-left':
            if (data & 0b0011) == 0: data = data & 0b1100 | 1
            elif (data & 0b0011) == 1: data = data & 0b1100 | 2
            elif (data & 0b0011) == 2: data = data & 0b1100 | 3
            elif (data & 0b0011) == 3: data = data & 0b1100 | 0
        elif _north == 'upper-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 2
            elif (data & 0b0011) == 1: data = data & 0b1100 | 3
            elif (data & 0b0011) == 2: data = data & 0b1100 | 0
            elif (data & 0b0011) == 3: data = data & 0b1100 | 1
        elif _north == 'lower-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 3
            elif (data & 0b0011) == 1: data = data & 0b1100 | 0
            elif (data & 0b0011) == 2: data = data & 0b1100 | 1
            elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    if blockID == 96: # trapdoor
        #Masked to not clobber opened/closed info
        if _north == 'upper-left':
            if (data & 0b0011) == 0: data = data & 0b1100 | 3
            elif (data & 0b0011) == 1: data = data & 0b1100 | 2
            elif (data & 0b0011) == 2: data = data & 0b1100 | 0
            elif (data & 0b0011) == 3: data = data & 0b1100 | 1
        elif _north == 'upper-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 1
            elif (data & 0b0011) == 1: data = data & 0b1100 | 0
            elif (data & 0b0011) == 2: data = data & 0b1100 | 3
            elif (data & 0b0011) == 3: data = data & 0b1100 | 2
        elif _north == 'lower-right':
            if (data & 0b0011) == 0: data = data & 0b1100 | 2
            elif (data & 0b0011) == 1: data = data & 0b1100 | 3
            elif (data & 0b0011) == 2: data = data & 0b1100 | 1
            elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    if blockID == 99 or blockID == 100: # huge red and brown mushroom
        if _north == 'upper-left':
            if data == 1: data = 3
            elif data == 2: data = 6
            elif data == 3: data = 9
            elif data == 4: data = 2
            elif data == 6: data = 8
            elif data == 7: data = 1
            elif data == 8: data = 4
            elif data == 9: data = 7
        elif _north == 'upper-right':
            if data == 1: data = 9
            elif data == 2: data = 8
            elif data == 3: data = 7
            elif data == 4: data = 6
            elif data == 6: data = 4
            elif data == 7: data = 3
            elif data == 8: data = 2
            elif data == 9: data = 1
        elif _north == 'lower-right':
            if data == 1: data = 7
            elif data == 2: data = 4
            elif data == 3: data = 1
            elif data == 4: data = 2
            elif data == 6: data = 8
            elif data == 7: data = 9
            elif data == 8: data = 6
            elif data == 9: data = 3
    if blockID == 106: # vine
        if _north == 'upper-left':
            if data == 1: data = 2
            elif data == 4: data = 8
            elif data == 8: data = 1
            elif data == 2: data = 4
        elif _north == 'upper-right':
            if data == 1: data = 4
            elif data == 4: data = 1
            elif data == 8: data = 2
            elif data == 2: data = 8
        elif _north == 'lower-right':
            if data == 1: data = 8
            elif data == 4: data = 2
            elif data == 8: data = 4
            elif data == 2: data = 1
    if blockID == 107: # fence gates
        opened = False
        if data & 0x4:
            data = data & 0x3
            opened = True
        if _north == 'upper-left':
            if data == 0: data = 1
            elif data == 1: data = 2
            elif data == 2: data = 3
            elif data == 3: data = 0
        elif _north == 'upper-right':
            if data == 0: data = 2
            elif data == 1: data = 3
            elif data == 2: data = 0
            elif data == 3: data = 1
        elif _north == 'lower-right':
            if data == 0: data = 3
            elif data == 1: data = 0
            elif data == 2: data = 1
            elif data == 3: data = 2
        if opened:
            data = data | 0x4
            
    return data

def tintTexture(im, c):
    # apparently converting to grayscale drops the alpha channel?
    i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
    i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
    return i

currentBiomeFile = None
currentBiomeData = None
grasscolor = None
foliagecolor = None
watercolor = None

def prepareBiomeData(worlddir):
    global grasscolor, foliagecolor, watercolor
    
    # skip if the color files are already loaded
    if grasscolor and foliagecolor:
        return
    
    biomeDir = os.path.join(worlddir, "biomes")
    if not os.path.exists(biomeDir):
        raise Exception("biomes not found")

    # try to find the biome color images.  If _find_file can't locate them
    # then try looking in the EXTRACTEDBIOMES folder
    try:
        grasscolor = list(_load_image("grasscolor.png").getdata())
        foliagecolor = list(_load_image("foliagecolor.png").getdata())
        # don't force the water color just yet
        # since the biome extractor doesn't know about it
        try:
            watercolor = list(_load_image("watercolor.png").getdata())
        except IOError:
            pass
    except IOError:
        try:
            grasscolor = list(Image.open(os.path.join(biomeDir,"grasscolor.png")).getdata())
            foliagecolor = list(Image.open(os.path.join(biomeDir,"foliagecolor.png")).getdata())
        except Exception:
            # clear anything that managed to get set
            grasscolor = None
            foliagecolor = None
            watercolor = None

def getBiomeData(worlddir, chunkX, chunkY):
    '''Opens the worlddir and reads in the biome color information
    from the .biome files.  See also:
    http://www.minecraftforum.net/viewtopic.php?f=25&t=80902
    '''

    global currentBiomeFile, currentBiomeData
    biomeX = chunkX // 32
    biomeY = chunkY // 32
    rots = 0
    if _north == 'upper-left':
        temp = biomeX
        biomeX = biomeY
        biomeY = -temp-1
        rots = 3
    elif _north == 'upper-right':
        biomeX = -biomeX-1
        biomeY = -biomeY-1
        rots = 2
    elif _north == 'lower-right':
        temp = biomeX
        biomeX = -biomeY-1
        biomeY = temp
        rots = 1

    biomeFile = "b.%d.%d.biome" % (biomeX, biomeY)
    if biomeFile == currentBiomeFile:
        return currentBiomeData

    try:
        with open(os.path.join(worlddir, "biomes", biomeFile), "rb") as f:
            rawdata = f.read()
            # make sure the file size is correct
            if not len(rawdata) == 512 * 512 * 2:
                raise Exception("Biome file %s is not valid." % (biomeFile,))
            data = numpy.reshape(numpy.rot90(numpy.reshape(
                    numpy.frombuffer(rawdata, dtype=numpy.dtype(">u2")),
                    (512,512)),rots), -1)
    except IOError:
        data = None
        pass # no biome data   

    currentBiomeFile = biomeFile
    currentBiomeData = data
    return data

lightcolor = None
lightcolor_checked = False
def loadLightColor():
    global lightcolor, lightcolor_checked
    
    if not lightcolor_checked:
        lightcolor_checked = True
        try:
            lightcolor = list(_load_image("light_normal.png").getdata())
        except Exception:
            logging.warning("Light color image could not be found.")
            lightcolor = None
    return lightcolor

# This set holds block ids that require special pre-computing.  These are typically
# things that require ancillary data to render properly (i.e. ladder plus orientation)
# A good source of information is:
#  http://www.minecraftwiki.net/wiki/Data_values
# (when adding new blocks here and in generate_special_textures,
# please, if possible, keep the ascending order of blockid value)

special_blocks = set([ 2,  6,  9, 17, 18, 20, 26, 23, 27, 28, 29, 31, 33,
                      34, 35, 43, 44, 50, 51, 53, 54, 55, 58, 59, 61, 62,
                      63, 64, 65, 66, 67, 68, 70, 71, 72, 75, 76, 79, 85,
                      86, 90, 91, 92, 93, 94, 96, 98, 99, 100, 101, 102,
                      104, 105, 106, 107, 108, 109])

# this is a map of special blockIDs to a list of all 
# possible values for ancillary data that it might have.

special_map = {}

# 0x10 means SNOW sides
special_map[2] = range(11) + [0x10,]  # grass, grass has not ancildata but is
                                      # used in the mod WildGrass, and this
                                      # small fix shows the map as expected,
                                      # and is harmless for normal maps
special_map[6] = range(16)  # saplings: usual, spruce, birch and future ones (rendered as usual saplings)
special_map[9] = range(32)  # water: spring,flowing, waterfall, and others (unknown) ancildata values, uses pseudo data
special_map[17] = range(3)  # wood: normal, birch and pine
special_map[18] = range(16) # leaves, birch, normal or pine leaves
special_map[20] = range(32) # glass, used to only render the exterior surface, uses pseudo data
special_map[26] = range(12) # bed, orientation
special_map[23] = range(6)  # dispensers, orientation
special_map[27] = range(14) # powered rail, orientation/slope and powered/unpowered
special_map[28] = range(6) # detector rail, orientation/slope
special_map[29] = (0,1,2,3,4,5,8,9,10,11,12,13) # sticky piston body, orientation, pushed in/out
special_map[31] = range(3) # tall grass, dead shrub, fern and tall grass itself
special_map[33] = (0,1,2,3,4,5,8,9,10,11,12,13) # normal piston body, orientation, pushed in/out
special_map[34] = (0,1,2,3,4,5,8,9,10,11,12,13) # normal and sticky piston extension, orientation, sticky/normal
special_map[35] = range(16) # wool, colored and white
special_map[43] = range(6)  # stone, sandstone, wooden and cobblestone double-slab
special_map[44] = range(6)  # stone, sandstone, wooden and cobblestone slab
special_map[50] = (1,2,3,4,5) # torch, position in the block
special_map[51] = range(16) # fire, position in the block (not implemented)
special_map[53] = range(4)  # wooden stairs, orientation
special_map[54] = range(12) # chests, orientation and type (single or double), uses pseudo data
special_map[55] = range(128) # redstone wire, all the possible combinations, uses pseudo data
special_map[58] = (0,)      # crafting table, it has 2 different sides
special_map[59] = range(8)  # crops, grow from 0 to 7
special_map[61] = range(6)  # furnace, orientation
special_map[62] = range(6)  # burning furnace, orientation
special_map[63] = range(16) # signpost, orientation
special_map[64] = range(16) # wooden door, open/close and orientation
special_map[65] = (2,3,4,5) # ladder, orientation
special_map[66] = range(10) # minecrart tracks, orientation, slope
special_map[67] = range(4)  # cobblestone stairs, orientation
special_map[68] = (2,3,4,5) # wall sing, orientation
special_map[70] = (0,1)     # stone pressure plate, non pressed and pressed
special_map[71] = range(16) # iron door, open/close and orientation
special_map[72] = (0,1)     # wooden pressure plate, non pressed and pressed
special_map[75] = (1,2,3,4,5) # off redstone torch, orientation
special_map[76] = (1,2,3,4,5) # on redstone torch, orientation
special_map[79] = range(32) # ice, used to only render the exterior surface, uses pseudo data
special_map[85] = range(17) # fences, all the possible combination, uses pseudo data
special_map[86] = range(5)  # pumpkin, orientation
special_map[90] = (1,2,4,8) # portal, in 2 orientations, 4 cases, uses pseudo data
special_map[91] = range(5)  # jack-o-lantern, orientation
special_map[92] = range(6) # cake, eaten amount, (not implemented)
special_map[93] = range(16) # OFF redstone repeater, orientation and delay
special_map[94] = range(16) # ON redstone repeater, orientation and delay
special_map[96] = range(8)  # trapdoor, open, closed, orientation
special_map[98] = range(3)  # stone brick, normal, mossy and cracked
special_map[99] = range(11) # huge brown mushroom, side, corner, etc, piece
special_map[100] = range(11) # huge red mushroom, side, corner, etc, piece
special_map[101]= range(16)  # iron bars, all the possible combination, uses pseudo data
special_map[102]= range(16)  # glass panes, all the possible combination, uses pseudo data
special_map[104] = range(8) # pumpkin stem, size of the stem
special_map[105] = range(8) # melon stem, size of the stem
special_map[106] = (1,2,4,8) # vine, orientation
special_map[107] = range(8) # fence gates, orientation + open bit
special_map[108]= range(4)  # red stairs, orientation
special_map[109]= range(4)  # stonebrick stairs, orientation

# placeholders that are generated in generate()
bgcolor = None
terrain_images = None
blockmap = None
biome_grass_texture = None
specialblockmap = None

def generate(path=None,texture_size=24,bgc = (26,26,26,0),north_direction='lower-left'):
    global _north
    _north = north_direction
    global _find_file_local_path
    global bgcolor
    bgcolor = bgc
    global _find_file_local_path, texture_dimensions
    _find_file_local_path = path
    texture_dimensions = (texture_size, texture_size)
    
    # This maps terainids to 16x16 images
    global terrain_images
    terrain_images = _split_terrain(_get_terrain_image())
    
    # generate the normal blocks
    global blockmap
    blockmap = _build_blockimages()
    load_water()
    
    # generate biome grass mask
    global biome_grass_texture
    biome_grass_texture = _build_block(terrain_images[0], terrain_images[38], 2)
    
    # generate the special blocks
    global specialblockmap, special_blocks
    specialblockmap = {}
    for blockID in special_blocks:
        for data in special_map[blockID]:
            specialblockmap[(blockID, data)] = generate_special_texture(blockID, data)

    if texture_size != 24:
        # rescale biome textures.
        biome_grass_texture = biome_grass_texture.resize(texture_dimensions, Image.ANTIALIAS)

        # rescale the normal block images
        for i in range(len(blockmap)):
            if blockmap[i] != None:
                block = blockmap[i]
                alpha = block[1]
                block = block[0]
                block.putalpha(alpha)
                scaled_block = block.resize(texture_dimensions, Image.ANTIALIAS)
                blockmap[i] = generate_texture_tuple(scaled_block, i)

        # rescale the special block images
        for blockid, data in iter(specialblockmap):
            block = specialblockmap[(blockid,data)]
            if block != None:
                alpha = block[1]
                block = block[0]
                block.putalpha(alpha)
                scaled_block = block.resize(texture_dimensions, Image.ANTIALIAS)
                specialblockmap[(blockid,data)] = generate_texture_tuple(scaled_block, blockid)
