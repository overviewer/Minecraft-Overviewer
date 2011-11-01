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
import functools

import util
import composite

##
## useful global variables
##

# user-provided path given by --texture-path
_find_file_local_path = None
# image background color to use
bgcolor = None
# an array of the textures in terrain.png, split up
terrain_images = None

##
## Helpers for opening textures
##

def _find_file(filename, mode="rb", verbose=False):
    """Searches for the given file and returns an open handle to it.
    This searches the following locations in this order:
    
    * the textures_path given in the config file (if present)
    * The program dir (same dir as overviewer.py)
    * The overviewer_core/data/textures dir
    * On Darwin, in /Applications/Minecraft
    * Inside minecraft.jar, which is looked for at these locations

      * On Windows, at %APPDATA%/.minecraft/bin/minecraft.jar
      * On Darwin, at $HOME/Library/Application Support/minecraft/bin/minecraft.jar
      * at $HOME/.minecraft/bin/minecraft.jar

    """
    
    if _find_file_local_path:
        path = os.path.join(_find_file_local_path, filename)
        if os.path.exists(path):
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)
    
    programdir = util.get_program_path()
    path = os.path.join(programdir, filename)
    if os.path.exists(path):
        if verbose: logging.info("Found %s in '%s'", filename, path)
        return open(path, mode)
    
    path = os.path.join(programdir, "overviewer_core", "data", "textures", filename)
    if os.path.exists(path):
        return open(path, mode)
    elif hasattr(sys, "frozen") or imp.is_frozen("__main__"):
        # windows special case, when the package dir doesn't exist
        path = os.path.join(programdir, "textures", filename)
        if os.path.exists(path):
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)

    if sys.platform == "darwin":
        path = os.path.join("/Applications/Minecraft", filename)
        if os.path.exists(path):
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
        if os.path.exists(jarpath):
            jar = zipfile.ZipFile(jarpath)
            for jarfilename in [filename, 'misc/' + filename, 'environment/' + filename]:
                try:
                    if verbose: logging.info("Found %s in '%s'", jarfilename, jarpath)
                    return jar.open(jarfilename)
                except (KeyError, IOError), e:
                    pass

    raise IOError("Could not find the file `{0}'. You can either place it in the same place as overviewer.py, use --textures-path, or install the Minecraft client.".format(filename))

def _load_image(filename):
    """Returns an image object"""
    fileobj = _find_file(filename)
    buffer = StringIO(fileobj.read())
    return Image.open(buffer).convert("RGBA")

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

##
## Image Transformation Functions
##

def transform_image_top(img):
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

def transform_image_side(img):
    """Takes an image and shears it for the left side of the cube (reflect for
    the right side)"""

    # Size of the cube side before shear
    img = img.resize((12,12), Image.ANTIALIAS)

    # Apply shear
    transform = numpy.matrix(numpy.identity(3))
    transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

    transform = numpy.array(transform)[:2,:].ravel().tolist()

    newimg = img.transform((12,18), Image.AFFINE, transform)
    return newimg

def transform_image_slope(img):
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


def transform_image_angle(img, angle):
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


def build_block(top, side):
    """From a top texture and a side texture, build a block image.
    top and side should be 16x16 image objects. Returns a 24x24 image

    """
    img = Image.new("RGBA", (24,24), bgcolor)
    
    original_texture = top.copy()
    top = transform_image_top(top)

    if not side:
        composite.alpha_over(img, top, (0,0), top)
        return img

    side = transform_image_side(side)
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

def build_full_block(top, side1, side2, side3, side4, bottom=None):
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
        side1 = transform_image_side(side1)
        side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Darken this side.
        sidealpha = side1.split()[3]
        side1 = ImageEnhance.Brightness(side1).enhance(0.9)
        side1.putalpha(sidealpha)        
        
        composite.alpha_over(img, side1, (0,0), side1)

        
    if side2 != None :
        side2 = transform_image_side(side2)

        # Darken this side.
        sidealpha2 = side2.split()[3]
        side2 = ImageEnhance.Brightness(side2).enhance(0.8)
        side2.putalpha(sidealpha2)

        composite.alpha_over(img, side2, (12,0), side2)

    if bottom != None :
        bottom = transform_image_top(bottom)
        composite.alpha_over(img, bottom, (0,12), bottom)
        
    # front sides
    if side3 != None :
        side3 = transform_image_side(side3)
        
        # Darken this side
        sidealpha = side3.split()[3]
        side3 = ImageEnhance.Brightness(side3).enhance(0.9)
        side3.putalpha(sidealpha)
        
        composite.alpha_over(img, side3, (0,6), side3)
        
    if side4 != None :
        side4 = transform_image_side(side4)
        side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken this side
        sidealpha = side4.split()[3]
        side4 = ImageEnhance.Brightness(side4).enhance(0.8)
        side4.putalpha(sidealpha)
        
        composite.alpha_over(img, side4, (12,6), side4)

    if top != None :
        top = transform_image_top(top)
        composite.alpha_over(img, top, (0, increment), top)

    return img

def build_sprite(side):
    """From a side texture, create a sprite-like texture such as those used
    for spiderwebs or flowers."""
    img = Image.new("RGBA", (24,24), bgcolor)
    
    side = transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    composite.alpha_over(img, side, (6,3), side)
    composite.alpha_over(img, otherside, (6,3), otherside)
    return img

def generate_opaque_mask(img):
    """ Takes the alpha channel of the image and generates a mask
    (used for lighting the block) that deprecates values of alpha
    smallers than 50, and sets every other value to 255. """
    
    alpha = img.split()[3]
    return alpha.point(lambda a: int(min(a, 25.5) * 10))

def tintTexture(im, c):
    # apparently converting to grayscale drops the alpha channel?
    i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
    i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
    return i

def generate_texture_tuple(img):
    """ This takes an image and returns the needed tuple for the
    blockmap dictionary."""
    if img is None:
        return None
    return (img, generate_opaque_mask(img))

##
## Biomes
##

currentBiomeFile = None
currentBiomeData = None
grasscolor = None
foliagecolor = None
watercolor = None
_north = None

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
    global _north
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

##
## Color Light
##

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

##
## The big one: generate() and associated framework
##

# placeholders that are generated in generate()
texture_dimensions = None
blockmap_generators = {}
blockmap = {}
biome_grass_texture = None

transparent_blocks = set([0,])
solid_blocks = set()
fluid_blocks = set()
nospawn_blocks = set()

# the material registration decorator
def material(blockid=[], data=[0], **kwargs):
    # mapping from property name to the set to store them in
    properties = {"transparent" : transparent_blocks, "solid" : solid_blocks, "fluid" : fluid_blocks, "nospawn" : nospawn_blocks}
    
    # make sure blockid and data are iterable
    try:
        iter(blockid)
    except:
        blockid = [blockid,]
    try:
        iter(data)
    except:
        data = [data,]
        
    def inner_material(func):
        global blockmap_generators

        # create a wrapper function with a known signature
        @functools.wraps(func)
        def func_wrapper(blockid, data, north):
            try:
                return func(blockid, data, north)
            except TypeError:
                return func(blockid, data)
        
        for block in blockid:
            # set the property sets appropriately
            for prop in properties:
                if kwargs.get(prop, False):
                    properties[prop].update([block])
            
            # populate blockmap_generators with our function
            for d in data:
                blockmap_generators[(block, d)] = func_wrapper
        
        return func_wrapper
    return inner_material

# shortcut function for pure blocks, default to solid
def block(blockid=[], top_index=None, side_index=None, **kwargs):
    new_kwargs = {'solid' : True}
    new_kwargs.update(kwargs)
    
    if top_index is None:
        raise ValueError("top_index was not provided")
    
    if side_index is None:
        side_index = top_index
    
    @material(blockid=blockid, **new_kwargs)
    def inner_block(unused_id, unused_data):
        return build_block(terrain_images[top_index], terrain_images[side_index])
    return inner_block

def generate(path=None,texture_size=24,bgc = (26,26,26,0),north_direction='lower-left'):
    global _find_file_local_path
    global bgcolor
    global texture_dimensions
    global _north
    bgcolor = bgc
    _find_file_local_path = path
    _north = north_direction
    texture_dimensions = (texture_size, texture_size)
    
    # This maps terainids to 16x16 images
    global terrain_images
    terrain_images = _split_terrain(_load_image("terrain.png"))
    
    # generate biome grass mask
    global biome_grass_texture
    biome_grass_texture = build_block(terrain_images[0], terrain_images[38])

    # generate the blocks
    global blockmap, blockmap_generators
    blockmap = {}
    for blockid, data in blockmap_generators:
        texgen = blockmap_generators[(blockid, data)]
        tex = texgen(blockid, data, north_direction)
        blockmap[(blockid, data)] = generate_texture_tuple(tex)
    
    if texture_size != 24:
        # rescale biome textures.
        biome_grass_texture = biome_grass_texture.resize(texture_dimensions, Image.ANTIALIAS)

        # rescale the special block images
        for blockid, data in iter(blockmap):
            block = blockmap[(blockid,data)]
            if block != None:
                block = block[0]
                scaled_block = block.resize(texture_dimensions, Image.ANTIALIAS)
                blockmap[(blockid,data)] = generate_texture_tuple(scaled_block, blockid)

##
## and finally: actual texture definitions
##

# stone
block(blockid=1, top_index=1)

@material(blockid=2, data=range(11)+[0x10,], solid=True)
def grass(blockid, data):
    # 0x10 bit means SNOW
    side_img = terrain_images[3]
    if data & 0x10:
        side_img = terrain_images[68]
    img = build_block(terrain_images[0], side_img)
    if not data & 0x10:
        global biome_grass_texture
        composite.alpha_over(img, biome_grass_texture, (0, 0), biome_grass_texture)
    return img

# dirt
block(blockid=3, top_index=2)
# cobblestone
block(blockid=4, top_index=16)
# wooden plank
block(blockid=5, top_index=4)

@material(blockid=6, data=range(16), transparent=True)
def saplings(blockid, data):
    # usual saplings
    tex = terrain_images[15]
    
    if data & 0x3 == 1: # spruce sapling
        tex = terrain_images[63]
    if data & 0x3 == 2: # birch sapling
        tex = terrain_images[79]
    
    return build_sprite(tex)

# bedrock
block(blockid=7, top_index=17)

@material(blockid=8, data=range(16), fluid=True, transparent=True)
def water(blockid, data):
    watertex = _load_image("water.png")
    return build_block(watertex, watertex)

# other water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[9, 20, 79], data=range(32), transparent=True, nospawn=True)
def no_inner_surfaces(blockid, data):
    if blockid == 9:
        texture = _load_image("water.png")
    elif blockid == 20:
        texture = terrain_images[49]
    else:
        texture = terrain_images[67]
        
    if (data & 0b10000) == 16:
        top = texture
    else:
        top = None
        
    if (data & 0b0001) == 1:
        side1 = texture    # top left
    else:
        side1 = None
    
    if (data & 0b1000) == 8:
        side2 = texture    # top right           
    else:
        side2 = None
    
    if (data & 0b0010) == 2:
        side3 = texture    # bottom left    
    else:
        side3 = None
    
    if (data & 0b0100) == 4:
        side4 = texture    # bottom right
    else:
        side4 = None
    
    # if nothing shown do not draw at all
    if top is None and side3 is None and side4 is None:
        return None
    
    img = build_full_block(top,None,None,side3,side4)
    return img

@material(blockid=[10, 11], data=range(16), fluid=True, transparent=False)
def lava(blockid, data):
    lavatex = _load_image("lava.png")
    return build_block(lavatex, lavatex)

# sand
block(blockid=12, top_index=18)
# gravel
block(blockid=13, top_index=19)
# gold ore
block(blockid=14, top_index=32)
# iron ore
block(blockid=15, top_index=33)
# coal ore
block(blockid=16, top_index=34)

@material(blockid=17, data=range(3), solid=True)
def wood(blockid, data):
    top = terrain_images[21]
    if data == 0: # normal
        return build_block(top, terrain_images[20])
    if data == 1: # birch
        return build_block(top, terrain_images[116])
    if data == 2: # pine
        return build_block(top, terrain_images[117])

@material(blockid=18, data=range(16), transparent=True, solid=True)
def leaves(blockid, data):
    t = terrain_images[52]
    if data == 1:
        # pine!
        t = terrain_images[132]
    return build_block(t, t)

# sponge
block(blockid=19, top_index=48)
# lapis lazuli ore
block(blockid=21, top_index=160)
# lapis lazuli block
block(blockid=22, top_index=144)

# dispensers, furnaces, and burning furnaces
@material(blockid=[23, 61, 62], data=range(6), solid=True)
def furnaces(blockid, data, north):
    # first, do the north rotation if needed
    if north == 'upper-left':
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif north == 'upper-right':
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif north == 'lower-right':
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2
    
    top = terrain_images[62]
    side = terrain_images[45]
    
    if blockid == 61:
        front = terrain_images[44]
    elif blockid == 62:
        front = terrain_images[61]
    elif blockid == 23:
        front = terrain_images[46]
    
    if data == 3: # pointing west
        return build_full_block(top, None, None, side, front)
    elif data == 4: # pointing north
        return build_full_block(top, None, None, front, side)
    else: # in any other direction the front can't be seen
        return build_full_block(top, None, None, side, side)

# sandstone
block(blockid=24, top_index=176, side_index=192)
# note block
block(blockid=25, top_index=74)
