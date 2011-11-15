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

def build_billboard(tex):
    """From a texture, create a billboard-like texture such as those used for
    tall grass or melon stems."""
    img = Image.new("RGBA", (24,24), bgcolor)
    
    front = tex.resize((14, 11), Image.ANTIALIAS)
    composite.alpha_over(img, front, (5,9))
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
    blockmap array."""
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
blockmap = []
biome_grass_texture = None

known_blocks = set()
used_datas = set()
max_blockid = 0
max_data = 0

transparent_blocks = set()
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
        
        used_datas.update(data)
        for block in blockid:
            # set the property sets appropriately
            known_blocks.update([block])
            for prop in properties:
                try:
                    if block in kwargs.get(prop, []):
                        properties[prop].update([block])
                except TypeError:
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

# shortcut function for sprite blocks, defaults to transparent
def sprite(blockid=[], index=None, **kwargs):
    new_kwargs = {'transparent' : True}
    new_kwargs.update(kwargs)
    
    if index is None:
        raise ValueError("index was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_sprite(unused_id, unused_data):
        return build_sprite(terrain_images[index])
    return inner_sprite

# shortcut function for billboard blocks, defaults to transparent
def billboard(blockid=[], index=None, **kwargs):
    new_kwargs = {'transparent' : True}
    new_kwargs.update(kwargs)
    
    if index is None:
        raise ValueError("index was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_billboard(unused_id, unused_data):
        return build_billboard(terrain_images[index])
    return inner_billboard

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
    global max_blockid, max_data
    max_blockid = max(known_blocks) + 1
    max_data = max(used_datas) + 1
    blockmap = [None] * max_blockid * max_data
    
    for blockid, data in blockmap_generators:
        texgen = blockmap_generators[(blockid, data)]
        tex = texgen(blockid, data, north_direction)
        blockmap[blockid * max_data + data] = generate_texture_tuple(tex)
    
    if texture_size != 24:
        # rescale biome textures.
        biome_grass_texture = biome_grass_texture.resize(texture_dimensions, Image.ANTIALIAS)

        # rescale the special block images
        for i, tex in enumerate(blockmap):
            if tex != None:
                block = tex[0]
                scaled_block = block.resize(texture_dimensions, Image.ANTIALIAS)
                blockmap[i] = generate_texture_tuple(scaled_block)

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

@material(blockid=8, data=range(16), fluid=True, transparent=True, nospawn=True)
def water(blockid, data):
    watertex = _load_image("water.png")
    return build_block(watertex, watertex)

# other water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[9, 20, 79], data=range(32), fluid=(9,), transparent=True, nospawn=True, solid=(79, 20))
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

@material(blockid=[10, 11], data=range(16), fluid=True, transparent=False, nospawn=True)
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

@material(blockid=26, data=range(12), transparent=True, nospawn=True)
def bed(blockid, data, north):
    # first get north rotation done
    # Masked to not clobber block head/foot info
    if north == 'upper-left':
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif north == 'upper-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif north == 'lower-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    
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
    return build_full_block(top, None, None, left_face, right_face)

# powered, detector, and normal rails
@material(blockid=[27, 28, 66], data=range(14), transparent=True)
def rails(blockid, data, north):
    # first, do north rotation
    # Masked to not clobber powered rail on/off info
    # Ascending and flat straight
    if north == 'upper-left':
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif north == 'upper-right':
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif north == 'lower-right':
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    if blockid == 66: # normal minetrack only
        #Corners
        if north == 'upper-left':
            if data == 6: data = 7
            elif data == 7: data = 8
            elif data == 8: data = 6
            elif data == 9: data = 9
        elif north == 'upper-right':
            if data == 6: data = 8
            elif data == 7: data = 9
            elif data == 8: data = 6
            elif data == 9: data = 7
        elif north == 'lower-right':
            if data == 6: data = 9
            elif data == 7: data = 6
            elif data == 8: data = 8
            elif data == 9: data = 7
    img = Image.new("RGBA", (24,24), bgcolor)
    
    if blockid == 27: # powered rail
        if data & 0x8 == 0: # unpowered
            raw_straight = terrain_images[163]
            raw_corner = terrain_images[112]    # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = terrain_images[179]
            raw_corner = terrain_images[112]    # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7
            
    elif blockid == 28: # detector rail
        raw_straight = terrain_images[195]
        raw_corner = terrain_images[112]    # leave corners for code simplicity
        
    elif blockid == 66: # normal rail
        raw_straight = terrain_images[128]
        raw_corner = terrain_images[112]
        
    ## use transform_image to scale and shear
    if data == 0:
        track = transform_image_top(raw_straight)
        composite.alpha_over(img, track, (0,12), track)
    elif data == 6:
        track = transform_image_top(raw_corner)
        composite.alpha_over(img, track, (0,12), track)
    elif data == 7:
        track = transform_image_top(raw_corner.rotate(270))
        composite.alpha_over(img, track, (0,12), track)
    elif data == 8:
        # flip
        track = transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM).rotate(90))
        composite.alpha_over(img, track, (0,12), track)
    elif data == 9:
        track = transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM))
        composite.alpha_over(img, track, (0,12), track)
    elif data == 1:
        track = transform_image_top(raw_straight.rotate(90))
        composite.alpha_over(img, track, (0,12), track)
        
    #slopes
    elif data == 2: # slope going up in +x direction
        track = transform_image_slope(raw_straight)
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
        track = transform_image_slope(raw_straight)
        composite.alpha_over(img, track, (0,0), track)
        
    elif data == 5: # slope going up in +y direction
        # same as "data == 3"
        ImageDraw.Draw(img).line([(1,17),(12,11)],fill=(164,164,164))
        
    return img

# sticky and normal piston body
@material(blockid=[29, 33], data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, solid=True, nospawn=True)
def piston(blockid, data, north):
    # first, north rotation
    # Masked to not clobber block head/foot info
    if north == 'upper-left':
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif north == 'upper-right':
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif north == 'lower-right':
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    
    if blockid == 29: # sticky
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
            img = build_full_block(back_t ,None ,None ,side_t, side_t)
            
        elif data & 0x07 == 0x1: # up
            img = build_full_block((interior_t, 4) ,None ,None ,side_t, side_t)
            
        elif data & 0x07 == 0x2: # east
            img = build_full_block(side_t , None, None ,side_t.rotate(90), back_t)
            
        elif data & 0x07 == 0x3: # west
            img = build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), None)
            temp = transform_image_side(interior_t)
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            composite.alpha_over(img, temp, (9,5), temp)
            
        elif data & 0x07 == 0x4: # north
            img = build_full_block(side_t.rotate(90) ,None ,None , None, side_t.rotate(270))
            temp = transform_image_side(interior_t)
            composite.alpha_over(img, temp, (3,5), temp)
            
        elif data & 0x07 == 0x5: # south
            img = build_full_block(side_t.rotate(270) ,None , None ,back_t, side_t.rotate(90))

    else: # pushed in, normal full blocks, easy stuff
        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = build_full_block(back_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x1: # up
            img = build_full_block(piston_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x2: # east 
            img = build_full_block(side_t ,None ,None ,side_t.rotate(90), back_t)
        elif data & 0x07 == 0x3: # west
            img = build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        elif data & 0x07 == 0x4: # north
            img = build_full_block(side_t.rotate(90) ,None ,None ,piston_t, side_t.rotate(270))
        elif data & 0x07 == 0x5: # south
            img = build_full_block(side_t.rotate(270) ,None ,None ,back_t, side_t.rotate(90))
            
    return img

# sticky and normal piston shaft
@material(blockid=34, data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, nospawn=True)
def piston_extension(blockid, data, north):
    # first, north rotation
    # Masked to not clobber block head/foot info
    if north == 'upper-left':
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif north == 'upper-right':
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif north == 'lower-right':
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    
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
    temp = transform_image_side(side_t)
    composite.alpha_over(h_stick, temp, (1,7), temp)
    temp = transform_image_top(side_t.rotate(90))
    composite.alpha_over(h_stick, temp, (1,1), temp)
    # Darken it
    sidealpha = h_stick.split()[3]
    h_stick = ImageEnhance.Brightness(h_stick).enhance(0.85)
    h_stick.putalpha(sidealpha)
    
    # generate the vertical piston extension stick
    v_stick = Image.new("RGBA", (24,24), bgcolor)
    temp = transform_image_side(side_t.rotate(90))
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
        img = build_full_block((back_t, 12) ,None ,None ,side_t, side_t)
        composite.alpha_over(img, v_stick, (0,-3), v_stick)
    elif data & 0x07 == 0x1: # up
        img = Image.new("RGBA", (24,24), bgcolor)
        img2 = build_full_block(piston_t ,None ,None ,side_t, side_t)
        composite.alpha_over(img, v_stick, (0,4), v_stick)
        composite.alpha_over(img, img2, (0,0), img2)
    elif data & 0x07 == 0x2: # east 
        img = build_full_block(side_t ,None ,None ,side_t.rotate(90), None)
        temp = transform_image_side(back_t).transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(img, temp, (2,2), temp)
        composite.alpha_over(img, h_stick, (6,3), h_stick)
    elif data & 0x07 == 0x3: # west
        img = Image.new("RGBA", (24,24), bgcolor)
        img2 = build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        composite.alpha_over(img, h_stick, (0,0), h_stick)
        composite.alpha_over(img, img2, (0,0), img2)            
    elif data & 0x07 == 0x4: # north
        img = build_full_block(side_t.rotate(90) ,None ,None , piston_t, side_t.rotate(270))
        composite.alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (0,0), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
    elif data & 0x07 == 0x5: # south
        img = Image.new("RGBA", (24,24), bgcolor)
        img2 = build_full_block(side_t.rotate(270) ,None ,None ,None, side_t.rotate(90))
        temp = transform_image_side(back_t)
        composite.alpha_over(img2, temp, (10,2), temp)
        composite.alpha_over(img, img2, (0,0), img2)
        composite.alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (-3,2), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
        
    return img

# cobweb
sprite(blockid=30, index=11, nospawn=True)

@material(blockid=31, data=range(3), transparent=True)
def tall_grass(blockid, data):
    if data == 0: # dead shrub
        texture = terrain_images[55]
    elif data == 1: # tall grass
        texture = terrain_images[39]
    elif data == 2: # fern
        texture = terrain_images[56]
    
    return build_billboard(texture)

# dead bush
billboard(blockid=32, index=55)

@material(blockid=35, data=range(16), solid=True)
def wool(blockid, data):
    if data == 0: # white
        texture = terrain_images[64]
    elif data == 1: # orange
        texture = terrain_images[210]
    elif data == 2: # magenta
        texture = terrain_images[194]
    elif data == 3: # light blue
        texture = terrain_images[178]
    elif data == 4: # yellow
        texture = terrain_images[162]
    elif data == 5: # light green
        texture = terrain_images[146]
    elif data == 6: # pink
        texture = terrain_images[130]
    elif data == 7: # grey
        texture = terrain_images[114]
    elif data == 8: # light grey
        texture = terrain_images[225]
    elif data == 9: # cyan
        texture = terrain_images[209]
    elif data == 10: # purple
        texture = terrain_images[193]
    elif data == 11: # blue
        texture = terrain_images[177]
    elif data == 12: # brown
        texture = terrain_images[161]
    elif data == 13: # dark green
        texture = terrain_images[145]
    elif data == 14: # red
        texture = terrain_images[129]
    elif data == 15: # black
        texture = terrain_images[113]
    
    return build_block(texture, texture)

# dandelion
sprite(blockid=37, index=13)
# rose
sprite(blockid=38, index=12)
# brown mushroom
sprite(blockid=39, index=29)
# red mushroom
sprite(blockid=40, index=28)
# block of gold
block(blockid=41, top_index=23)
# block of iron
block(blockid=42, top_index=22)

# double slabs and slabs
@material(blockid=[43, 44], data=range(6), transparent=(44,), solid=True)
def slabs(blockid, data):
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
    
    if blockid == 43: # double slab
        return build_block(top, side)
    
    # cut the side texture in half
    mask = side.crop((0,8,16,16))
    side = Image.new(side.mode, side.size, bgcolor)
    composite.alpha_over(side, mask,(0,0,16,8), mask)
    
    # plain slab
    top = transform_image_top(top)
    side = transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)
    
    img = Image.new("RGBA", (24,24), bgcolor)
    composite.alpha_over(img, side, (0,12), side)
    composite.alpha_over(img, otherside, (12,12), otherside)
    composite.alpha_over(img, top, (0,6), top)
    
    return img

# brick block
block(blockid=45, top_index=7)
# TNT
block(blockid=46, top_index=9, side_index=8, nospawn=True)
# bookshelf
block(blockid=47, top_index=4, side_index=35)
# moss stone
block(blockid=48, top_index=36)
# obsidian
block(blockid=49, top_index=37)

# torch, redstone torch (off), redstone torch(on)
@material(blockid=[50, 75, 76], data=[1, 2, 3, 4, 5], transparent=True)
def torches(blockid, data, north):
    # first, north rotations
    if north == 'upper-left':
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
    elif north == 'upper-right':
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif north == 'lower-right':
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2
    
    # choose the proper texture
    if blockid == 50: # torch
        small = terrain_images[80]
    elif blockid == 75: # off redstone torch
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
        img = build_full_block(None, None, None, torch, None, None)
        
    elif data == 2: # pointing north
        torch = torch.rotate(rotation, Image.NEAREST)
        img = build_full_block(None, None, torch, None, None, None)
        
    elif data == 3: # pointing west
        torch = torch.rotate(rotation, Image.NEAREST)
        img = build_full_block(None, torch, None, None, None, None)
        
    elif data == 4: # pointing east
        torch = torch.rotate(-rotation, Image.NEAREST)
        img = build_full_block(None, None, None, None, torch, None)
        
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
        
    return img

# fire
@material(blockid=51, data=range(16), transparent=True)
def fire(blockid, data):
    firetexture = _load_image("fire.png")
    side1 = transform_image_side(firetexture)
    side2 = transform_image_side(firetexture).transpose(Image.FLIP_LEFT_RIGHT)
    
    img = Image.new("RGBA", (24,24), bgcolor)

    composite.alpha_over(img, side1, (12,0), side1)
    composite.alpha_over(img, side2, (0,0), side2)

    composite.alpha_over(img, side1, (0,6), side1)
    composite.alpha_over(img, side2, (12,6), side2)
    
    return img

# monster spawner
block(blockid=52, top_index=34, transparent=True)

# wooden, cobblestone, red brick, stone brick and netherbrick stairs.
@material(blockid=[53,67,108,109,114], data=range(4), transparent=True, solid=True, nospawn=True)
def stairs(blockid, data, north):

    # first, north rotations
    if north == 'upper-left':
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 1
        elif data == 3: data = 0
    elif north == 'upper-right':
        if data == 0: data = 1
        elif data == 1: data = 0
        elif data == 2: data = 3
        elif data == 3: data = 2
    elif north == 'lower-right':
        if data == 0: data = 3
        elif data == 1: data = 2
        elif data == 2: data = 0
        elif data == 3: data = 1

    if blockid == 53: # wooden
        texture = terrain_images[4]
    elif blockid == 67: # cobblestone
        texture = terrain_images[16]
    elif blockid == 108: # red brick stairs
        texture = terrain_images[7]
    elif blockid == 109: # stone brick stairs
        texture = terrain_images[54]
    elif blockid == 114: # netherbrick stairs
        texture = terrain_images[224]
    
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
        img = build_full_block(half_block_r, None, None, half_block_d, side.transpose(Image.FLIP_LEFT_RIGHT))
        tmp1 = transform_image_side(half_block_u)
        
        # Darken the vertical part of the second step
        sidealpha = tmp1.split()[3]
        # darken it a bit more than usual, looks better
        tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.8)
        tmp1.putalpha(sidealpha)
        
        composite.alpha_over(img, tmp1, (6,4)) #workaround, fixes a hole
        composite.alpha_over(img, tmp1, (6,3))
        tmp2 = transform_image_top(half_block_l)
        composite.alpha_over(img, tmp2, (0,6))
        
    elif data == 1: # ascending north
        img = Image.new("RGBA", (24,24), bgcolor) # first paste the texture in the back
        tmp1 = transform_image_top(half_block_r)
        composite.alpha_over(img, tmp1, (0,6))
        tmp2 = build_full_block(half_block_l, None, None, texture, side)
        composite.alpha_over(img, tmp2)
    
    elif data == 2: # ascending west
        img = Image.new("RGBA", (24,24), bgcolor) # first paste the texture in the back
        tmp1 = transform_image_top(half_block_u)
        composite.alpha_over(img, tmp1, (0,6))
        tmp2 = build_full_block(half_block_d, None, None, side, texture)
        composite.alpha_over(img, tmp2)
        
    elif data == 3: # ascending east
        img = build_full_block(half_block_u, None, None, side.transpose(Image.FLIP_LEFT_RIGHT), half_block_d)
        tmp1 = transform_image_side(half_block_u).transpose(Image.FLIP_LEFT_RIGHT)
        
        # Darken the vertical part of the second step
        sidealpha = tmp1.split()[3]
        # darken it a bit more than usual, looks better
        tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.7)
        tmp1.putalpha(sidealpha)
        
        composite.alpha_over(img, tmp1, (6,4)) #workaround, fixes a hole
        composite.alpha_over(img, tmp1, (6,3))
        tmp2 = transform_image_top(half_block_d)
        composite.alpha_over(img, tmp2, (0,6))
        
        # touch up a (horrible) pixel
        img.putpixel((18,3),(0,0,0,0))
        
    return img

# normal and locked chest (locked was the one used in april fools' day)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[54,95], data=range(12), solid=True)
def chests(blockid, data):
    # First two bits of the pseudo data store if it's a single chest
    # or it's a double chest, first half or second half (left to right).
    # The last two bits store the orientation.
    
    # No need for north stuff, uses pseudo data and rotates with the map

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
        img = build_full_block(top, None, None, side, front)

    elif data & 3 == 1: # north
        img = build_full_block(top, None, None, front, side)

    elif data & 3 == 2: # east
        img = build_full_block(top, None, None, side, back)

    elif data & 3 == 3: # south
        img = build_full_block(top, None, None, back, side)
        
    else:
        img = build_full_block(top, None, None, back, side)

    return img

# redstone wire
# uses pseudo-ancildata found in iterate.c
@material(blockid=55, data=range(128), transparent=True)
def wire(blockid, data):

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
        
    img = build_full_block(None,side1,side2,None,None,bottom)

    return img

# diamond ore
block(blockid=56, top_index=50)
# diamond block
block(blockid=57, top_index=24)

# crafting table
# needs two different sides
@material(blockid=58, solid=True)
def crafting_table(blockid, data):
    top = terrain_images[43]
    side3 = terrain_images[43+16]
    side4 = terrain_images[43+16+1]
    
    img = build_full_block(top, None, None, side3, side4, None)
    return img

# crops
@material(blockid=59, data=range(8), transparent=True, nospawn=True)
def crops(blockid, data):
    raw_crop = terrain_images[88+data]
    crop1 = transform_image_top(raw_crop)
    crop2 = transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), bgcolor)
    composite.alpha_over(img, crop1, (0,12), crop1)
    composite.alpha_over(img, crop2, (6,3), crop2)
    composite.alpha_over(img, crop3, (6,3), crop3)
    return img

# farmland
@material(blockid=60, data=range(9), solid=True)
def farmland(blockid, data):
    top = terrain_images[86]
    if data == 0:
        top = terrain_images[87]
    return build_block(top, terrain_images[2])

# signposts
@material(blockid=63, data=range(16), transparent=True)
def signpost(blockid, data, north):

    # first north rotations
    if north == 'upper-left':
        data = (data + 4) % 16
    elif north == 'upper-right':
        data = (data + 8) % 16
    elif north == 'lower-right':
        data = (data + 12) % 16

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
    # post2 is a brighter signpost pasted with a small shift,
    # gives to the signpost some 3D effect.
    post2 = ImageEnhance.Brightness(post).enhance(1.2)
    composite.alpha_over(img, post2,(incrementx, -3),post2)
    composite.alpha_over(img, post, (0,-2), post)

    return img


# wooden and iron door
@material(blockid=[64,71], data=range(16), transparent=True)
def door(blockid, data, north):
    #Masked to not clobber block top/bottom & swung info
    if north == 'upper-left':
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif north == 'upper-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif north == 'lower-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2

    if data & 0x8 == 0x8: # top of the door
        raw_door = terrain_images[81 if blockid == 64 else 82]
    else: # bottom of the door
        raw_door = terrain_images[97 if blockid == 64 else 98]
    
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
    
    return img

# ladder
@material(blockd=65, data=[2, 3, 4, 5], transparent=True)
def ladder(blockid, data, north):

    # first north rotations
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

    img = Image.new("RGBA", (24,24), bgcolor)
    raw_texture = terrain_images[83]

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


# wall signs
@material(blockid=68, data=[2, 3, 4, 5], transparent=True)
def wall_sign(blockid, data, north): # wall sign

    # first north rotations
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
        sign = build_full_block(None, None, None, None, texture)
    elif data == 3:  # west
        incrementx = -1
        sign = build_full_block(None, texture, None, None, None)
    elif data == 4:  # north
        incrementx = +1
        sign = build_full_block(None, None, texture, None, None)
    elif data == 5:  # south
        incrementx = -1
        sign = build_full_block(None, None, None, texture, None)

    sign2 = ImageEnhance.Brightness(sign).enhance(1.2)
    composite.alpha_over(img, sign2,(incrementx, 2),sign2)
    composite.alpha_over(img, sign, (0,3), sign)

    return img

##
## not rendered: levers
##
@material(blockid=69, data=range(16), transparent=True)
def levers(blockid, data, north):
    # place holder, used to mae the block transparent
    return None

# wooden and stone pressure plates
@material(blockid=[70, 72], data=[0,1], transparent=True)
def pressure_plate(blockid, data):
    if blockid == 70: # stone
        t = terrain_images[1].copy()
    else: # wooden
        t = terrain_images[4].copy()
    
    # cut out the outside border, pressure plates are smaller
    # than a normal block
    ImageDraw.Draw(t).rectangle((0,0,15,15),outline=(0,0,0,0))
    
    # create the textures and a darker version to make a 3d by 
    # pasting them with an offstet of 1 pixel
    img = Image.new("RGBA", (24,24), bgcolor)
    
    top = transform_image_top(t)
    
    alpha = top.split()[3]
    topd = ImageEnhance.Brightness(top).enhance(0.8)
    topd.putalpha(alpha)
    
    #show it 3d or 2d if unpressed or pressed
    if data == 0:
        composite.alpha_over(img,topd, (0,12),topd)
        composite.alpha_over(img,top, (0,11),top)
    elif data == 1:
        composite.alpha_over(img,top, (0,12),top)
    
    return img

# normal and glowing redstone ore
block(blockid=[73, 74], top_index=51)

@material(blockid=77, data=range(16), transparent=True)
def buttons(blockid, data, north):

    # 0x8 is set if the button is pressed mask this info and render
    # it as unpressed
    data = data & 0x7

    if north == 'upper-left':
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
    elif north == 'upper-right':
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif north == 'lower-right':
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2

    t = terrain_images[1].copy()

    # generate the texture for the button
    ImageDraw.Draw(t).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), bgcolor)

    button = transform_image_side(t)
    
    if data == 1: # facing SOUTH
        # buttons can't be placed in transparent blocks, so this
        # direction can't be seen
        return None

    elif data == 2: # facing NORTH
        # paste it twice with different brightness to make a 3D effect
        composite.alpha_over(img, button, (12,-1), button)

        alpha = button.split()[3]
        button = ImageEnhance.Brightness(button).enhance(0.9)
        button.putalpha(alpha)
        
        composite.alpha_over(img, button, (11,0), button)

    elif data == 3: # facing WEST
        # paste it twice with different brightness to make a 3D effect
        button = button.transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(img, button, (0,-1), button)

        alpha = button.split()[3]
        button = ImageEnhance.Brightness(button).enhance(0.9)
        button.putalpha(alpha)
        
        composite.alpha_over(img, button, (1,0), button)

    elif data == 4: # facing EAST
        # buttons can't be placed in transparent blocks, so this
        # direction can't be seen
        return None

    return img

# snow
@material(blockid=78, data=range(8), transparent=True, solid=True)
def snow(blockid, data):
    # still not rendered correctly: data other than 0
    
    tex = terrain_images[66]
    
    # make the side image, top 3/4 transparent
    mask = tex.crop((0,12,16,16))
    sidetex = Image.new(tex.mode, tex.size, bgcolor)
    composite.alpha_over(sidetex, mask, (0,12,16,16), mask)
    
    img = Image.new("RGBA", (24,24), bgcolor)
    
    top = transform_image_top(tex)
    side = transform_image_side(sidetex)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    composite.alpha_over(img, side, (0,6), side)
    composite.alpha_over(img, otherside, (12,6), otherside)
    composite.alpha_over(img, top, (0,9), top)
    
    return img

# snow block
block(blockid=80, top_index=66)

# cactus
@material(blockid=81, data=range(15), transparent=True, solid=True, nospawn=True)
def cactus(blockid, data):
    top = terrain_images[69]
    side = terrain_images[70]

    img = Image.new("RGBA", (24,24), bgcolor)
    
    top = transform_image_top(top)
    side = transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)

    composite.alpha_over(img, side, (1,6), side)
    composite.alpha_over(img, otherside, (11,6), otherside)
    composite.alpha_over(img, top, (0,0), top)
    
    return img

# clay block
block(blockid=82, top_index=72)

# sugar cane
@material(blockid=83, data=range(16), transparent=True)
def sugar_cane(blockid, data):
    tex = terrain_images[73]
    return build_sprite(tex)

# jukebox
@material(blockid=84, data=range(16), solid=True)
def jukebox(blockid, data):
    return build_block(terrain_images[75], terrain_images[74])

# nether and normal fences
# uses pseudo-ancildata found in iterate.c
@material(blockid=[85, 113], data=range(16), transparent=True, nospawn=True)
def fence(blockid, data):
    # no need for north rotations, it uses pseudo data.
    # create needed images for Big stick fence
    if blockid == 85: # normal fence
        fence_top = terrain_images[4].copy()
        fence_side = terrain_images[4].copy()
        fence_small_side = terrain_images[4].copy()
    else: # netherbrick fence
        fence_top = terrain_images[224].copy()
        fence_side = terrain_images[224].copy()
        fence_small_side = terrain_images[224].copy()

    # generate the textures of the fence
    ImageDraw.Draw(fence_top).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(fence_side).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_side).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the big stick
    fence_side = transform_image_side(fence_side)
    fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
    fence_top = transform_image_top(fence_top)

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
    ImageDraw.Draw(fence_small_side).rectangle((0,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,4,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the small sticks
    fence_small_side = transform_image_side(fence_small_side)
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
        
    return img

# pumpkin
@material(blockid=[86, 91], data=range(4), solid=True)
def pumpkin(blockid, data, north): # pumpkins, jack-o-lantern
    # north rotation
    if north == 'upper-left':
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif north == 'upper-right':
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif north == 'lower-right':
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2
    
    # texture generation
    top = terrain_images[102]
    frontID = 119 if blockid == 86 else 120
    front = terrain_images[frontID]
    side = terrain_images[118]

    if data == 0: # pointing west
        img = build_full_block(top, None, None, side, front)

    elif data == 1: # pointing north
        img = build_full_block(top, None, None, front, side)

    else: # in any other direction the front can't be seen
        img = build_full_block(top, None, None, side, side)

    return img

# netherrack
block(blockid=87, top_index=103)

# soul sand
block(blockid=88, top_index=104)

# glowstone
block(blockid=89, top_index=105)

# portal
@material(blockid=90, data=[1, 2, 4, 8], transparent=True)
def portal(blockid, data):
    # no north orientation uses pseudo data
    portaltexture = _load_image("portal.png")
    img = Image.new("RGBA", (24,24), bgcolor)

    side = transform_image_side(portaltexture)
    otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

    if data in (1,4):
        composite.alpha_over(img, side, (5,4), side)

    if data in (2,8):
        composite.alpha_over(img, otherside, (5,4), otherside)

    return img

# cake!
# TODO is rendered un-bitten
@material(blockid=92, data=range(6), transparent=True, nospawn=True)
def cake(blockid, data):

    # choose textures for cake
    top = terrain_images[121]
    side = terrain_images[122]
    top = transform_image_top(top)
    side = transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    # darken sides slightly
    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)
    
    img = Image.new("RGBA", (24,24), bgcolor)
    
    # composite the cake
    composite.alpha_over(img, side, (1,6), side)
    composite.alpha_over(img, otherside, (11,7), otherside) # workaround, fixes a hole
    composite.alpha_over(img, otherside, (12,6), otherside)
    composite.alpha_over(img, top, (0,6), top)

    return img

# redstone repeaters ON and OFF
@material(blockid=[93,94], data=range(16), transparent=True, nospawn=True)
def repeater(blockid, data, north):
    # north rotation
    # Masked to not clobber delay info
    if north == 'upper-left':
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif north == 'upper-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif north == 'lower-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    
    # generate the diode
    top = terrain_images[131] if blockid == 93 else terrain_images[147]
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

    img = build_full_block( (top, increment), None, None, side, side)

    # compose a "3d" redstone torch
    t = terrain_images[115].copy() if blockid == 93 else terrain_images[99].copy()
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

    return img
    
# trapdoor
# TODO the trapdoor is looks like a sprite when opened, that's not good
@material(blockid=96, data=range(8), transparent=True, nospawn=True)
def trapdoor(blockid, data, north):

    # north rotation
    # Masked to not clobber opened/closed info
    if north == 'upper-left':
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif north == 'upper-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    elif north == 'lower-right':
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0

    # texture generation
    texture = terrain_images[84]
    if data & 0x4 == 0x4: # opened trapdoor
        if data & 0x3 == 0: # west
            img = build_full_block(None, None, None, None, texture)
        if data & 0x3 == 1: # east
            img = build_full_block(None, texture, None, None, None)
        if data & 0x3 == 2: # south
            img = build_full_block(None, None, texture, None, None)
        if data & 0x3 == 3: # north
            img = build_full_block(None, None, None, texture, None)
        
    elif data & 0x4 == 0: # closed trapdoor
        img = build_full_block((texture, 12), None, None, texture, texture)
    
    return img

# block with hidden silverfish (stone, cobblestone and stone brick)
@material(blockid=97, data=range(3), solid=True)
def hidden_silverfish(blockid, data):
    if data == 0: # stone
        t = terrain_images[1]
    elif data == 1: # cobblestone
        t = terrain_images[16]
    elif data == 2: # stone brick
        t = terrain_images[54]
    
    img = build_block(t, t)
    
    return img

# stone brick
@material(blockid=98, data=range(3), solid=True)
def stone_brick(blockid, data):
    if data == 0: # normal
        t = terrain_images[54]
    elif data == 1: # mossy
        t = terrain_images[100]
    else: # cracked
        t = terrain_images[101]

    img = build_full_block(t, None, None, t, t)

    return img

# huge brown and red mushroom
@material(blockid=[99,100], data=range(11), solid=True)
def huge_mushroom(blockid, data, north):
    # north rotation
    if north == 'upper-left':
        if data == 1: data = 3
        elif data == 2: data = 6
        elif data == 3: data = 9
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 1
        elif data == 8: data = 4
        elif data == 9: data = 7
    elif north == 'upper-right':
        if data == 1: data = 9
        elif data == 2: data = 8
        elif data == 3: data = 7
        elif data == 4: data = 6
        elif data == 6: data = 4
        elif data == 7: data = 3
        elif data == 8: data = 2
        elif data == 9: data = 1
    elif north == 'lower-right':
        if data == 1: data = 7
        elif data == 2: data = 4
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 9
        elif data == 8: data = 6
        elif data == 9: data = 3

    # texture generation
    if blockid == 99: # brown
        cap = terrain_images[126]
    else: # red
        cap = terrain_images[125]

    stem = terrain_images[141]
    porous = terrain_images[142]
    
    if data == 0: # fleshy piece
        img = build_full_block(porous, None, None, porous, porous)

    if data == 1: # north-east corner
        img = build_full_block(cap, None, None, cap, porous)

    if data == 2: # east side
        img = build_full_block(cap, None, None, porous, porous)

    if data == 3: # south-east corner
        img = build_full_block(cap, None, None, porous, cap)

    if data == 4: # north side
        img = build_full_block(cap, None, None, cap, porous)

    if data == 5: # top piece
        img = build_full_block(cap, None, None, porous, porous)

    if data == 6: # south side
        img = build_full_block(cap, None, None, cap, porous)

    if data == 7: # north-west corner
        img = build_full_block(cap, None, None, cap, cap)

    if data == 8: # west side
        img = build_full_block(cap, None, None, porous, cap)

    if data == 9: # south-west corner
        img = build_full_block(cap, None, None, porous, cap)

    if data == 10: # stem
        img = build_full_block(porous, None, None, stem, stem)

    return img

# iron bars and glass pane
# TODO glass pane is not a sprite, it has a texture for the side,
# at the moment is not used
@material(blockid=[101,102], data=range(16), transparent=True, nospawn=True)
def panes(blockid, data):
    # no north rotation, uses pseudo data
    if blockid == 101:
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

    return img

# melon
block(blockid=103, top_index=137, side_index=136, solid=True)

# pumpkin and melon stem
# TODO To render it as in game needs from pseudo data and ancil data:
# once fully grown the stem bends to the melon/pumpkin block,
# at the moment only render the growing stem
@material(blockid=[104,105], data=range(8), transparent=True)
def stem(blockid, data, north):
    # the ancildata value indicates how much of the texture
    # is shown.

    # not fully grown stem or no pumpkin/melon touching it,
    # straight up stem
    t = terrain_images[111].copy()
    img = Image.new("RGBA", (16,16), bgcolor)
    composite.alpha_over(img, t, (0, int(16 - 16*((data + 1)/8.))), t)
    img = build_sprite(t)
    if data & 7 == 7:
        # fully grown stem gets brown color!
        # there is a conditional in rendermode-normal.c to not
        # tint the data value 7
        img = tintTexture(img, (211,169,116))
    return img
    

# vines
# TODO multiple sides of a block can contain vines! At the moment
# only pure directions are rendered
# (source http://www.minecraftwiki.net/wiki/Data_values#Vines)
@material(blockid=106, data=range(8), transparent=True)
def vines(blockid, data, north):
    # north rotation
    if north == 'upper-left':
        if data == 1: data = 2
        elif data == 4: data = 8
        elif data == 8: data = 1
        elif data == 2: data = 4
    elif north == 'upper-right':
        if data == 1: data = 4
        elif data == 4: data = 1
        elif data == 8: data = 2
        elif data == 2: data = 8
    elif north == 'lower-right':
        if data == 1: data = 8
        elif data == 4: data = 2
        elif data == 8: data = 4
        elif data == 2: data = 1

    # texture generation
    img = Image.new("RGBA", (24,24), bgcolor)
    raw_texture = terrain_images[143]

    if data == 2:   # south
        tex = transform_image_side(raw_texture)
        composite.alpha_over(img, tex, (0,6), tex)

    if data == 1:	# east
        tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(img, tex, (12,6), tex)

    if data == 4:	# west
        tex = transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(img, tex, (0,0), tex)

    if data == 8:	# north
        tex = transform_image_side(raw_texture)
        composite.alpha_over(img, tex, (12,0), tex)

    return img

# fence gates
@material(blockid=107, data=range(8), transparent=True, nospawn=True)
def fence_gate(blockid, data, north):

    # north rotation
    opened = False
    if data & 0x4:
        data = data & 0x3
        opened = True
    if north == 'upper-left':
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif north == 'upper-right':
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif north == 'lower-right':
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2
    if opened:
        data = data | 0x4

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
    mirror_gate_side = transform_image_side(gate_side.transpose(Image.FLIP_LEFT_RIGHT))
    gate_side = transform_image_side(gate_side)
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
    
    return img

# mycelium
block(blockid=110, top_index=78, side_index=77)

# lilypad
# TODO the data-block orientation relation is not clear
@material(blockid=111, data=range(4), transparent=True)
def lilypad(blockid, data, north):
    if north == 'upper-left':
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 1
        elif data == 3: data = 0
    elif north == 'upper-right':
        if data == 0: data = 1
        elif data == 1: data = 0
        elif data == 2: data = 3
        elif data == 3: data = 2
    elif north == 'lower-right':
        if data == 0: data = 3
        elif data == 1: data = 2
        elif data == 2: data = 0
        elif data == 3: data = 1

    t = terrain_images[76] # NOTE: using same data as stairs, no 
                           # info in minepedia at the moment.
    if data == 0: # pointing south
        img = build_full_block(None, None, None, None, None, t)
    elif data == 1: # pointing north
        img = build_full_block(None, None, None, None, None, t.rotate(180))
    elif data == 2: # pointing west
        img = build_full_block(None, None, None, None, None, t.rotate(270))
    elif data == 3: # pointing east
        img = build_full_block(None, None, None, None, None, t.rotate(90))
    
    return img

# nether brick
block(blockid=112, top_index=224, side_index=224)

# nether wart
@material(blockid=115, data=range(4), transparent=True)
def nether_wart(blockid, data):
    if data == 0: # just come up
        t = terrain_images[226]
    elif data in (1, 2):
        t = terrain_images[227]
    else: # fully grown
        t = terrain_images[228]
    
    # use the same technic as tall grass
    img = build_billboard(t)

    return img

# enchantment table
# TODO there's no book at the moment
@material(blockid=116, transparent=True)
def enchantment_table(blockid, data):
    # no book at the moment
    top = terrain_images[166]
    side = terrain_images[182]
    img = build_full_block((top, 4), None, None, side, side)

    return img

# brewing stand
# TODO this is a place holder, is a 2d image pasted
@material(blockid=117, data=range(5), transparent=True)
def brewing_stand(blockid, data, north):
    t = terrain_images[157]
    img = build_billboard(t)
    return img

# cauldron
@material(blockid=118, data=range(4), transparent=True)
def cauldron(blockid, data):
    side = terrain_images[154]
    top = terrain_images[138]
    bottom = terrain_images[139]
    water = transform_image_top(_load_image("water.png"))
    if data == 0: # empty
        img = build_full_block(top, side, side, side, side)
    if data == 1: # 1/3 filled
        img = build_full_block(None , side, side, None, None)
        composite.alpha_over(img, water, (0,8), water)
        img2 = build_full_block(top , None, None, side, side)
        composite.alpha_over(img, img2, (0,0), img2)
    if data == 2: # 2/3 filled
        img = build_full_block(None , side, side, None, None)
        composite.alpha_over(img, water, (0,4), water)
        img2 = build_full_block(top , None, None, side, side)
        composite.alpha_over(img, img2, (0,0), img2)
    if data == 3: # 3/3 filled
        img = build_full_block(None , side, side, None, None)
        composite.alpha_over(img, water, (0,0), water)
        img2 = build_full_block(top , None, None, side, side)
        composite.alpha_over(img, img2, (0,0), img2)

    return img

# end portal
@material(blockid=119, transparent=True)
def end_portal(blockid, data):
    img = Image.new("RGBA", (24,24), bgcolor)
    # generate a black texure with white, blue and grey dots resembling stars
    t = Image.new("RGBA", (16,16), (0,0,0,255))
    for color in [(155,155,155,255), (100,255,100,255), (255,255,255,255)]:
        for i in range(6):
            x = randint(0,15)
            y = randint(0,15)
            t.putpixel((x,y),color)

    t = transform_image_top(t)
    composite.alpha_over(img, t, (0,0), t)

    return img

# end portal frame
@material(blockid=120, data=range(5), transparent=True)
def end_porta_frame(blockid, data):
    # The bottom 2 bits are oritation info but seems there is no
    # graphical difference between orientations
    top = terrain_images[158]
    eye_t = terrain_images[174]
    side = terrain_images[159]
    img = build_full_block((top, 4), None, None, side, side)
    if data & 0x4 == 0x4: # ender eye on it
        # generate the eye
        eye_t = terrain_images[174].copy()
        eye_t_s = terrain_images[174].copy()
        # cut out from the texture the side and the top of the eye
        ImageDraw.Draw(eye_t).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(eye_t_s).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        # trnasform images and paste
        eye = transform_image_top(eye_t)
        eye_s = transform_image_side(eye_t_s)
        eye_os = eye_s.transpose(Image.FLIP_LEFT_RIGHT)
        composite.alpha_over(img, eye_s, (5,5), eye_s)
        composite.alpha_over(img, eye_os, (9,5), eye_os)
        composite.alpha_over(img, eye, (0,0), eye)

    return img

# end stone
block(blockid=121, top_index=175)
