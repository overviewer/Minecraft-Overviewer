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
from c_overviewer import alpha_over

##
## Textures object
##
class Textures(object):
    """An object that generates a set of block sprites to use while
    rendering. It accepts a background color, north direction, and
    local textures path.
    """
    def __init__(self, texturepath=None, bgcolor=(26, 26, 26, 0), northdirection=0):
        self.bgcolor = bgcolor
        self.rotation = northdirection
        self.find_file_local_path = texturepath
        
        # not yet configurable
        self.texture_size = 24
        self.texture_dimensions = (self.texture_size, self.texture_size)
        
        # this is set in in generate()
        self.generated = False
    
    ##
    ## pickle support
    ##
    
    def __getstate__(self):
        # we must get rid of the huge image lists, and other images
        attributes = self.__dict__.copy()
        for attr in ['terrain_images', 'blockmap', 'biome_grass_texture', 'watertexture', 'lavatexture', 'firetexture', 'portaltexture', 'lightcolor', 'grasscolor', 'foliagecolor', 'watercolor']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        return attributes
    def __setstate__(self, attrs):
        # regenerate textures, if needed
        for attr, val in attrs.iteritems():
            setattr(self, attr, val)
        if self.generated:
            self.generate()
    
    ##
    ## The big one: generate()
    ##
    
    def generate(self):
        # maps terrainids to 16x16 images
        self.terrain_images = self._split_terrain(self.load_image("terrain.png"))
        
        # generate biome grass mask
        self.biome_grass_texture = self.build_block(self.terrain_images[0], self.terrain_images[38])
        
        # generate the blocks
        global blockmap_generators
        global known_blocks, used_datas
        self.blockmap = [None] * max_blockid * max_data
        
        for (blockid, data), texgen in blockmap_generators.iteritems():
            tex = texgen(self, blockid, data)
            self.blockmap[blockid * max_data + data] = self.generate_texture_tuple(tex)
        
        if self.texture_size != 24:
            # rescale biome grass
            self.biome_grass_texture = self.biome_grass_texture.resize(self.texture_dimensions, Image.ANTIALIAS)
            
            # rescale the rest
            for i, tex in enumerate(blockmap):
                if tex is None:
                    continue
                block = tex[0]
                scaled_block = block.resize(self.texture_dimensions, Image.ANTIALIAS)
                blockmap[i] = self.generate_texture_tuple(scaled_block)
        
        self.generated = True
    
    ##
    ## Helpers for opening textures
    ##
    
    def find_file(self, filename, mode="rb", verbose=False):
        """Searches for the given file and returns an open handle to it.
        This searches the following locations in this order:
        
        * the textures_path given in the initializer
        this can be either a directory or a zip file (texture pack)
        * The program dir (same dir as overviewer.py)
        * On Darwin, in /Applications/Minecraft
        * Inside minecraft.jar, which is looked for at these locations
        
            * On Windows, at %APPDATA%/.minecraft/bin/minecraft.jar
            * On Darwin, at
                $HOME/Library/Application Support/minecraft/bin/minecraft.jar
            * at $HOME/.minecraft/bin/minecraft.jar

        * The overviewer_core/data/textures dir
        
        In all of these, files are searched for in '.', 'anim', 'misc/', and
        'environment/'.
        
        """

        # a list of subdirectories to search for a given file,
        # after the obvious '.'
        search_dirs = ['anim', 'misc', 'environment', 'item']
        search_zip_paths = [filename,] + [d + '/' + filename for d in search_dirs]
        def search_dir(base):
            """Search the given base dir for filename, in search_dirs."""
            for path in [os.path.join(base, d, filename) for d in ['',] + search_dirs]:
                if os.path.isfile(path):
                    return path
            return None

        if self.find_file_local_path:
            if os.path.isdir(self.find_file_local_path):
                path = search_dir(self.find_file_local_path)
                if path:
                    if verbose: logging.info("Found %s in '%s'", filename, path)
                    return open(path, mode)
            elif os.path.isfile(self.find_file_local_path):
                try:
                    pack = zipfile.ZipFile(self.find_file_local_path)
                    for packfilename in search_zip_paths:
                        try:
                            pack.getinfo(packfilename)
                            if verbose: logging.info("Found %s in '%s'", packfilename, nery.find_file_local_path)
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
        if self.find_file_local_path:
            jarpaths.append(os.path.join(self.find_file_local_path, "minecraft.jar"))

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

        raise IOError("Could not find the file `{0}'. Try specifying the 'texturepath' option in your config file. Set it to the directory where I can find {0}. Also see <http://docs.overviewer.org/en/latest/running/#installing-the-textures>".format(filename))

    def load_image(self, filename):
        """Returns an image object"""
        fileobj = self.find_file(filename)
        buffer = StringIO(fileobj.read())
        return Image.open(buffer).convert("RGBA")

    def load_water(self):
        """Special-case function for loading water, handles
        MCPatcher-compliant custom animated water."""
        watertexture = getattr(self, "watertexture", None)
        if watertexture:
            return watertexture
        try:
            # try the MCPatcher case first
            watertexture = self.load_image("custom_water_still.png")
            watertexture = watertexture.crop((0, 0, watertexture.size[0], watertexture.size[0]))
        except IOError:
            watertexture = self.load_image("water.png")
        self.watertexture = watertexture
        return watertexture

    def load_lava(self):
        """Special-case function for loading lava, handles
        MCPatcher-compliant custom animated lava."""
        lavatexture = getattr(self, "lavatexture", None)
        if lavatexture:
            return lavatexture
        try:
            # try the MCPatcher lava first, in case it's present
            lavatexture = self.load_image("custom_lava_still.png")
            lavatexture = lavatexture.crop((0, 0, lavatexture.size[0], lavatexture.size[0]))
        except IOError:
            lavatexture = self.load_image("lava.png")
        self.lavatexture = lavatexture
        return lavatexture
    
    def load_fire(self):
        """Special-case function for loading fire, handles
        MCPatcher-compliant custom animated fire."""
        firetexture = getattr(self, "firetexture", None)
        if firetexture:
            return firetexture
        try:
            # try the MCPatcher case first
            firetextureNS = self.load_image("custom_fire_n_s.png")
            firetextureNS = firetextureNS.crop((0, 0, firetextureNS.size[0], firetextureNS.size[0]))
            firetextureEW = self.load_image("custom_fire_e_w.png")
            firetextureEW = firetextureEW.crop((0, 0, firetextureEW.size[0], firetextureEW.size[0]))
            firetexture = (firetextureNS,firetextureEW)
        except IOError:
            fire = self.load_image("fire.png")
            firetexture = (fire, fire)
        self.firetexture = firetexture
        return firetexture
    
    def load_portal(self):
        """Special-case function for loading portal, handles
        MCPatcher-compliant custom animated portal."""
        portaltexture = getattr(self, "portaltexture", None)
        if portaltexture:
            return portaltexture
        try:
            # try the MCPatcher case first
            portaltexture = self.load_image("custom_portal.png")
            portaltexture = portaltexture.crop((0, 0, portaltexture.size[0], portaltexture.size[1]))
        except IOError:
            portaltexture = self.load_image("portal.png")
        self.portaltexture = portaltexture
        return portaltexture
    
    def load_light_color(self):
        """Helper function to load the light color texture."""
        if hasattr(self, "lightcolor"):
            return self.lightcolor
        try:
            lightcolor = list(self.load_image("light_normal.png").getdata())
        except Exception:
            logging.warning("Light color image could not be found.")
            lightcolor = None
        self.lightcolor = lightcolor
        return lightcolor
    
    def load_grass_color(self):
        """Helper function to load the grass color texture."""
        if not hasattr(self, "grasscolor"):
            self.grasscolor = list(self.load_image("grasscolor.png").getdata())
        return self.grasscolor

    def load_foliage_color(self):
        """Helper function to load the foliage color texture."""
        if not hasattr(self, "foliagecolor"):
            self.foliagecolor = list(self.load_image("foliagecolor.png").getdata())
        return self.foliagecolor

    def load_water_color(self):
        """Helper function to load the water color texture."""
        if not hasattr(self, "watercolor"):
            self.watercolor = list(self.load_image("watercolor.png").getdata())
        return self.watercolor

    def _split_terrain(self, terrain):
        """Builds and returns a length 256 array of each 16x16 chunk
        of texture.
        """
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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


    @staticmethod
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


    def build_block(self, top, side):
        """From a top texture and a side texture, build a block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        original_texture = top.copy()
        top = self.transform_image_top(top)

        if not side:
            alpha_over(img, top, (0,0), top)
            return img

        side = self.transform_image_side(side)
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

        alpha_over(img, top, (0,0), top)
        alpha_over(img, side, (0,6), side)
        alpha_over(img, otherside, (12,6), otherside)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))

        return img

    def build_full_block(self, top, side1, side2, side3, side4, bottom=None):
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
        you use an increment of 8, it will draw a half-block.

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

        img = Image.new("RGBA", (24,24), self.bgcolor)

        # first back sides
        if side1 != None :
            side1 = self.transform_image_side(side1)
            side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side.
            sidealpha = side1.split()[3]
            side1 = ImageEnhance.Brightness(side1).enhance(0.9)
            side1.putalpha(sidealpha)        

            alpha_over(img, side1, (0,0), side1)


        if side2 != None :
            side2 = self.transform_image_side(side2)

            # Darken this side.
            sidealpha2 = side2.split()[3]
            side2 = ImageEnhance.Brightness(side2).enhance(0.8)
            side2.putalpha(sidealpha2)

            alpha_over(img, side2, (12,0), side2)

        if bottom != None :
            bottom = self.transform_image_top(bottom)
            alpha_over(img, bottom, (0,12), bottom)

        # front sides
        if side3 != None :
            side3 = self.transform_image_side(side3)

            # Darken this side
            sidealpha = side3.split()[3]
            side3 = ImageEnhance.Brightness(side3).enhance(0.9)
            side3.putalpha(sidealpha)

            alpha_over(img, side3, (0,6), side3)

        if side4 != None :
            side4 = self.transform_image_side(side4)
            side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side
            sidealpha = side4.split()[3]
            side4 = ImageEnhance.Brightness(side4).enhance(0.8)
            side4.putalpha(sidealpha)

            alpha_over(img, side4, (12,6), side4)

        if top != None :
            top = self.transform_image_top(top)
            alpha_over(img, top, (0, increment), top)

        return img

    def build_sprite(self, side):
        """From a side texture, create a sprite-like texture such as those used
        for spiderwebs or flowers."""
        img = Image.new("RGBA", (24,24), self.bgcolor)

        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        alpha_over(img, side, (6,3), side)
        alpha_over(img, otherside, (6,3), otherside)
        return img

    def build_billboard(self, tex):
        """From a texture, create a billboard-like texture such as
        those used for tall grass or melon stems.
        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        front = tex.resize((14, 11), Image.ANTIALIAS)
        alpha_over(img, front, (5,9))
        return img

    def generate_opaque_mask(self, img):
        """ Takes the alpha channel of the image and generates a mask
        (used for lighting the block) that deprecates values of alpha
        smallers than 50, and sets every other value to 255. """

        alpha = img.split()[3]
        return alpha.point(lambda a: int(min(a, 25.5) * 10))

    def tint_texture(self, im, c):
        # apparently converting to grayscale drops the alpha channel?
        i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
        i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
        return i

    def generate_texture_tuple(self, img):
        """ This takes an image and returns the needed tuple for the
        blockmap array."""
        if img is None:
            return None
        return (img, self.generate_opaque_mask(img))

##
## The other big one: @material and associated framework
##

# global variables to collate information in @material decorators
blockmap_generators = {}

known_blocks = set()
used_datas = set()
max_blockid = 0
max_data = 0

transparent_blocks = set()
solid_blocks = set()
fluid_blocks = set()
nospawn_blocks = set()
nodata_blocks = set()

# the material registration decorator
def material(blockid=[], data=[0], **kwargs):
    # mapping from property name to the set to store them in
    properties = {"transparent" : transparent_blocks, "solid" : solid_blocks, "fluid" : fluid_blocks, "nospawn" : nospawn_blocks, "nodata" : nodata_blocks}
    
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
        global max_data, max_blockid

        # create a wrapper function with a known signature
        @functools.wraps(func)
        def func_wrapper(texobj, blockid, data):
            return func(texobj, blockid, data)
        
        used_datas.update(data)
        if max(data) >= max_data:
            max_data = max(data) + 1
        
        for block in blockid:
            # set the property sets appropriately
            known_blocks.update([block])
            if block >= max_blockid:
                max_blockid = block + 1
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

# shortcut function for pure blocks, default to solid, nodata
def block(blockid=[], top_index=None, side_index=None, **kwargs):
    new_kwargs = {'solid' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if top_index is None:
        raise ValueError("top_index was not provided")
    
    if side_index is None:
        side_index = top_index
    
    @material(blockid=blockid, **new_kwargs)
    def inner_block(self, unused_id, unused_data):
        return self.build_block(self.terrain_images[top_index], self.terrain_images[side_index])
    return inner_block

# shortcut function for sprite blocks, defaults to transparent, nodata
def sprite(blockid=[], index=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if index is None:
        raise ValueError("index was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_sprite(self, unused_id, unused_data):
        return self.build_sprite(self.terrain_images[index])
    return inner_sprite

# shortcut function for billboard blocks, defaults to transparent, nodata
def billboard(blockid=[], index=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if index is None:
        raise ValueError("index was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_billboard(self, unused_id, unused_data):
        return self.build_billboard(self.terrain_images[index])
    return inner_billboard

##
## and finally: actual texture definitions
##

# stone
block(blockid=1, top_index=1)

@material(blockid=2, data=range(11)+[0x10,], solid=True)
def grass(self, blockid, data):
    # 0x10 bit means SNOW
    side_img = self.terrain_images[3]
    if data & 0x10:
        side_img = self.terrain_images[68]
    img = self.build_block(self.terrain_images[0], side_img)
    if not data & 0x10:
        alpha_over(img, self.biome_grass_texture, (0, 0), self.biome_grass_texture)
    return img

# dirt
block(blockid=3, top_index=2)
# cobblestone
block(blockid=4, top_index=16)

# wooden planks
@material(blockid=5, data=range(4), solid=True)
def wooden_planks(self, blockid, data):
    if data == 0: # normal
        return self.build_block(self.terrain_images[4],self.terrain_images[4])
    if data == 1: # pine
        return self.build_block(self.terrain_images[198],self.terrain_images[198])
    if data == 2: # birch
        return self.build_block(self.terrain_images[214],self.terrain_images[214])
    if data == 3: # jungle wood
        return self.build_block(self.terrain_images[199],self.terrain_images[199])

@material(blockid=6, data=range(16), transparent=True)
def saplings(self, blockid, data):
    # usual saplings
    tex = self.terrain_images[15]
    
    if data & 0x3 == 1: # spruce sapling
        tex = self.terrain_images[63]
    elif data & 0x3 == 2: # birch sapling
        tex = self.terrain_images[79]
    elif data & 0x3 == 3: # jungle sapling
        tex = self.terrain_images[30]
    return self.build_sprite(tex)

# bedrock
block(blockid=7, top_index=17)

@material(blockid=8, data=range(16), fluid=True, transparent=True, nospawn=True)
def water(self, blockid, data):
    watertex = self.load_water()
    return self.build_block(watertex, watertex)

# other water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[9, 20, 79], data=range(32), fluid=(9,), transparent=True, nospawn=True, solid=(79, 20))
def no_inner_surfaces(self, blockid, data):
    if blockid == 9:
        texture = self.load_water()
    elif blockid == 20:
        texture = self.terrain_images[49]
    else:
        texture = self.terrain_images[67]
        
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
    
    img = self.build_full_block(top,None,None,side3,side4)
    return img

@material(blockid=[10, 11], data=range(16), fluid=True, transparent=False, nospawn=True)
def lava(self, blockid, data):
    lavatex = self.load_lava()
    return self.build_block(lavatex, lavatex)

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

@material(blockid=17, data=range(12), solid=True)
def wood(self, blockid, data):
    # extract orientation and wood type frorm data bits
    wood_type = data & 3
    wood_orientation = data & 12
    if self.rotation == 1:
        if wood_orientation == 4: wood_orientation = 8
        elif wood_orientation == 8: wood_orientation = 4
    elif self.rotation == 3:
        if wood_orientation == 4: wood_orientation = 8
        elif wood_orientation == 8: wood_orientation = 4

    # choose textures
    top = self.terrain_images[21]
    if wood_type == 0: # normal
        side = self.terrain_images[20]
    if wood_type == 1: # birch
        side = self.terrain_images[116]
    if wood_type == 2: # pine
        side = self.terrain_images[117]
    if wood_type == 3: # jungle wood
        side = self.terrain_images[153]

    # choose orientation and paste textures
    if wood_orientation == 0:
        return self.build_block(top, side)
    elif wood_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif wood_orientation == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(270), top)

@material(blockid=18, data=range(16), transparent=True, solid=True)
def leaves(self, blockid, data):
    # mask out the bits 4 and 8
    # they are used for player placed and check-for-decay blocks
    data = data & 0x3
    t = self.terrain_images[52]
    if data == 1:
        # pine!
        t = self.terrain_images[132]
    elif data == 3:
        # jungle tree
        t = self.terrain_images[196]
    return self.build_block(t, t)

# sponge
block(blockid=19, top_index=48)
# lapis lazuli ore
block(blockid=21, top_index=160)
# lapis lazuli block
block(blockid=22, top_index=144)

# dispensers, furnaces, and burning furnaces
@material(blockid=[23, 61, 62], data=range(6), solid=True)
def furnaces(self, blockid, data):
    # first, do the rotation if needed
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2
    
    top = self.terrain_images[62]
    side = self.terrain_images[45]
    
    if blockid == 61:
        front = self.terrain_images[44]
    elif blockid == 62:
        front = self.terrain_images[61]
    elif blockid == 23:
        front = self.terrain_images[46]
    
    if data == 3: # pointing west
        return self.build_full_block(top, None, None, side, front)
    elif data == 4: # pointing north
        return self.build_full_block(top, None, None, front, side)
    else: # in any other direction the front can't be seen
        return self.build_full_block(top, None, None, side, side)

# sandstone
@material(blockid=24, data=range(3), solid=True)
def sandstone(self, blockid, data):
    top = self.terrain_images[176]
    if data == 0: # normal
        return self.build_block(top, self.terrain_images[192])
    if data == 1: # hieroglyphic
        return self.build_block(top, self.terrain_images[229])
    if data == 2: # soft
        return self.build_block(top, self.terrain_images[230])

# note block
block(blockid=25, top_index=74)

@material(blockid=26, data=range(12), transparent=True, nospawn=True)
def bed(self, blockid, data):
    # first get rotation done
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    
    increment = 8
    left_face = None
    right_face = None
    if data & 0x8 == 0x8: # head of the bed
        top = self.terrain_images[135]
        if data & 0x00 == 0x00: # head pointing to West
            top = top.copy().rotate(270)
            left_face = self.terrain_images[151]
            right_face = self.terrain_images[152]
        if data & 0x01 == 0x01: # ... North
            top = top.rotate(270)
            left_face = self.terrain_images[152]
            right_face = self.terrain_images[151]
        if data & 0x02 == 0x02: # East
            top = top.rotate(180)
            left_face = self.terrain_images[151].transpose(Image.FLIP_LEFT_RIGHT)
            right_face = None
        if data & 0x03 == 0x03: # South
            right_face = None
            right_face = self.terrain_images[151].transpose(Image.FLIP_LEFT_RIGHT)
    
    else: # foot of the bed
        top = self.terrain_images[134]
        if data & 0x00 == 0x00: # head pointing to West
            top = top.rotate(270)
            left_face = self.terrain_images[150]
            right_face = None
        if data & 0x01 == 0x01: # ... North
            top = top.rotate(270)
            left_face = None
            right_face = self.terrain_images[150]
        if data & 0x02 == 0x02: # East
            top = top.rotate(180)
            left_face = self.terrain_images[150].transpose(Image.FLIP_LEFT_RIGHT)
            right_face = self.terrain_images[149].transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x03 == 0x03: # South
            left_face = self.terrain_images[149]
            right_face = self.terrain_images[150].transpose(Image.FLIP_LEFT_RIGHT)
    
    top = (top, increment)
    return self.build_full_block(top, None, None, left_face, right_face)

# powered, detector, and normal rails
@material(blockid=[27, 28, 66], data=range(14), transparent=True)
def rails(self, blockid, data):
    # first, do rotation
    # Masked to not clobber powered rail on/off info
    # Ascending and flat straight
    if self.rotation == 1:
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    if blockid == 66: # normal minetrack only
        #Corners
        if self.rotation == 1:
            if data == 6: data = 7
            elif data == 7: data = 8
            elif data == 8: data = 6
            elif data == 9: data = 9
        elif self.rotation == 2:
            if data == 6: data = 8
            elif data == 7: data = 9
            elif data == 8: data = 6
            elif data == 9: data = 7
        elif self.rotation == 3:
            if data == 6: data = 9
            elif data == 7: data = 6
            elif data == 8: data = 8
            elif data == 9: data = 7
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    if blockid == 27: # powered rail
        if data & 0x8 == 0: # unpowered
            raw_straight = self.terrain_images[163]
            raw_corner = self.terrain_images[112]    # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = self.terrain_images[179]
            raw_corner = self.terrain_images[112]    # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7
            
    elif blockid == 28: # detector rail
        raw_straight = self.terrain_images[195]
        raw_corner = self.terrain_images[112]    # leave corners for code simplicity
        
    elif blockid == 66: # normal rail
        raw_straight = self.terrain_images[128]
        raw_corner = self.terrain_images[112]
        
    ## use transform_image to scale and shear
    if data == 0:
        track = self.transform_image_top(raw_straight)
        alpha_over(img, track, (0,12), track)
    elif data == 6:
        track = self.transform_image_top(raw_corner)
        alpha_over(img, track, (0,12), track)
    elif data == 7:
        track = self.transform_image_top(raw_corner.rotate(270))
        alpha_over(img, track, (0,12), track)
    elif data == 8:
        # flip
        track = self.transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM).rotate(90))
        alpha_over(img, track, (0,12), track)
    elif data == 9:
        track = self.transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM))
        alpha_over(img, track, (0,12), track)
    elif data == 1:
        track = self.transform_image_top(raw_straight.rotate(90))
        alpha_over(img, track, (0,12), track)
        
    #slopes
    elif data == 2: # slope going up in +x direction
        track = self.transform_image_slope(raw_straight)
        track = track.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, track, (2,0), track)
        # the 2 pixels move is needed to fit with the adjacent tracks
        
    elif data == 3: # slope going up in -x direction
        # tracks are sprites, in this case we are seeing the "side" of 
        # the sprite, so draw a line to make it beautiful.
        ImageDraw.Draw(img).line([(11,11),(23,17)],fill=(164,164,164))
        # grey from track texture (exterior grey).
        # the track doesn't start from image corners, be carefull drawing the line!
    elif data == 4: # slope going up in -y direction
        track = self.transform_image_slope(raw_straight)
        alpha_over(img, track, (0,0), track)
        
    elif data == 5: # slope going up in +y direction
        # same as "data == 3"
        ImageDraw.Draw(img).line([(1,17),(12,11)],fill=(164,164,164))
        
    return img

# sticky and normal piston body
@material(blockid=[29, 33], data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, solid=True, nospawn=True)
def piston(self, blockid, data):
    # first, rotation
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    
    if blockid == 29: # sticky
        piston_t = self.terrain_images[106].copy()
    else: # normal
        piston_t = self.terrain_images[107].copy()
        
    # other textures
    side_t = self.terrain_images[108].copy()
    back_t = self.terrain_images[109].copy()
    interior_t = self.terrain_images[110].copy()
    
    if data & 0x08 == 0x08: # pushed out, non full blocks, tricky stuff
        # remove piston texture from piston body
        ImageDraw.Draw(side_t).rectangle((0, 0,16,3),outline=(0,0,0,0),fill=(0,0,0,0))
        
        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = self.build_full_block(back_t ,None ,None ,side_t, side_t)
            
        elif data & 0x07 == 0x1: # up
            img = self.build_full_block((interior_t, 4) ,None ,None ,side_t, side_t)
            
        elif data & 0x07 == 0x2: # east
            img = self.build_full_block(side_t , None, None ,side_t.rotate(90), back_t)
            
        elif data & 0x07 == 0x3: # west
            img = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), None)
            temp = self.transform_image_side(interior_t)
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            alpha_over(img, temp, (9,5), temp)
            
        elif data & 0x07 == 0x4: # north
            img = self.build_full_block(side_t.rotate(90) ,None ,None , None, side_t.rotate(270))
            temp = self.transform_image_side(interior_t)
            alpha_over(img, temp, (3,5), temp)
            
        elif data & 0x07 == 0x5: # south
            img = self.build_full_block(side_t.rotate(270) ,None , None ,back_t, side_t.rotate(90))

    else: # pushed in, normal full blocks, easy stuff
        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = self.build_full_block(back_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x1: # up
            img = self.build_full_block(piston_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x2: # east 
            img = self.build_full_block(side_t ,None ,None ,side_t.rotate(90), back_t)
        elif data & 0x07 == 0x3: # west
            img = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        elif data & 0x07 == 0x4: # north
            img = self.build_full_block(side_t.rotate(90) ,None ,None ,piston_t, side_t.rotate(270))
        elif data & 0x07 == 0x5: # south
            img = self.build_full_block(side_t.rotate(270) ,None ,None ,back_t, side_t.rotate(90))
            
    return img

# sticky and normal piston shaft
@material(blockid=34, data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, nospawn=True)
def piston_extension(self, blockid, data):
    # first, rotation
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    
    if (data & 0x8) == 0x8: # sticky
        piston_t = self.terrain_images[106].copy()
    else: # normal
        piston_t = self.terrain_images[107].copy()
    
    # other textures
    side_t = self.terrain_images[108].copy()
    back_t = self.terrain_images[107].copy()
    # crop piston body
    ImageDraw.Draw(side_t).rectangle((0, 4,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
    
    # generate the horizontal piston extension stick
    h_stick = Image.new("RGBA", (24,24), self.bgcolor)
    temp = self.transform_image_side(side_t)
    alpha_over(h_stick, temp, (1,7), temp)
    temp = self.transform_image_top(side_t.rotate(90))
    alpha_over(h_stick, temp, (1,1), temp)
    # Darken it
    sidealpha = h_stick.split()[3]
    h_stick = ImageEnhance.Brightness(h_stick).enhance(0.85)
    h_stick.putalpha(sidealpha)
    
    # generate the vertical piston extension stick
    v_stick = Image.new("RGBA", (24,24), self.bgcolor)
    temp = self.transform_image_side(side_t.rotate(90))
    alpha_over(v_stick, temp, (12,6), temp)
    temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over(v_stick, temp, (1,6), temp)
    # Darken it
    sidealpha = v_stick.split()[3]
    v_stick = ImageEnhance.Brightness(v_stick).enhance(0.85)
    v_stick.putalpha(sidealpha)
    
    # Piston orientation is stored in the 3 first bits
    if data & 0x07 == 0x0: # down
        side_t = side_t.rotate(180)
        img = self.build_full_block((back_t, 12) ,None ,None ,side_t, side_t)
        alpha_over(img, v_stick, (0,-3), v_stick)
    elif data & 0x07 == 0x1: # up
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(piston_t ,None ,None ,side_t, side_t)
        alpha_over(img, v_stick, (0,4), v_stick)
        alpha_over(img, img2, (0,0), img2)
    elif data & 0x07 == 0x2: # east 
        img = self.build_full_block(side_t ,None ,None ,side_t.rotate(90), None)
        temp = self.transform_image_side(back_t).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, temp, (2,2), temp)
        alpha_over(img, h_stick, (6,3), h_stick)
    elif data & 0x07 == 0x3: # west
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        alpha_over(img, h_stick, (0,0), h_stick)
        alpha_over(img, img2, (0,0), img2)            
    elif data & 0x07 == 0x4: # north
        img = self.build_full_block(side_t.rotate(90) ,None ,None , piston_t, side_t.rotate(270))
        alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (0,0), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
    elif data & 0x07 == 0x5: # south
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(side_t.rotate(270) ,None ,None ,None, side_t.rotate(90))
        temp = self.transform_image_side(back_t)
        alpha_over(img2, temp, (10,2), temp)
        alpha_over(img, img2, (0,0), img2)
        alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (-3,2), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
        
    return img

# cobweb
sprite(blockid=30, index=11, nospawn=True)

@material(blockid=31, data=range(3), transparent=True)
def tall_grass(self, blockid, data):
    if data == 0: # dead shrub
        texture = self.terrain_images[55]
    elif data == 1: # tall grass
        texture = self.terrain_images[39]
    elif data == 2: # fern
        texture = self.terrain_images[56]
    
    return self.build_billboard(texture)

# dead bush
billboard(blockid=32, index=55)

@material(blockid=35, data=range(16), solid=True)
def wool(self, blockid, data):
    if data == 0: # white
        texture = self.terrain_images[64]
    elif data == 1: # orange
        texture = self.terrain_images[210]
    elif data == 2: # magenta
        texture = self.terrain_images[194]
    elif data == 3: # light blue
        texture = self.terrain_images[178]
    elif data == 4: # yellow
        texture = self.terrain_images[162]
    elif data == 5: # light green
        texture = self.terrain_images[146]
    elif data == 6: # pink
        texture = self.terrain_images[130]
    elif data == 7: # grey
        texture = self.terrain_images[114]
    elif data == 8: # light grey
        texture = self.terrain_images[225]
    elif data == 9: # cyan
        texture = self.terrain_images[209]
    elif data == 10: # purple
        texture = self.terrain_images[193]
    elif data == 11: # blue
        texture = self.terrain_images[177]
    elif data == 12: # brown
        texture = self.terrain_images[161]
    elif data == 13: # dark green
        texture = self.terrain_images[145]
    elif data == 14: # red
        texture = self.terrain_images[129]
    elif data == 15: # black
        texture = self.terrain_images[113]
    
    return self.build_block(texture, texture)

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
# these wooden slabs are unobtainable without cheating, they are still
# here because lots of pre-1.3 worlds use this blocks
@material(blockid=[43, 44], data=range(16), transparent=(44,), solid=True)
def slabs(self, blockid, data):
    texture = data & 7
    if texture== 0: # stone slab
        top = self.terrain_images[6]
        side = self.terrain_images[5]
    elif texture== 1: # stone slab
        top = self.terrain_images[176]
        side = self.terrain_images[192]
    elif texture== 2: # wooden slab
        top = side = self.terrain_images[4]
    elif texture== 3: # cobblestone slab
        top = side = self.terrain_images[16]
    elif texture== 4: # brick
        top = side = self.terrain_images[7]
    elif texture== 5: # stone brick
        top = side = self.terrain_images[54]
    else:
        return None
    
    if blockid == 43: # double slab
        return self.build_block(top, side)
    
    # cut the side texture in half
    mask = side.crop((0,8,16,16))
    side = Image.new(side.mode, side.size, self.bgcolor)
    alpha_over(side, mask,(0,0,16,8), mask)
    
    # plain slab
    top = self.transform_image_top(top)
    side = self.transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)
    
    # upside down slab
    delta = 0
    if data & 8 == 8:
        delta = 6
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, side, (0,12 - delta), side)
    alpha_over(img, otherside, (12,12 - delta), otherside)
    alpha_over(img, top, (0,6 - delta), top)
    
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
def torches(self, blockid, data):
    # first, rotations
    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2
    
    # choose the proper texture
    if blockid == 50: # torch
        small = self.terrain_images[80]
    elif blockid == 75: # off redstone torch
        small = self.terrain_images[115]
    else: # on redstone torch
        small = self.terrain_images[99]
        
    # compose a torch bigger than the normal
    # (better for doing transformations)
    torch = Image.new("RGBA", (16,16), self.bgcolor)
    alpha_over(torch,small,(-4,-3))
    alpha_over(torch,small,(-5,-2))
    alpha_over(torch,small,(-3,-2))
    
    # angle of inclination of the texture
    rotation = 15
    
    if data == 1: # pointing south
        torch = torch.rotate(-rotation, Image.NEAREST) # nearest filter is more nitid.
        img = self.build_full_block(None, None, None, torch, None, None)
        
    elif data == 2: # pointing north
        torch = torch.rotate(rotation, Image.NEAREST)
        img = self.build_full_block(None, None, torch, None, None, None)
        
    elif data == 3: # pointing west
        torch = torch.rotate(rotation, Image.NEAREST)
        img = self.build_full_block(None, torch, None, None, None, None)
        
    elif data == 4: # pointing east
        torch = torch.rotate(-rotation, Image.NEAREST)
        img = self.build_full_block(None, None, None, None, torch, None)
        
    elif data == 5: # standing on the floor
        # compose a "3d torch".
        img = Image.new("RGBA", (24,24), self.bgcolor)
        
        small_crop = small.crop((2,2,14,14))
        slice = small_crop.copy()
        ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
        
        alpha_over(img, slice, (7,5))
        alpha_over(img, small_crop, (6,6))
        alpha_over(img, small_crop, (7,6))
        alpha_over(img, slice, (7,7))
        
    return img

# fire
@material(blockid=51, data=range(16), transparent=True)
def fire(self, blockid, data):
    firetextures = self.load_fire()
    side1 = self.transform_image_side(firetextures[0])
    side2 = self.transform_image_side(firetextures[1]).transpose(Image.FLIP_LEFT_RIGHT)
    
    img = Image.new("RGBA", (24,24), self.bgcolor)

    alpha_over(img, side1, (12,0), side1)
    alpha_over(img, side2, (0,0), side2)

    alpha_over(img, side1, (0,6), side1)
    alpha_over(img, side2, (12,6), side2)
    
    return img

# monster spawner
block(blockid=52, top_index=34, transparent=True)

# wooden, cobblestone, red brick, stone brick, netherbrick, sandstone, spruce, birch and jungle stairs.
@material(blockid=[53,67,108,109,114,128,134,135,136], data=range(8), transparent=True, solid=True, nospawn=True)
def stairs(self, blockid, data):

    # first, rotations
    # preserve the upside-down bit
    upside_down = data & 0x4
    data = data & 0x3
    if self.rotation == 1:
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 1
        elif data == 3: data = 0
    elif self.rotation == 2:
        if data == 0: data = 1
        elif data == 1: data = 0
        elif data == 2: data = 3
        elif data == 3: data = 2
    elif self.rotation == 3:
        if data == 0: data = 3
        elif data == 1: data = 2
        elif data == 2: data = 0
        elif data == 3: data = 1
    data = data | upside_down

    if blockid == 53: # wooden
        texture = self.terrain_images[4]
    elif blockid == 67: # cobblestone
        texture = self.terrain_images[16]
    elif blockid == 108: # red brick stairs
        texture = self.terrain_images[7]
    elif blockid == 109: # stone brick stairs
        texture = self.terrain_images[54]
    elif blockid == 114: # netherbrick stairs
        texture = self.terrain_images[224]
    elif blockid == 128: # sandstone stairs
        texture = self.terrain_images[192]
    elif blockid == 134: # spruce wood stairs
        texture = self.terrain_images[198]
    elif blockid == 135: # birch wood  stairs
        texture = self.terrain_images[214]
    elif blockid == 136: # jungle good stairs
        texture = self.terrain_images[199]


    side = texture.copy()
    half_block_u = texture.copy() # up, down, left, right
    half_block_d = texture.copy()
    half_block_l = texture.copy()
    half_block_r = texture.copy()

    # sandstone stairs have spcial top texture
    if blockid == 128:
        half_block_u = self.terrain_images[176].copy()
        half_block_d = self.terrain_images[176].copy()
        texture = self.terrain_images[176].copy()

    # generate needed geometries
    ImageDraw.Draw(side).rectangle((0,0,7,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(half_block_u).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(half_block_d).rectangle((0,0,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(half_block_l).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(half_block_r).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
    
    if data & 0x4 == 0x4: # upside doen stair
        side = side.transpose(Image.FLIP_TOP_BOTTOM)
        if data & 0x3 == 0: # ascending east
            img = Image.new("RGBA", (24,24), self.bgcolor) # first paste the texture in the back
            tmp = self.transform_image_side(half_block_d)
            alpha_over(img, tmp, (6,3))
            alpha_over(img, self.build_full_block(texture, None, None, half_block_u, side.transpose(Image.FLIP_LEFT_RIGHT)))

        elif data & 0x3 == 0x1: # ascending west
            img = self.build_full_block(texture, None, None, texture, side)
        
        elif data & 0x3 == 0x2: # ascending south
            img = self.build_full_block(texture, None, None, side, texture)
            
        elif data & 0x3 == 0x3: # ascending north
            img = Image.new("RGBA", (24,24), self.bgcolor) # first paste the texture in the back
            tmp = self.transform_image_side(half_block_d).transpose(Image.FLIP_LEFT_RIGHT)
            alpha_over(img, tmp, (6,3))
            alpha_over(img, self.build_full_block(texture, None, None, side.transpose(Image.FLIP_LEFT_RIGHT), half_block_u))
        
    else: # normal stair
        if data == 0: # ascending east
            img = self.build_full_block(half_block_r, None, None, half_block_d, side.transpose(Image.FLIP_LEFT_RIGHT))
            tmp1 = self.transform_image_side(half_block_u)
            
            # Darken the vertical part of the second step
            sidealpha = tmp1.split()[3]
            # darken it a bit more than usual, looks better
            tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.8)
            tmp1.putalpha(sidealpha)
            
            alpha_over(img, tmp1, (6,4)) #workaround, fixes a hole
            alpha_over(img, tmp1, (6,3))
            tmp2 = self.transform_image_top(half_block_l)
            alpha_over(img, tmp2, (0,6))
            
        elif data == 1: # ascending west
            img = Image.new("RGBA", (24,24), self.bgcolor) # first paste the texture in the back
            tmp1 = self.transform_image_top(half_block_r)
            alpha_over(img, tmp1, (0,6))
            tmp2 = self.build_full_block(half_block_l, None, None, texture, side)
            alpha_over(img, tmp2)
        
        elif data == 2: # ascending south
            img = Image.new("RGBA", (24,24), self.bgcolor) # first paste the texture in the back
            tmp1 = self.transform_image_top(half_block_u)
            alpha_over(img, tmp1, (0,6))
            tmp2 = self.build_full_block(half_block_d, None, None, side, texture)
            alpha_over(img, tmp2)
            
        elif data == 3: # ascending north
            img = self.build_full_block(half_block_u, None, None, side.transpose(Image.FLIP_LEFT_RIGHT), half_block_d)
            tmp1 = self.transform_image_side(half_block_u).transpose(Image.FLIP_LEFT_RIGHT)
            
            # Darken the vertical part of the second step
            sidealpha = tmp1.split()[3]
            # darken it a bit more than usual, looks better
            tmp1 = ImageEnhance.Brightness(tmp1).enhance(0.7)
            tmp1.putalpha(sidealpha)
            
            alpha_over(img, tmp1, (6,4)) #workaround, fixes a hole
            alpha_over(img, tmp1, (6,3))
            tmp2 = self.transform_image_top(half_block_d)
            alpha_over(img, tmp2, (0,6))
        
        # touch up a (horrible) pixel
        img.putpixel((18,3),(0,0,0,0))
        
    return img

# normal, locked (used in april's fool day) and ender chests chests
@material(blockid=[54,95,130], data=range(30), transparent = True)
def chests(self, blockid, data):
    # the first 3 bits are the orientation as stored in minecraft, 
    # bits 0x8 and 0x10 indicate which half of the double chest is it.
    
    # first, do the rotation if needed
    orientation_data = data & 7
    if self.rotation == 1:
        if orientation_data == 2: data = 5 | (data & 24)
        elif orientation_data == 3: data = 4 | (data & 24)
        elif orientation_data == 4: data = 2 | (data & 24)
        elif orientation_data == 5: data = 3 | (data & 24)
    elif self.rotation == 2:
        if orientation_data == 2: data = 3 | (data & 24)
        elif orientation_data == 3: data = 2 | (data & 24)
        elif orientation_data == 4: data = 5 | (data & 24)
        elif orientation_data == 5: data = 4 | (data & 24)
    elif self.rotation == 3:
        if orientation_data == 2: data = 4 | (data & 24)
        elif orientation_data == 3: data = 5 | (data & 24)
        elif orientation_data == 4: data = 3 | (data & 24)
        elif orientation_data == 5: data = 2 | (data & 24)
    
    if blockid in (95,130) and not data in [2,3,4,5]: return None
        # iterate.c will only return the ancil data (without pseudo 
        # ancil data) for locked and ender chests, so only 
        # ancilData = 2,3,4,5 are used for this blockids
    
    if data & 24 == 0:
        if blockid == 130: t = self.load_image("enderchest.png")
        else: t = self.load_image("chest.png")

        # the textures is no longer in terrain.png, get it from 
        # item/chest.png and get by cropping all the needed stuff
        if t.size != (64,64): t = t.resize((64,64), Image.ANTIALIAS)
        # top
        top = t.crop((14,0,28,14))
        top.load() # every crop need a load, crop is a lazy operation
                   # see PIL manual
        img = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(img,top,(1,1))
        top = img
        # front
        front_top = t.crop((14,14,28,19))
        front_top.load()
        front_bottom = t.crop((14,34,28,43))
        front_bottom.load()
        front_lock = t.crop((1,0,3,4))
        front_lock.load()
        front = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(front,front_top, (1,1))
        alpha_over(front,front_bottom, (1,6))
        alpha_over(front,front_lock, (7,3))
        # left side
        # left side, right side, and back are esentially the same for
        # the default texture, we take it anyway just in case other
        # textures make use of it.
        side_l_top = t.crop((0,14,14,19))
        side_l_top.load()
        side_l_bottom = t.crop((0,34,14,43))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(side_l,side_l_top, (1,1))
        alpha_over(side_l,side_l_bottom, (1,6))
        # right side
        side_r_top = t.crop((28,14,43,20))
        side_r_top.load()
        side_r_bottom = t.crop((28,33,42,43))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(side_r,side_l_top, (1,1))
        alpha_over(side_r,side_l_bottom, (1,6))
        # back
        back_top = t.crop((42,14,56,18))
        back_top.load()
        back_bottom = t.crop((42,33,56,43))
        back_bottom.load()
        back = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(back,side_l_top, (1,1))
        alpha_over(back,side_l_bottom, (1,6))

    else:
        # large chest
        # the textures is no longer in terrain.png, get it from 
        # item/chest.png and get all the needed stuff
        t = self.load_image("largechest.png")
        if t.size != (128,64): t = t.resize((128,64), Image.ANTIALIAS)
        # top
        top = t.crop((14,0,44,14))
        top.load()
        img = Image.new("RGBA", (32,16), self.bgcolor)
        alpha_over(img,top,(1,1))
        top = img
        # front
        front_top = t.crop((14,14,44,18))
        front_top.load()
        front_bottom = t.crop((14,33,44,43))
        front_bottom.load()
        front_lock = t.crop((1,0,3,5))
        front_lock.load()
        front = Image.new("RGBA", (32,16), self.bgcolor)
        alpha_over(front,front_top,(1,1))
        alpha_over(front,front_bottom,(1,5))
        alpha_over(front,front_lock,(15,3))
        # left side
        side_l_top = t.crop((0,14,14,18))
        side_l_top.load()
        side_l_bottom = t.crop((0,33,14,43))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(side_l,side_l_top, (1,1))
        alpha_over(side_l,side_l_bottom,(1,5))
        # right side
        side_r_top = t.crop((44,14,58,18))
        side_r_top.load()
        side_r_bottom = t.crop((44,33,58,43))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16,16), self.bgcolor)
        alpha_over(side_r,side_r_top, (1,1))
        alpha_over(side_r,side_r_bottom,(1,5))
        # back
        back_top = t.crop((58,14,88,18))
        back_top.load()
        back_bottom = t.crop((58,33,88,43))
        back_bottom.load()
        back = Image.new("RGBA", (32,16), self.bgcolor)
        alpha_over(back,back_top,(1,1))
        alpha_over(back,back_bottom,(1,5))
        

        if data & 24 == 8: # double chest, first half
            top = top.crop((0,0,16,16))
            top.load()
            front = front.crop((0,0,16,16))
            front.load()
            back = back.crop((0,0,16,16))
            back.load()
            #~ side = side_l

        elif data & 24 == 16: # double, second half
            top = top.crop((16,0,32,16))
            top.load()
            front = front.crop((16,0,32,16))
            front.load()
            back = back.crop((16,0,32,16))
            back.load()
            #~ side = side_r

        else: # just in case
            return None

    # compose the final block
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if data & 7 == 2: # north
        side = self.transform_image_side(side_r)
        alpha_over(img, side, (1,7))
        back = self.transform_image_side(back)
        alpha_over(img, back.transpose(Image.FLIP_LEFT_RIGHT), (11,7))
        front = self.transform_image_side(front)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0,2))

    elif data & 7 == 3: # south
        side = self.transform_image_side(side_l)
        alpha_over(img, side, (1,7))
        front = self.transform_image_side(front).transpose(Image.FLIP_LEFT_RIGHT)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0,2))
        alpha_over(img, front,(11,7))

    elif data & 7 == 4: # west
        side = self.transform_image_side(side_r)
        alpha_over(img, side.transpose(Image.FLIP_LEFT_RIGHT), (11,7))
        front = self.transform_image_side(front)
        alpha_over(img, front,(1,7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0,2))

    elif data & 7 == 5: # east
        back = self.transform_image_side(back)
        side = self.transform_image_side(side_l).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, side, (11,7))
        alpha_over(img, back, (1,7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0,2))
        
    else: # just in case
        img = None

    return img

# redstone wire
# uses pseudo-ancildata found in iterate.c
@material(blockid=55, data=range(128), transparent=True)
def wire(self, blockid, data):

    if data & 0b1000000 == 64: # powered redstone wire
        redstone_wire_t = self.terrain_images[165]
        redstone_wire_t = self.tint_texture(redstone_wire_t,(255,0,0))

        redstone_cross_t = self.terrain_images[164]
        redstone_cross_t = self.tint_texture(redstone_cross_t,(255,0,0))

        
    else: # unpowered redstone wire
        redstone_wire_t = self.terrain_images[165]
        redstone_wire_t = self.tint_texture(redstone_wire_t,(48,0,0))
        
        redstone_cross_t = self.terrain_images[164]
        redstone_cross_t = self.tint_texture(redstone_cross_t,(48,0,0))

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
        bottom = Image.new("RGBA", (16,16), self.bgcolor)
        if (data & 0b0001) == 1:
            alpha_over(bottom,branch_top_left)
            
        if (data & 0b1000) == 8:
            alpha_over(bottom,branch_top_right)
            
        if (data & 0b0010) == 2:
            alpha_over(bottom,branch_bottom_left)
            
        if (data & 0b0100) == 4:
            alpha_over(bottom,branch_bottom_right)

    # check for going up redstone wire
    if data & 0b100000 == 32:
        side1 = redstone_wire_t.rotate(90)
    else:
        side1 = None
        
    if data & 0b010000 == 16:
        side2 = redstone_wire_t.rotate(90)
    else:
        side2 = None
        
    img = self.build_full_block(None,side1,side2,None,None,bottom)

    return img

# diamond ore
block(blockid=56, top_index=50)
# diamond block
block(blockid=57, top_index=24)

# crafting table
# needs two different sides
@material(blockid=58, solid=True, nodata=True)
def crafting_table(self, blockid, data):
    top = self.terrain_images[43]
    side3 = self.terrain_images[43+16]
    side4 = self.terrain_images[43+16+1]
    
    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

# crops
@material(blockid=59, data=range(8), transparent=True, nospawn=True)
def crops(self, blockid, data):
    raw_crop = self.terrain_images[88+data]
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# farmland
@material(blockid=60, data=range(9), solid=True)
def farmland(self, blockid, data):
    top = self.terrain_images[86]
    if data == 0:
        top = self.terrain_images[87]
    return self.build_block(top, self.terrain_images[2])

# signposts
@material(blockid=63, data=range(16), transparent=True)
def signpost(self, blockid, data):

    # first rotations
    if self.rotation == 1:
        data = (data + 4) % 16
    elif self.rotation == 2:
        data = (data + 8) % 16
    elif self.rotation == 3:
        data = (data + 12) % 16

    texture = self.terrain_images[4].copy()
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
    texture_stick = self.terrain_images[20]
    texture_stick = texture_stick.resize((12,12), Image.ANTIALIAS)
    ImageDraw.Draw(texture_stick).rectangle((2,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    #         W                N      ~90       E                   S        ~270
    angles = (330.,345.,0.,15.,30.,55.,95.,120.,150.,165.,180.,195.,210.,230.,265.,310.)
    angle = math.radians(angles[data])
    post = self.transform_image_angle(texture, angle)

    # choose the position of the "3D effect"
    incrementx = 0
    if data in (1,6,7,8,9,14):
        incrementx = -1
    elif data in (3,4,5,11,12,13):
        incrementx = +1

    alpha_over(img, texture_stick,(11, 8),texture_stick)
    # post2 is a brighter signpost pasted with a small shift,
    # gives to the signpost some 3D effect.
    post2 = ImageEnhance.Brightness(post).enhance(1.2)
    alpha_over(img, post2,(incrementx, -3),post2)
    alpha_over(img, post, (0,-2), post)

    return img


# wooden and iron door
# uses pseudo-ancildata found in iterate.c
@material(blockid=[64,71], data=range(32), transparent=True)
def door(self, blockid, data):
    #Masked to not clobber block top/bottom & swung info
    if self.rotation == 1:
        if (data & 0b00011) == 0: data = data & 0b11100 | 1
        elif (data & 0b00011) == 1: data = data & 0b11100 | 2
        elif (data & 0b00011) == 2: data = data & 0b11100 | 3
        elif (data & 0b00011) == 3: data = data & 0b11100 | 0
    elif self.rotation == 2:
        if (data & 0b00011) == 0: data = data & 0b11100 | 2
        elif (data & 0b00011) == 1: data = data & 0b11100 | 3
        elif (data & 0b00011) == 2: data = data & 0b11100 | 0
        elif (data & 0b00011) == 3: data = data & 0b11100 | 1
    elif self.rotation == 3:
        if (data & 0b00011) == 0: data = data & 0b11100 | 3
        elif (data & 0b00011) == 1: data = data & 0b11100 | 0
        elif (data & 0b00011) == 2: data = data & 0b11100 | 1
        elif (data & 0b00011) == 3: data = data & 0b11100 | 2

    if data & 0x8 == 0x8: # top of the door
        raw_door = self.terrain_images[81 if blockid == 64 else 82]
    else: # bottom of the door
        raw_door = self.terrain_images[97 if blockid == 64 else 98]
    
    # if you want to render all doors as closed, then force
    # force closed to be True
    if data & 0x4 == 0x4:
        closed = False
    else:
        closed = True
    
    if data & 0x10 == 0x10:
        # hinge on the left (facing same door direction)
        hinge_on_left = True
    else:
        # hinge on the right (default single door)
        hinge_on_left = False

    # mask out the high bits to figure out the orientation 
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if (data & 0x03) == 0: # facing west when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door)    
                alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
    
    if (data & 0x03) == 1: # facing north when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (0,6), tex)

        else:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (12,0), tex)

                
    if (data & 0x03) == 2: # facing east when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (12,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (12,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)

    if (data & 0x03) == 3: # facing south when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (12,0), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (0,6), tex)

    return img

# ladder
@material(blockid=65, data=[2, 3, 4, 5], transparent=True)
def ladder(self, blockid, data):

    # first rotations
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    img = Image.new("RGBA", (24,24), self.bgcolor)
    raw_texture = self.terrain_images[83]

    if data == 5:
        # normally this ladder would be obsured by the block it's attached to
        # but since ladders can apparently be placed on transparent blocks, we 
        # have to render this thing anyway.  same for data == 2
        tex = self.transform_image_side(raw_texture)
        alpha_over(img, tex, (0,6), tex)
        return img
    if data == 2:
        tex = self.transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, tex, (12,6), tex)
        return img
    if data == 3:
        tex = self.transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, tex, (0,0), tex)
        return img
    if data == 4:
        tex = self.transform_image_side(raw_texture)
        alpha_over(img, tex, (12,0), tex)
        return img


# wall signs
@material(blockid=68, data=[2, 3, 4, 5], transparent=True)
def wall_sign(self, blockid, data): # wall sign

    # first rotations
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    texture = self.terrain_images[4].copy()
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
    
    img = Image.new("RGBA", (24,24), self.bgcolor)

    incrementx = 0
    if data == 2:  # east
        incrementx = +1
        sign = self.build_full_block(None, None, None, None, texture)
    elif data == 3:  # west
        incrementx = -1
        sign = self.build_full_block(None, texture, None, None, None)
    elif data == 4:  # north
        incrementx = +1
        sign = self.build_full_block(None, None, texture, None, None)
    elif data == 5:  # south
        incrementx = -1
        sign = self.build_full_block(None, None, None, texture, None)

    sign2 = ImageEnhance.Brightness(sign).enhance(1.2)
    alpha_over(img, sign2,(incrementx, 2),sign2)
    alpha_over(img, sign, (0,3), sign)

    return img

# levers
@material(blockid=69, data=range(16), transparent=True)
def levers(self, blockid, data):
    if data & 8 == 8: powered = True
    else: powered = False

    data = data & 7

    # first rotations
    if self.rotation == 1:
        # on wall levers
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
        # on floor levers
        elif data == 5: data = 6
        elif data == 6: data = 5
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
        elif data == 5: data = 5
        elif data == 6: data = 6
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 5: data = 6
        elif data == 6: data = 5

    # generate the texture for the base of the lever
    t_base = self.terrain_images[16].copy()

    ImageDraw.Draw(t_base).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # generate the texture for the stick
    stick = self.terrain_images[96].copy()
    c_stick = Image.new("RGBA", (16,16), self.bgcolor)
    
    tmp = ImageEnhance.Brightness(stick).enhance(0.8)
    alpha_over(c_stick, tmp, (1,0), tmp)
    alpha_over(c_stick, stick, (0,0), stick)
    t_stick = self.transform_image_side(c_stick.rotate(45, Image.NEAREST))

    # where the lever will be composed
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    # wall levers
    if data == 1: # facing SOUTH
        # levers can't be placed in transparent blocks, so this
        # direction is almost invisible
        return None

    elif data == 2: # facing NORTH
        base = self.transform_image_side(t_base)
        
        # paste it twice with different brightness to make a fake 3D effect
        alpha_over(img, base, (12,-1), base)

        alpha = base.split()[3]
        base = ImageEnhance.Brightness(base).enhance(0.9)
        base.putalpha(alpha)
        
        alpha_over(img, base, (11,0), base)

        # paste the lever stick
        pos = (7,-7)
        if powered:
            t_stick = t_stick.transpose(Image.FLIP_TOP_BOTTOM)
            pos = (7,6)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 3: # facing WEST
        base = self.transform_image_side(t_base)
        
        # paste it twice with different brightness to make a fake 3D effect
        base = base.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, base, (0,-1), base)

        alpha = base.split()[3]
        base = ImageEnhance.Brightness(base).enhance(0.9)
        base.putalpha(alpha)
        
        alpha_over(img, base, (1,0), base)
        
        # paste the lever stick
        t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
        pos = (5,-7)
        if powered:
            t_stick = t_stick.transpose(Image.FLIP_TOP_BOTTOM)
            pos = (6,6)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 4: # facing EAST
        # levers can't be placed in transparent blocks, so this
        # direction is almost invisible
        return None

    # floor levers
    elif data == 5: # pointing south when off
        # lever base, fake 3d again
        base = self.transform_image_top(t_base)

        alpha = base.split()[3]
        tmp = ImageEnhance.Brightness(base).enhance(0.8)
        tmp.putalpha(alpha)
        
        alpha_over(img, tmp, (0,12), tmp)
        alpha_over(img, base, (0,11), base)

        # lever stick
        pos = (3,2)
        if not powered:
            t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
            pos = (11,2)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 6: # pointing east when off
        # lever base, fake 3d again
        base = self.transform_image_top(t_base.rotate(90))

        alpha = base.split()[3]
        tmp = ImageEnhance.Brightness(base).enhance(0.8)
        tmp.putalpha(alpha)
        
        alpha_over(img, tmp, (0,12), tmp)
        alpha_over(img, base, (0,11), base)

        # lever stick
        pos = (2,3)
        if not powered:
            t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
            pos = (10,2)
        alpha_over(img, t_stick, pos, t_stick)

    return img

# wooden and stone pressure plates
@material(blockid=[70, 72], data=[0,1], transparent=True)
def pressure_plate(self, blockid, data):
    if blockid == 70: # stone
        t = self.terrain_images[1].copy()
    else: # wooden
        t = self.terrain_images[4].copy()
    
    # cut out the outside border, pressure plates are smaller
    # than a normal block
    ImageDraw.Draw(t).rectangle((0,0,15,15),outline=(0,0,0,0))
    
    # create the textures and a darker version to make a 3d by 
    # pasting them with an offstet of 1 pixel
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    top = self.transform_image_top(t)
    
    alpha = top.split()[3]
    topd = ImageEnhance.Brightness(top).enhance(0.8)
    topd.putalpha(alpha)
    
    #show it 3d or 2d if unpressed or pressed
    if data == 0:
        alpha_over(img,topd, (0,12),topd)
        alpha_over(img,top, (0,11),top)
    elif data == 1:
        alpha_over(img,top, (0,12),top)
    
    return img

# normal and glowing redstone ore
block(blockid=[73, 74], top_index=51)

@material(blockid=77, data=range(16), transparent=True)
def buttons(self, blockid, data):

    # 0x8 is set if the button is pressed mask this info and render
    # it as unpressed
    data = data & 0x7

    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2

    t = self.terrain_images[1].copy()

    # generate the texture for the button
    ImageDraw.Draw(t).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    button = self.transform_image_side(t)
    
    if data == 1: # facing SOUTH
        # buttons can't be placed in transparent blocks, so this
        # direction can't be seen
        return None

    elif data == 2: # facing NORTH
        # paste it twice with different brightness to make a 3D effect
        alpha_over(img, button, (12,-1), button)

        alpha = button.split()[3]
        button = ImageEnhance.Brightness(button).enhance(0.9)
        button.putalpha(alpha)
        
        alpha_over(img, button, (11,0), button)

    elif data == 3: # facing WEST
        # paste it twice with different brightness to make a 3D effect
        button = button.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, button, (0,-1), button)

        alpha = button.split()[3]
        button = ImageEnhance.Brightness(button).enhance(0.9)
        button.putalpha(alpha)
        
        alpha_over(img, button, (1,0), button)

    elif data == 4: # facing EAST
        # buttons can't be placed in transparent blocks, so this
        # direction can't be seen
        return None

    return img

# snow
@material(blockid=78, data=range(16), transparent=True, solid=True)
def snow(self, blockid, data):
    # still not rendered correctly: data other than 0
    
    tex = self.terrain_images[66]
    
    # make the side image, top 3/4 transparent
    mask = tex.crop((0,12,16,16))
    sidetex = Image.new(tex.mode, tex.size, self.bgcolor)
    alpha_over(sidetex, mask, (0,12,16,16), mask)
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    top = self.transform_image_top(tex)
    side = self.transform_image_side(sidetex)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    alpha_over(img, side, (0,6), side)
    alpha_over(img, otherside, (12,6), otherside)
    alpha_over(img, top, (0,9), top)
    
    return img

# snow block
block(blockid=80, top_index=66)

# cactus
@material(blockid=81, data=range(15), transparent=True, solid=True, nospawn=True)
def cactus(self, blockid, data):
    top = self.terrain_images[69]
    side = self.terrain_images[70]

    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    top = self.transform_image_top(top)
    side = self.transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)

    alpha_over(img, side, (1,6), side)
    alpha_over(img, otherside, (11,6), otherside)
    alpha_over(img, top, (0,0), top)
    
    return img

# clay block
block(blockid=82, top_index=72)

# sugar cane
@material(blockid=83, data=range(16), transparent=True)
def sugar_cane(self, blockid, data):
    tex = self.terrain_images[73]
    return self.build_sprite(tex)

# jukebox
@material(blockid=84, data=range(16), solid=True)
def jukebox(self, blockid, data):
    return self.build_block(self.terrain_images[75], self.terrain_images[74])

# nether and normal fences
# uses pseudo-ancildata found in iterate.c
@material(blockid=[85, 113], data=range(16), transparent=True, nospawn=True)
def fence(self, blockid, data):
    # no need for rotations, it uses pseudo data.
    # create needed images for Big stick fence
    if blockid == 85: # normal fence
        fence_top = self.terrain_images[4].copy()
        fence_side = self.terrain_images[4].copy()
        fence_small_side = self.terrain_images[4].copy()
    else: # netherbrick fence
        fence_top = self.terrain_images[224].copy()
        fence_side = self.terrain_images[224].copy()
        fence_small_side = self.terrain_images[224].copy()

    # generate the textures of the fence
    ImageDraw.Draw(fence_top).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(fence_side).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_side).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the big stick
    fence_side = self.transform_image_side(fence_side)
    fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
    fence_top = self.transform_image_top(fence_top)

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
    fence_big = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(fence_big,fence_side, (5,4),fence_side)
    alpha_over(fence_big,fence_other_side, (7,4),fence_other_side)
    alpha_over(fence_big,fence_top, (0,0),fence_top)
    
    # Now render the small sticks.
    # Create needed images
    ImageDraw.Draw(fence_small_side).rectangle((0,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,4,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the small sticks
    fence_small_side = self.transform_image_side(fence_small_side)
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
    img = Image.new("RGBA", (24,24), self.bgcolor)

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
        alpha_over(img,fence_small_side, pos_top_left,fence_small_side)                # top left
    if (data & 0b1000) == 8:
        alpha_over(img,fence_small_other_side, pos_top_right,fence_small_other_side)    # top right
        
    alpha_over(img,fence_big,(0,0),fence_big)
        
    if (data & 0b0010) == 2:
        alpha_over(img,fence_small_other_side, pos_bottom_left,fence_small_other_side)      # bottom left    
    if (data & 0b0100) == 4:
        alpha_over(img,fence_small_side, pos_bottom_right,fence_small_side)                  # bottom right
    
    return img

# pumpkin
@material(blockid=[86, 91], data=range(4), solid=True)
def pumpkin(self, blockid, data): # pumpkins, jack-o-lantern
    # rotation
    if self.rotation == 1:
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif self.rotation == 2:
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif self.rotation == 3:
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2
    
    # texture generation
    top = self.terrain_images[102]
    frontID = 119 if blockid == 86 else 120
    front = self.terrain_images[frontID]
    side = self.terrain_images[118]

    if data == 0: # pointing west
        img = self.build_full_block(top, None, None, side, front)

    elif data == 1: # pointing north
        img = self.build_full_block(top, None, None, front, side)

    else: # in any other direction the front can't be seen
        img = self.build_full_block(top, None, None, side, side)

    return img

# netherrack
block(blockid=87, top_index=103)

# soul sand
block(blockid=88, top_index=104)

# glowstone
block(blockid=89, top_index=105)

# portal
@material(blockid=90, data=[1, 2, 4, 8], transparent=True)
def portal(self, blockid, data):
    # no rotations, uses pseudo data
    portaltexture = self.load_portal()
    img = Image.new("RGBA", (24,24), self.bgcolor)

    side = self.transform_image_side(portaltexture)
    otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

    if data in (1,4):
        alpha_over(img, side, (5,4), side)

    if data in (2,8):
        alpha_over(img, otherside, (5,4), otherside)

    return img

# cake!
@material(blockid=92, data=range(6), transparent=True, nospawn=True)
def cake(self, blockid, data):
    
    # cake textures
    top = self.terrain_images[121].copy()
    side = self.terrain_images[122].copy()
    fullside = side.copy()
    inside = self.terrain_images[123]
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if data == 0: # unbitten cake
        top = self.transform_image_top(top)
        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
        
        # darken sides slightly
        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)
        
        # composite the cake
        alpha_over(img, side, (1,6), side)
        alpha_over(img, otherside, (11,7), otherside) # workaround, fixes a hole
        alpha_over(img, otherside, (12,6), otherside)
        alpha_over(img, top, (0,6), top)
    
    else:
        # cut the textures for a bitten cake
        coord = int(16./6.*data)
        ImageDraw.Draw(side).rectangle((16 - coord,0,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(top).rectangle((0,0,coord,16),outline=(0,0,0,0),fill=(0,0,0,0))

        # the bitten part of the cake always points to the west
        # composite the cake for every north orientation
        if self.rotation == 0: # north top-left
            # create right side
            rs = self.transform_image_side(side).transpose(Image.FLIP_LEFT_RIGHT)
            # create bitten side and its coords
            deltax = 2*data
            deltay = -1*data
            if data == 3: deltax += 1 # special case fixing pixel holes
            ls = self.transform_image_side(inside)
            # create top side
            t = self.transform_image_top(top)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, rs, (12,6), rs)
            alpha_over(img, ls, (1 + deltax,6 + deltay), ls)
            alpha_over(img, t, (0,6), t)

        elif self.rotation == 1: # north top-right
            # bitten side not shown
            # create left side
            ls = self.transform_image_side(side.transpose(Image.FLIP_LEFT_RIGHT))
            # create top
            t = self.transform_image_top(top.rotate(-90))
            # create right side
            rs = self.transform_image_side(fullside).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (0,6), t)
            alpha_over(img, rs, (12,6), rs)

        elif self.rotation == 2: # north bottom-right
            # bitten side not shown
            # left side
            ls = self.transform_image_side(fullside)
            # top
            t = self.transform_image_top(top.rotate(180))
            # right side
            rs = self.transform_image_side(side.transpose(Image.FLIP_LEFT_RIGHT)).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (1,6), t)
            alpha_over(img, rs, (12,6), rs)

        elif self.rotation == 3: # north bottom-left
            # create left side
            ls = self.transform_image_side(side)
            # create top
            t = self.transform_image_top(top.rotate(90))
            # create right side and its coords
            deltax = 12-2*data
            deltay = -1*data
            if data == 3: deltax += -1 # special case fixing pixel holes
            rs = self.transform_image_side(inside).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (1,6), t)
            alpha_over(img, rs, (1 + deltax,6 + deltay), rs)

    return img

# redstone repeaters ON and OFF
@material(blockid=[93,94], data=range(16), transparent=True, nospawn=True)
def repeater(self, blockid, data):
    # rotation
    # Masked to not clobber delay info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    
    # generate the diode
    top = self.terrain_images[131] if blockid == 93 else self.terrain_images[147]
    side = self.terrain_images[5]
    increment = 13
    
    if (data & 0x3) == 0: # pointing east
        pass
    
    if (data & 0x3) == 1: # pointing south
        top = top.rotate(270)

    if (data & 0x3) == 2: # pointing west
        top = top.rotate(180)

    if (data & 0x3) == 3: # pointing north
        top = top.rotate(90)

    img = self.build_full_block( (top, increment), None, None, side, side)

    # compose a "3d" redstone torch
    t = self.terrain_images[115].copy() if blockid == 93 else self.terrain_images[99].copy()
    torch = Image.new("RGBA", (24,24), self.bgcolor)
    
    t_crop = t.crop((2,2,14,14))
    slice = t_crop.copy()
    ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
    
    alpha_over(torch, slice, (6,4))
    alpha_over(torch, t_crop, (5,5))
    alpha_over(torch, t_crop, (6,5))
    alpha_over(torch, slice, (6,6))
    
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
    alpha_over(img, torch, static_torch, torch) 
    alpha_over(img, torch, moving_torch, torch)

    return img
    
# trapdoor
# TODO the trapdoor is looks like a sprite when opened, that's not good
@material(blockid=96, data=range(8), transparent=True, nospawn=True)
def trapdoor(self, blockid, data):

    # rotation
    # Masked to not clobber opened/closed info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0

    # texture generation
    texture = self.terrain_images[84]
    if data & 0x4 == 0x4: # opened trapdoor
        if data & 0x3 == 0: # west
            img = self.build_full_block(None, None, None, None, texture)
        if data & 0x3 == 1: # east
            img = self.build_full_block(None, texture, None, None, None)
        if data & 0x3 == 2: # south
            img = self.build_full_block(None, None, texture, None, None)
        if data & 0x3 == 3: # north
            img = self.build_full_block(None, None, None, texture, None)
        
    elif data & 0x4 == 0: # closed trapdoor
        img = self.build_full_block((texture, 12), None, None, texture, texture)
    
    return img

# block with hidden silverfish (stone, cobblestone and stone brick)
@material(blockid=97, data=range(3), solid=True)
def hidden_silverfish(self, blockid, data):
    if data == 0: # stone
        t = self.terrain_images[1]
    elif data == 1: # cobblestone
        t = self.terrain_images[16]
    elif data == 2: # stone brick
        t = self.terrain_images[54]
    
    img = self.build_block(t, t)
    
    return img

# stone brick
@material(blockid=98, data=range(4), solid=True)
def stone_brick(self, blockid, data):
    if data == 0: # normal
        t = self.terrain_images[54]
    elif data == 1: # mossy
        t = self.terrain_images[100]
    elif data == 2: # cracked
        t = self.terrain_images[101]
    elif data == 3: # "circle" stone brick
        t = self.terrain_images[213]

    img = self.build_full_block(t, None, None, t, t)

    return img

# huge brown and red mushroom
@material(blockid=[99,100], data=range(11), solid=True)
def huge_mushroom(self, blockid, data):
    # rotation
    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 6
        elif data == 3: data = 9
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 1
        elif data == 8: data = 4
        elif data == 9: data = 7
    elif self.rotation == 2:
        if data == 1: data = 9
        elif data == 2: data = 8
        elif data == 3: data = 7
        elif data == 4: data = 6
        elif data == 6: data = 4
        elif data == 7: data = 3
        elif data == 8: data = 2
        elif data == 9: data = 1
    elif self.rotation == 3:
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
        cap = self.terrain_images[126]
    else: # red
        cap = self.terrain_images[125]

    stem = self.terrain_images[141]
    porous = self.terrain_images[142]
    
    if data == 0: # fleshy piece
        img = self.build_full_block(porous, None, None, porous, porous)

    if data == 1: # north-east corner
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 2: # east side
        img = self.build_full_block(cap, None, None, porous, porous)

    if data == 3: # south-east corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 4: # north side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 5: # top piece
        img = self.build_full_block(cap, None, None, porous, porous)

    if data == 6: # south side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 7: # north-west corner
        img = self.build_full_block(cap, None, None, cap, cap)

    if data == 8: # west side
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 9: # south-west corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 10: # stem
        img = self.build_full_block(porous, None, None, stem, stem)

    return img

# iron bars and glass pane
# TODO glass pane is not a sprite, it has a texture for the side,
# at the moment is not used
@material(blockid=[101,102], data=range(16), transparent=True, nospawn=True)
def panes(self, blockid, data):
    # no rotation, uses pseudo data
    if blockid == 101:
        # iron bars
        t = self.terrain_images[85]
    else:
        # glass panes
        t = self.terrain_images[49]
    left = t.copy()
    right = t.copy()

    # generate the four small pieces of the glass pane
    ImageDraw.Draw(right).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(left).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    
    up_left = self.transform_image_side(left)
    up_right = self.transform_image_side(right).transpose(Image.FLIP_TOP_BOTTOM)
    dw_right = self.transform_image_side(right)
    dw_left = self.transform_image_side(left).transpose(Image.FLIP_TOP_BOTTOM)

    # Create img to compose the texture
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # +x axis points top right direction
    # +y axis points bottom right direction
    # First compose things in the back of the image, 
    # then things in the front.

    if (data & 0b0001) == 1 or data == 0:
        alpha_over(img,up_left, (6,3),up_left)    # top left
    if (data & 0b1000) == 8 or data == 0:
        alpha_over(img,up_right, (6,3),up_right)  # top right
    if (data & 0b0010) == 2 or data == 0:
        alpha_over(img,dw_left, (6,3),dw_left)    # bottom left    
    if (data & 0b0100) == 4 or data == 0:
        alpha_over(img,dw_right, (6,3),dw_right)  # bottom right

    return img

# melon
block(blockid=103, top_index=137, side_index=136, solid=True)

# pumpkin and melon stem
# TODO To render it as in game needs from pseudo data and ancil data:
# once fully grown the stem bends to the melon/pumpkin block,
# at the moment only render the growing stem
@material(blockid=[104,105], data=range(8), transparent=True)
def stem(self, blockid, data):
    # the ancildata value indicates how much of the texture
    # is shown.

    # not fully grown stem or no pumpkin/melon touching it,
    # straight up stem
    t = self.terrain_images[111].copy()
    img = Image.new("RGBA", (16,16), self.bgcolor)
    alpha_over(img, t, (0, int(16 - 16*((data + 1)/8.))), t)
    img = self.build_sprite(t)
    if data & 7 == 7:
        # fully grown stem gets brown color!
        # there is a conditional in rendermode-normal.c to not
        # tint the data value 7
        img = self.tint_texture(img, (211,169,116))
    return img
    

# vines
@material(blockid=106, data=range(16), transparent=True)
def vines(self, blockid, data):
    # rotation
    # vines data is bit coded. decode it first.
    # NOTE: the directions used in this function are the new ones used
    # in minecraft 1.0.0, no the ones used by overviewer 
    # (i.e. north is top-left by defalut)

    # rotate the data by bitwise shift
    shifts = 0
    if self.rotation == 1:
        shifts = 1
    elif self.rotation == 2:
        shifts = 2
    elif self.rotation == 3:
        shifts = 3
    
    for i in range(shifts):
        data = data * 2
        if data & 16:
            data = (data - 16) | 1

    # decode data and prepare textures
    raw_texture = self.terrain_images[143]
    s = w = n = e = None

    if data & 1: # south
        s = raw_texture
    if data & 2: # west
        w = raw_texture
    if data & 4: # north
        n = raw_texture
    if data & 8: # east
        e = raw_texture

    # texture generation
    img = self.build_full_block(None, n, e, w, s)

    return img

# fence gates
@material(blockid=107, data=range(8), transparent=True, nospawn=True)
def fence_gate(self, blockid, data):

    # rotation
    opened = False
    if data & 0x4:
        data = data & 0x3
        opened = True
    if self.rotation == 1:
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif self.rotation == 2:
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif self.rotation == 3:
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2
    if opened:
        data = data | 0x4

    # create the closed gate side
    gate_side = self.terrain_images[4].copy()
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
    mirror_gate_side = self.transform_image_side(gate_side.transpose(Image.FLIP_LEFT_RIGHT))
    gate_side = self.transform_image_side(gate_side)
    gate_other_side = gate_side.transpose(Image.FLIP_LEFT_RIGHT)
    mirror_gate_other_side = mirror_gate_side.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Create img to compose the fence gate
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    if data & 0x4:
        # opened
        data = data & 0x3
        if data == 0:
            alpha_over(img, gate_side, (2,8), gate_side)
            alpha_over(img, gate_side, (13,3), gate_side)
        elif data == 1:
            alpha_over(img, gate_other_side, (-1,3), gate_other_side)
            alpha_over(img, gate_other_side, (10,8), gate_other_side)
        elif data == 2:
            alpha_over(img, mirror_gate_side, (-1,7), mirror_gate_side)
            alpha_over(img, mirror_gate_side, (10,2), mirror_gate_side)
        elif data == 3:
            alpha_over(img, mirror_gate_other_side, (2,1), mirror_gate_other_side)
            alpha_over(img, mirror_gate_other_side, (13,7), mirror_gate_other_side)
    else:
        # closed
        
        # positions for pasting the fence sides, as with fences
        pos_top_left = (2,3)
        pos_top_right = (10,3)
        pos_bottom_right = (10,7)
        pos_bottom_left = (2,7)
        
        if data == 0 or data == 2:
            alpha_over(img, gate_other_side, pos_top_right, gate_other_side)
            alpha_over(img, mirror_gate_other_side, pos_bottom_left, mirror_gate_other_side)
        elif data == 1 or data == 3:
            alpha_over(img, gate_side, pos_top_left, gate_side)
            alpha_over(img, mirror_gate_side, pos_bottom_right, mirror_gate_side)
    
    return img

# mycelium
block(blockid=110, top_index=78, side_index=77)

# lilypad
# At the moment of writing this lilypads has no ancil data and their
# orientation depends on their position on the map. Because lilypads had
# ancildata, leave some data values just in case an old map have some
# with ancil data.
@material(blockid=111, data=range(4), transparent=True)
def lilypad(self, blockid, data):
    t = self.terrain_images[76]
    return self.build_full_block(None, None, None, None, None, t)

# nether brick
block(blockid=112, top_index=224, side_index=224)

# nether wart
@material(blockid=115, data=range(4), transparent=True)
def nether_wart(self, blockid, data):
    if data == 0: # just come up
        t = self.terrain_images[226]
    elif data in (1, 2):
        t = self.terrain_images[227]
    else: # fully grown
        t = self.terrain_images[228]
    
    # use the same technic as tall grass
    img = self.build_billboard(t)

    return img

# enchantment table
# TODO there's no book at the moment
@material(blockid=116, transparent=True, nodata=True)
def enchantment_table(self, blockid, data):
    # no book at the moment
    top = self.terrain_images[166]
    side = self.terrain_images[182]
    img = self.build_full_block((top, 4), None, None, side, side)

    return img

# brewing stand
# TODO this is a place holder, is a 2d image pasted
@material(blockid=117, data=range(5), transparent=True)
def brewing_stand(self, blockid, data):
    base = self.terrain_images[156]
    img = self.build_full_block(None, None, None, None, None, base)
    t = self.terrain_images[157]
    stand = self.build_billboard(t)
    alpha_over(img,stand,(0,-2))
    return img

# cauldron
@material(blockid=118, data=range(4), transparent=True)
def cauldron(self, blockid, data):
    side = self.terrain_images[154]
    top = self.terrain_images[138]
    bottom = self.terrain_images[139]
    water = self.transform_image_top(self.load_water())
    if data == 0: # empty
        img = self.build_full_block(top, side, side, side, side)
    if data == 1: # 1/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,8), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)
    if data == 2: # 2/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,4), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)
    if data == 3: # 3/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,0), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)

    return img

# end portal
@material(blockid=119, transparent=True, nodata=True)
def end_portal(self, blockid, data):
    img = Image.new("RGBA", (24,24), self.bgcolor)
    # generate a black texure with white, blue and grey dots resembling stars
    t = Image.new("RGBA", (16,16), (0,0,0,255))
    for color in [(155,155,155,255), (100,255,100,255), (255,255,255,255)]:
        for i in range(6):
            x = randint(0,15)
            y = randint(0,15)
            t.putpixel((x,y),color)

    t = self.transform_image_top(t)
    alpha_over(img, t, (0,0), t)

    return img

# end portal frame
@material(blockid=120, data=range(5), transparent=True)
def end_portal_frame(self, blockid, data):
    # The bottom 2 bits are oritation info but seems there is no
    # graphical difference between orientations
    top = self.terrain_images[158]
    eye_t = self.terrain_images[174]
    side = self.terrain_images[159]
    img = self.build_full_block((top, 4), None, None, side, side)
    if data & 0x4 == 0x4: # ender eye on it
        # generate the eye
        eye_t = self.terrain_images[174].copy()
        eye_t_s = self.terrain_images[174].copy()
        # cut out from the texture the side and the top of the eye
        ImageDraw.Draw(eye_t).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(eye_t_s).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        # trnasform images and paste
        eye = self.transform_image_top(eye_t)
        eye_s = self.transform_image_side(eye_t_s)
        eye_os = eye_s.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, eye_s, (5,5), eye_s)
        alpha_over(img, eye_os, (9,5), eye_os)
        alpha_over(img, eye, (0,0), eye)

    return img

# end stone
block(blockid=121, top_index=175)

# dragon egg
# NOTE: this isn't a block, but I think it's better than nothing
block(blockid=122, top_index=37)

# inactive redstone lamp
block(blockid=123, top_index=211)

# active redstone lamp
block(blockid=124, top_index=212)

# wooden double and normal slabs
# these are the new wooden slabs, blockids 43 44 still have wooden
# slabs, but those are unobtainable without cheating
@material(blockid=[125, 126], data=range(16), transparent=(44,), solid=True)
def wooden_slabs(self, blockid, data):
    texture = data & 7
    if texture== 0: # oak 
        top = side = self.terrain_images[4]
    elif texture== 1: # spruce
        top = side = self.terrain_images[198]
    elif texture== 2: # birch
        top = side = self.terrain_images[214]
    elif texture== 3: # jungle
        top = side = self.terrain_images[199]
    else:
        return None
    
    if blockid == 125: # double slab
        return self.build_block(top, side)
    
    # cut the side texture in half
    mask = side.crop((0,8,16,16))
    side = Image.new(side.mode, side.size, self.bgcolor)
    alpha_over(side, mask,(0,0,16,8), mask)
    
    # plain slab
    top = self.transform_image_top(top)
    side = self.transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)
    
    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)
    
    # upside down slab
    delta = 0
    if data & 8 == 8:
        delta = 6
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, side, (0,12 - delta), side)
    alpha_over(img, otherside, (12,12 - delta), otherside)
    alpha_over(img, top, (0,6 - delta), top)
    
    return img

# emerald ore
block(blockid=129, top_index=171)

# emerald block
block(blockid=133, top_index=25)

# cocoa plant
@material(blockid=127, data=range(12), transparent=True)
def cocoa_plant(self, blockid, data):
    orientation = data & 3
    # rotation
    if self.rotation == 1:
        if orientation == 0: orientation = 1
        elif orientation == 1: orientation = 2
        elif orientation == 2: orientation = 3
        elif orientation == 3: orientation = 0
    elif self.rotation == 2:
        if orientation == 0: orientation = 2
        elif orientation == 1: orientation = 3
        elif orientation == 2: orientation = 0
        elif orientation == 3: orientation = 1
    elif self.rotation == 3:
        if orientation == 0: orientation = 3
        elif orientation == 1: orientation = 0
        elif orientation == 2: orientation = 1
        elif orientation == 3: orientation = 2

    size = data & 12
    if size == 8: # big
        t = self.terrain_images[168]
        c_left = (0,3)
        c_right = (8,3)
        c_top = (5,2)
    elif size == 4: # normal
        t = self.terrain_images[169]
        c_left = (-2,2)
        c_right = (8,2)
        c_top = (5,2)
    elif size == 0: # small
        t = self.terrain_images[170]
        c_left = (-3,2)
        c_right = (6,2)
        c_top = (5,2)

    # let's get every texture piece necessary to do this
    stalk = t.copy()
    ImageDraw.Draw(stalk).rectangle((0,0,11,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(stalk).rectangle((12,4,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
    
    top = t.copy() # warning! changes with plant size
    ImageDraw.Draw(top).rectangle((0,7,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(top).rectangle((7,0,16,6),outline=(0,0,0,0),fill=(0,0,0,0))

    side = t.copy() # warning! changes with plant size
    ImageDraw.Draw(side).rectangle((0,0,6,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(side).rectangle((0,0,16,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(side).rectangle((0,14,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
    
    # first compose the block of the cocoa plant
    block = Image.new("RGBA", (24,24), self.bgcolor)
    tmp = self.transform_image_side(side).transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over (block, tmp, c_right,tmp) # right side
    tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over (block, tmp, c_left,tmp) # left side
    tmp = self.transform_image_top(top)
    alpha_over(block, tmp, c_top,tmp)
    if size == 0:
        # fix a pixel hole
        block.putpixel((6,9), block.getpixel((6,10)))

    # compose the cocoa plant
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if orientation in (2,3): # south and west
        tmp = self.transform_image_side(stalk).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, block,(-1,-2), block)
        alpha_over(img, tmp, (4,-2), tmp)
        if orientation == 3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation in (0,1): # north and east
        tmp = self.transform_image_side(stalk.transpose(Image.FLIP_LEFT_RIGHT))
        alpha_over(img, block,(-1,5), block)
        alpha_over(img, tmp, (2,12), tmp)
        if orientation == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img
