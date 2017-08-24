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

class TextureException(Exception):
    "To be thrown when a texture is not found."
    pass


color_map = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
             "silver", "cyan", "purple", "blue", "brown", "green", "red", "black"]

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

        # see load_image_texture()
        self.texture_cache = {}

        # once we find a jarfile that contains a texture, we cache the ZipFile object here
        self.jar = None
        self.jarpath = ""
    
    ##
    ## pickle support
    ##
    
    def __getstate__(self):
        # we must get rid of the huge image lists, and other images
        attributes = self.__dict__.copy()
        for attr in ['blockmap', 'biome_grass_texture', 'watertexture', 'lavatexture', 'firetexture', 'portaltexture', 'lightcolor', 'grasscolor', 'foliagecolor', 'watercolor', 'texture_cache']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        return attributes
    def __setstate__(self, attrs):
        # regenerate textures, if needed
        for attr, val in attrs.iteritems():
            setattr(self, attr, val)
        self.texture_cache = {}
        if self.generated:
            self.generate()
    
    ##
    ## The big one: generate()
    ##
    
    def generate(self):
        
        # generate biome grass mask
        self.biome_grass_texture = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/grass_top.png"), self.load_image_texture("assets/minecraft/textures/blocks/grass_side_overlay.png"))
        
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
        
        * In the directory textures_path given in the initializer
        * In the resource pack given by textures_path
        * The program dir (same dir as overviewer.py) for extracted textures
        * On Darwin, in /Applications/Minecraft for extracted textures
        * Inside a minecraft client jar. Client jars are searched for in the
          following location depending on platform:
        
            * On Windows, at %APPDATA%/.minecraft/versions/
            * On Darwin, at
                $HOME/Library/Application Support/minecraft/versions
            * at $HOME/.minecraft/versions/

          Only the latest non-snapshot version >1.6 is used

        * The overviewer_core/data/textures dir
        
        In all of these, files are searched for in '.', 'anim', 'misc/', and
        'environment/'.
        
        """
        if verbose: logging.info("Starting search for {0}".format(filename))

        # a list of subdirectories to search for a given file,
        # after the obvious '.'
        search_dirs = ['anim', 'misc', 'environment', 'item', 'item/chests', 'entity', 'entity/chest']
        search_zip_paths = [filename,] + [d + '/' + filename for d in search_dirs]
        def search_dir(base):
            """Search the given base dir for filename, in search_dirs."""
            for path in [os.path.join(base, d, filename) for d in ['',] + search_dirs]:
                if verbose: logging.info('filename: ' + filename + ' ; path: ' + path)
                if os.path.isfile(path):
                    return path

            return None
        if verbose: logging.info('search_zip_paths: ' +  ', '.join(search_zip_paths))

        # A texture path was given on the command line. Search this location
        # for the file first.
        if self.find_file_local_path:
            if os.path.isdir(self.find_file_local_path):
                path = search_dir(self.find_file_local_path)
                if path:
                    if verbose: logging.info("Found %s in '%s'", filename, path)
                    return open(path, mode)
            elif os.path.isfile(self.find_file_local_path):
                # Must be a resource pack. Look for the requested file within
                # it.
                try:
                    pack = zipfile.ZipFile(self.find_file_local_path)
                    for packfilename in search_zip_paths:
                        try:
                            # pack.getinfo() will raise KeyError if the file is
                            # not found.
                            pack.getinfo(packfilename)
                            if verbose: logging.info("Found %s in '%s'", packfilename, self.find_file_local_path)
                            return pack.open(packfilename)
                        except (KeyError, IOError):
                            pass
                        
                        try:
                            # 2nd try with completed path.
                            packfilename = 'assets/minecraft/textures/' + packfilename
                            pack.getinfo(packfilename)
                            if verbose: logging.info("Found %s in '%s'", packfilename, self.find_file_local_path)
                            return pack.open(packfilename)
                        except (KeyError, IOError):
                            pass
                except (zipfile.BadZipfile, IOError):
                    pass

        # If we haven't returned at this point, then the requested file was NOT
        # found in the user-specified texture path or resource pack.
        if verbose: logging.info("Did not find the file in specified texture path")


        # Look in the location of the overviewer executable for the given path
        programdir = util.get_program_path()
        path = search_dir(programdir)
        if path:
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)

        if sys.platform.startswith("darwin"):
            path = search_dir("/Applications/Minecraft")
            if path:
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)

        if verbose: logging.info("Did not find the file in overviewer executable directory")
        if verbose: logging.info("Looking for installed minecraft jar files...")

        # we've sucessfully loaded something from here before, so let's quickly try
        # this before searching again
        if self.jar is not None:
            for jarfilename in search_zip_paths:
                try:
                    self.jar.getinfo(jarfilename)
                    if verbose: logging.info("Found (cached) %s in '%s'", jarfilename, self.jarpath)
                    return self.jar.open(jarfilename)
                except (KeyError, IOError), e:
                    pass

        # Find an installed minecraft client jar and look in it for the texture
        # file we need.
        versiondir = ""
        if "APPDATA" in os.environ and sys.platform.startswith("win"):
            versiondir = os.path.join(os.environ['APPDATA'], ".minecraft", "versions")
        elif "HOME" in os.environ:
            # For linux:
            versiondir = os.path.join(os.environ['HOME'], ".minecraft", "versions")
            if not os.path.exists(versiondir) and sys.platform.startswith("darwin"):
                # For Mac:
                versiondir = os.path.join(os.environ['HOME'], "Library",
                    "Application Support", "minecraft", "versions")

        try:
            if verbose: logging.info("Looking in the following directory: \"%s\"" % versiondir)
            versions = os.listdir(versiondir)
            if verbose: logging.info("Found these versions: {0}".format(versions))
        except OSError:
            # Directory doesn't exist? Ignore it. It will find no versions and
            # fall through the checks below to the error at the bottom of the
            # method.
            versions = []

        most_recent_version = [0,0,0]
        for version in versions:
            # Look for the latest non-snapshot that is at least 1.8. This
            # version is only compatible with >=1.8, and we cannot in general
            # tell if a snapshot is more or less recent than a release.

            # Allow two component names such as "1.8" and three component names
            # such as "1.8.1"
            if version.count(".") not in (1,2):
                continue
            try:
                versionparts = [int(x) for x in version.split(".")]
            except ValueError:
                continue

            if versionparts < [1,8]:
                continue

            if versionparts > most_recent_version:
                most_recent_version = versionparts

        if most_recent_version != [0,0,0]:
            if verbose: logging.info("Most recent version >=1.8.0: {0}. Searching it for the file...".format(most_recent_version))

            jarname = ".".join(str(x) for x in most_recent_version)
            jarpath = os.path.join(versiondir, jarname, jarname + ".jar")

            if os.path.isfile(jarpath):
                jar = zipfile.ZipFile(jarpath)
                for jarfilename in search_zip_paths:
                    try:
                        jar.getinfo(jarfilename)
                        if verbose: logging.info("Found %s in '%s'", jarfilename, jarpath)
                        self.jar, self.jarpath = jar, jarpath
                        return jar.open(jarfilename)
                    except (KeyError, IOError), e:
                        pass

            if verbose: logging.info("Did not find file {0} in jar {1}".format(filename, jarpath))
        else:
            if verbose: logging.info("Did not find any non-snapshot minecraft jars >=1.8.0")
            
        # Last ditch effort: look for the file is stored in with the overviewer
        # installation. We include a few files that aren't included with Minecraft
        # textures. This used to be for things such as water and lava, since
        # they were generated by the game and not stored as images. Nowdays I
        # believe that's not true, but we still have a few files distributed
        # with overviewer.
        if verbose: logging.info("Looking for texture in overviewer_core/data/textures")
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

        raise TextureException("Could not find the textures while searching for '{0}'. Try specifying the 'texturepath' option in your config file.\nSet it to the path to a Minecraft Resource pack.\nAlternately, install the Minecraft client (which includes textures)\nAlso see <http://docs.overviewer.org/en/latest/running/#installing-the-textures>\n(Remember, this version of Overviewer requires a 1.12-compatible resource pack)\n(Also note that I won't automatically use snapshots; you'll have to use the texturepath option to use a snapshot jar)".format(filename))

    def load_image_texture(self, filename):
        # Textures may be animated or in a different resolution than 16x16.  
        # This method will always return a 16x16 image

        img = self.load_image(filename)

        w,h = img.size
        if w != h:
            img = img.crop((0,0,w,w))
        if w != 16:
            img = img.resize((16, 16), Image.ANTIALIAS)

        self.texture_cache[filename] = img
        return img

    def load_image(self, filename):
        """Returns an image object"""

        if filename in self.texture_cache:
            return self.texture_cache[filename]
        
        fileobj = self.find_file(filename)
        buffer = StringIO(fileobj.read())
        img = Image.open(buffer).convert("RGBA")
        self.texture_cache[filename] = img
        return img



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
        except TextureException:
            watertexture = self.load_image_texture("assets/minecraft/textures/blocks/water_still.png")
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
        except TextureException:
            lavatexture = self.load_image_texture("assets/minecraft/textures/blocks/lava_still.png")
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
        except TextureException:
            fireNS = self.load_image_texture("assets/minecraft/textures/blocks/fire_layer_0.png")
            fireEW = self.load_image_texture("assets/minecraft/textures/blocks/fire_layer_1.png")
            firetexture = (fireNS, fireEW)
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
        except TextureException:
            portaltexture = self.load_image_texture("assets/minecraft/textures/blocks/portal.png")
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
            self.grasscolor = list(self.load_image("grass.png").getdata())
        return self.grasscolor

    def load_foliage_color(self):
        """Helper function to load the foliage color texture."""
        if not hasattr(self, "foliagecolor"):
            self.foliagecolor = list(self.load_image("foliage.png").getdata())
        return self.foliagecolor

    #I guess "watercolor" is wrong. But I can't correct as my texture pack don't define water color.
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

    def build_slab_block(self, top, side, upper):
        """From a top texture and a side texture, build a slab block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
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
        if upper:
            delta = 6

        img = Image.new("RGBA", (24,24), self.bgcolor)
        alpha_over(img, side, (0,12 - delta), side)
        alpha_over(img, otherside, (12,12 - delta), otherside)
        alpha_over(img, top, (0,6 - delta), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        if upper:
            for x,y in [(3,4), (7,2), (11,0)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,17), (17,15), (21,13)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))
        else:
            for x,y in [(3,10), (7,8), (11,6)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,23), (17,21), (21,19)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))

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
            if side1 is not None:
                side1 = side1.copy()
                ImageDraw.Draw(side1).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side2 is not None:
                side2 = side2.copy()
                ImageDraw.Draw(side2).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side3 is not None:
                side3 = side3.copy()
                ImageDraw.Draw(side3).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side4 is not None:
                side4 = side4.copy()
                ImageDraw.Draw(side4).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))

        img = Image.new("RGBA", (24,24), self.bgcolor)

        # first back sides
        if side1 is not None :
            side1 = self.transform_image_side(side1)
            side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side.
            sidealpha = side1.split()[3]
            side1 = ImageEnhance.Brightness(side1).enhance(0.9)
            side1.putalpha(sidealpha)        

            alpha_over(img, side1, (0,0), side1)


        if side2 is not None :
            side2 = self.transform_image_side(side2)

            # Darken this side.
            sidealpha2 = side2.split()[3]
            side2 = ImageEnhance.Brightness(side2).enhance(0.8)
            side2.putalpha(sidealpha2)

            alpha_over(img, side2, (12,0), side2)

        if bottom is not None :
            bottom = self.transform_image_top(bottom)
            alpha_over(img, bottom, (0,12), bottom)

        # front sides
        if side3 is not None :
            side3 = self.transform_image_side(side3)

            # Darken this side
            sidealpha = side3.split()[3]
            side3 = ImageEnhance.Brightness(side3).enhance(0.9)
            side3.putalpha(sidealpha)

            alpha_over(img, side3, (0,6), side3)

        if side4 is not None :
            side4 = self.transform_image_side(side4)
            side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side
            sidealpha = side4.split()[3]
            side4 = ImageEnhance.Brightness(side4).enhance(0.8)
            side4.putalpha(sidealpha)

            alpha_over(img, side4, (12,6), side4)

        if top is not None :
            top = self.transform_image_top(top)
            alpha_over(img, top, (0, increment), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))

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

        front = tex.resize((14, 12), Image.ANTIALIAS)
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
def block(blockid=[], top_image=None, side_image=None, **kwargs):
    new_kwargs = {'solid' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if top_image is None:
        raise ValueError("top_image was not provided")
    
    if side_image is None:
        side_image = top_image
    
    @material(blockid=blockid, **new_kwargs)
    def inner_block(self, unused_id, unused_data):
        return self.build_block(self.load_image_texture(top_image), self.load_image_texture(side_image))
    return inner_block

# shortcut function for sprite blocks, defaults to transparent, nodata
def sprite(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if imagename is None:
        raise ValueError("imagename was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_sprite(self, unused_id, unused_data):
        return self.build_sprite(self.load_image_texture(imagename))
    return inner_sprite

# shortcut function for billboard blocks, defaults to transparent, nodata
def billboard(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)
    
    if imagename is None:
        raise ValueError("imagename was not provided")
    
    @material(blockid=blockid, **new_kwargs)
    def inner_billboard(self, unused_id, unused_data):
        return self.build_billboard(self.load_image_texture(imagename))
    return inner_billboard

##
## and finally: actual texture definitions
##

# stone
@material(blockid=1, data=range(7), solid=True)
def stone(self, blockid, data):
    if data == 0: # regular old-school stone
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone.png")
    elif data == 1: # granite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_granite.png")
    elif data == 2: # polished granite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_granite_smooth.png")
    elif data == 3: # diorite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_diorite.png")
    elif data == 4: # polished diorite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_diorite_smooth.png")
    elif data == 5: # andesite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_andesite.png")
    elif data == 6: # polished andesite
        img = self.load_image_texture("assets/minecraft/textures/blocks/stone_andesite_smooth.png")
    return self.build_block(img, img)

@material(blockid=2, data=range(11)+[0x10,], solid=True)
def grass(self, blockid, data):
    # 0x10 bit means SNOW
    side_img = self.load_image_texture("assets/minecraft/textures/blocks/grass_side.png")
    if data & 0x10:
        side_img = self.load_image_texture("assets/minecraft/textures/blocks/grass_side_snowed.png")
    img = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/grass_top.png"), side_img)
    if not data & 0x10:
        alpha_over(img, self.biome_grass_texture, (0, 0), self.biome_grass_texture)
    return img

# dirt
@material(blockid=3, data=range(3), solid=True)
def dirt_blocks(self, blockid, data):
    side_img = self.load_image_texture("assets/minecraft/textures/blocks/dirt.png")
    if data == 0: # normal
        img =  self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/dirt.png"), side_img)
    if data == 1: # grassless
        img = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/dirt.png"), side_img)
    if data == 2: # podzol
        side_img = self.load_image_texture("assets/minecraft/textures/blocks/dirt_podzol_side.png")
        img = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/dirt_podzol_top.png"), side_img)
    return img

# cobblestone
block(blockid=4, top_image="assets/minecraft/textures/blocks/cobblestone.png")

# wooden planks
@material(blockid=5, data=range(6), solid=True)
def wooden_planks(self, blockid, data):
    if data == 0: # normal
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png"), self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png"))
    if data == 1: # pine
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png"),self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png"))
    if data == 2: # birch
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png"),self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png"))
    if data == 3: # jungle wood
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png"),self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png"))
    if data == 4: # acacia
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png"),self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png"))
    if data == 5: # dark oak
        return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png"),self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png"))

@material(blockid=6, data=range(16), transparent=True)
def saplings(self, blockid, data):
    # usual saplings
    tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_oak.png")
    
    if data & 0x3 == 1: # spruce sapling
        tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_spruce.png")
    elif data & 0x3 == 2: # birch sapling
        tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_birch.png")
    elif data & 0x3 == 3: # jungle sapling
        tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_jungle.png")
    elif data & 0x3 == 4: # acacia sapling
        tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_acacia.png")
    elif data & 0x3 == 5: # dark oak/roofed oak/big oak sapling
        tex = self.load_image_texture("assets/minecraft/textures/blocks/sapling_roofed_oak.png")
    return self.build_sprite(tex)

# bedrock
block(blockid=7, top_image="assets/minecraft/textures/blocks/bedrock.png")

@material(blockid=8, data=range(16), fluid=True, transparent=True, nospawn=True)
def water(self, blockid, data):
    watertex = self.load_water()
    return self.build_block(watertex, watertex)

# other water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[9, 20, 79, 95], data=range(512), fluid=(9,), transparent=True, nospawn=True, solid=(79, 20, 95))
def no_inner_surfaces(self, blockid, data):
    if blockid == 9:
        texture = self.load_water()
    elif blockid == 20:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/glass.png")
    elif blockid == 95:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/glass_%s.png" % color_map[data & 0x0f])
    else:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/ice.png")

    # now that we've used the lower 4 bits to get color, shift down to get the 5 bits that encode face hiding
    if blockid != 9: # water doesn't have a shifted pseudodata
        data = data >> 4

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
@material(blockid=12, data=range(2), solid=True)
def sand_blocks(self, blockid, data):
    if data == 0: # normal
        img = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/sand.png"), self.load_image_texture("assets/minecraft/textures/blocks/sand.png"))
    if data == 1: # red
        img = self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/red_sand.png"), self.load_image_texture("assets/minecraft/textures/blocks/red_sand.png"))
    return img

# gravel
block(blockid=13, top_image="assets/minecraft/textures/blocks/gravel.png")
# gold ore
block(blockid=14, top_image="assets/minecraft/textures/blocks/gold_ore.png")
# iron ore
block(blockid=15, top_image="assets/minecraft/textures/blocks/iron_ore.png")
# coal ore
block(blockid=16, top_image="assets/minecraft/textures/blocks/coal_ore.png")

@material(blockid=[17,162], data=range(12), solid=True)
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
    if blockid == 17: # regular wood:
        if wood_type == 0: # normal
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_oak_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_oak.png")
        if wood_type == 1: # spruce
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_spruce_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_spruce.png")
        if wood_type == 2: # birch
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_birch_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_birch.png")
        if wood_type == 3: # jungle wood
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_jungle_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_jungle.png")
    elif blockid == 162: # acacia/dark wood:
        if wood_type == 0: # acacia
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_acacia_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_acacia.png")
        elif wood_type == 1: # dark oak
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_big_oak_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_big_oak.png")
        else:
            top = self.load_image_texture("assets/minecraft/textures/blocks/log_acacia_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/log_acacia.png")

    # choose orientation and paste textures
    if wood_orientation == 0:
        return self.build_block(top, side)
    elif wood_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif wood_orientation == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(270), top)

@material(blockid=[18, 161], data=range(16), transparent=True, solid=True)
def leaves(self, blockid, data):
    # mask out the bits 4 and 8
    # they are used for player placed and check-for-decay blocks
    data = data & 0x7
    t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_oak.png")
    if (blockid, data) == (18, 1): # pine!
        t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_spruce.png")
    elif (blockid, data) == (18, 2): # birth tree
        t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_birch.png")
    elif (blockid, data) == (18, 3): # jungle tree
        t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_jungle.png")
    elif (blockid, data) == (161, 4): # acacia tree
        t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_acacia.png")
    elif (blockid, data) == (161, 5): 
        t = self.load_image_texture("assets/minecraft/textures/blocks/leaves_big_oak.png")
    return self.build_block(t, t)

# sponge
block(blockid=19, top_image="assets/minecraft/textures/blocks/sponge.png")
# lapis lazuli ore
block(blockid=21, top_image="assets/minecraft/textures/blocks/lapis_ore.png")
# lapis lazuli block
block(blockid=22, top_image="assets/minecraft/textures/blocks/lapis_block.png")

# dispensers, dropper, furnaces, and burning furnaces
@material(blockid=[23, 61, 62, 158], data=range(6), solid=True)
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
    
    top = self.load_image_texture("assets/minecraft/textures/blocks/furnace_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/furnace_side.png")
    
    if blockid == 61:
        front = self.load_image_texture("assets/minecraft/textures/blocks/furnace_front_off.png")
    elif blockid == 62:
        front = self.load_image_texture("assets/minecraft/textures/blocks/furnace_front_on.png")
    elif blockid == 23:
        front = self.load_image_texture("assets/minecraft/textures/blocks/dispenser_front_horizontal.png")
        if data == 0: # dispenser pointing down
            return self.build_block(top, top)
        elif data == 1: # dispenser pointing up
            dispenser_top = self.load_image_texture("assets/minecraft/textures/blocks/dispenser_front_vertical.png")
            return self.build_block(dispenser_top, top)
    elif blockid == 158:
        front = self.load_image_texture("assets/minecraft/textures/blocks/dropper_front_horizontal.png")
        if data == 0: # dropper pointing down
            return self.build_block(top, top)
        elif data == 1: # dispenser pointing up
            dropper_top = self.load_image_texture("assets/minecraft/textures/blocks/dropper_front_vertical.png")
            return self.build_block(dropper_top, top)
    
    if data == 3: # pointing west
        return self.build_full_block(top, None, None, side, front)
    elif data == 4: # pointing north
        return self.build_full_block(top, None, None, front, side)
    else: # in any other direction the front can't be seen
        return self.build_full_block(top, None, None, side, side)

# sandstone
@material(blockid=24, data=range(3), solid=True)
def sandstone(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_top.png")
    if data == 0: # normal
        return self.build_block(top, self.load_image_texture("assets/minecraft/textures/blocks/sandstone_normal.png"))
    if data == 1: # hieroglyphic
        return self.build_block(top, self.load_image_texture("assets/minecraft/textures/blocks/sandstone_carved.png"))
    if data == 2: # soft
        return self.build_block(top, self.load_image_texture("assets/minecraft/textures/blocks/sandstone_smooth.png"))
        
# red sandstone
@material(blockid=179, data=range(3), solid=True)
def sandstone(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_top.png")
    if data == 0: # normal
            side = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_normal.png")
            return self.build_full_block(top, None, None, side, side, self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_bottom.png") )
    if data == 1: # hieroglyphic
        return self.build_block(top, self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_carved.png"))
    if data == 2: # soft
        return self.build_block(top, self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_smooth.png"))

# note block
block(blockid=25, top_image="assets/minecraft/textures/blocks/noteblock.png")

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
    
    bed_texture = self.load_image("assets/minecraft/textures/entity/bed/red.png") # FIXME: do tile entity colours
    increment = 8
    left_face = None
    right_face = None
    top_face = None
    if data & 0x8 == 0x8: # head of the bed
        top = bed_texture.copy().crop((6,6,22,22))

        # Composing the side
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,6,6,22)).rotate(90, expand=True)
        # foot of the bed
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (0,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((6,0,22,6)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
            right_face = end
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            left_face = end
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            right_face = side
    
    else: # foot of the bed
        top = bed_texture.copy().crop((6,28,22,44))
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,28,6,44)).rotate(90, expand=True)
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (13,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((22,22,38,28)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
            right_face = end
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            left_face = end
            right_face = side

    top_face = (top_face, increment)
    return self.build_full_block(top_face, None, None, left_face, right_face)

# powered, detector, activator and normal rails
@material(blockid=[27, 28, 66, 157], data=range(14), transparent=True)
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
            raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_golden.png")
            raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")    # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_golden_powered.png")
            raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")    # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7
            
    elif blockid == 28: # detector rail
        raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_detector.png")
        raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")    # leave corners for code simplicity
        
    elif blockid == 66: # normal rail
        raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal.png")
        raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")

    elif blockid == 157: # activator rail
        if data & 0x8 == 0: # unpowered
            raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_activator.png")
            raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")    # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = self.load_image_texture("assets/minecraft/textures/blocks/rail_activator_powered.png")
            raw_corner = self.load_image_texture("assets/minecraft/textures/blocks/rail_normal_turned.png")    # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7
        
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
        piston_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_top_sticky.png").copy()
    else: # normal
        piston_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_top_normal.png").copy()
        
    # other textures
    side_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_side.png").copy()
    back_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_bottom.png").copy()
    interior_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_inner.png").copy()
    
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
        piston_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_top_sticky.png").copy()
    else: # normal
        piston_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_top_normal.png").copy()
    
    # other textures
    side_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_side.png").copy()
    back_t = self.load_image_texture("assets/minecraft/textures/blocks/piston_top_normal.png").copy()
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
sprite(blockid=30, imagename="assets/minecraft/textures/blocks/web.png", nospawn=True)

@material(blockid=31, data=range(3), transparent=True)
def tall_grass(self, blockid, data):
    if data == 0: # dead shrub
        texture = self.load_image_texture("assets/minecraft/textures/blocks/deadbush.png")
    elif data == 1: # tall grass
        texture = self.load_image_texture("assets/minecraft/textures/blocks/tallgrass.png")
    elif data == 2: # fern
        texture = self.load_image_texture("assets/minecraft/textures/blocks/fern.png")
    
    return self.build_billboard(texture)

# dead bush
billboard(blockid=32, imagename="assets/minecraft/textures/blocks/deadbush.png")

@material(blockid=35, data=range(16), solid=True)
def wool(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/wool_colored_%s.png" % color_map[data])
    
    return self.build_block(texture, texture)

# dandelion
sprite(blockid=37, imagename="assets/minecraft/textures/blocks/flower_dandelion.png")

# flowers
@material(blockid=38, data=range(10), transparent=True)
def flower(self, blockid, data):
    flower_map = ["rose", "blue_orchid", "allium", "houstonia", "tulip_red", "tulip_orange",
                  "tulip_white", "tulip_pink", "oxeye_daisy", "dandelion"]
    texture = self.load_image_texture("assets/minecraft/textures/blocks/flower_%s.png" % flower_map[data])

    return self.build_billboard(texture)

# brown mushroom
sprite(blockid=39, imagename="assets/minecraft/textures/blocks/mushroom_brown.png")
# red mushroom
sprite(blockid=40, imagename="assets/minecraft/textures/blocks/mushroom_red.png")
# block of gold
block(blockid=41, top_image="assets/minecraft/textures/blocks/gold_block.png")
# block of iron
block(blockid=42, top_image="assets/minecraft/textures/blocks/iron_block.png")

# double slabs and slabs
# these wooden slabs are unobtainable without cheating, they are still
# here because lots of pre-1.3 worlds use this blocks
@material(blockid=[43, 44, 181, 182, 204, 205], data=range(16), transparent=(44,182,205), solid=True)
def slabs(self, blockid, data):
    if blockid == 44 or blockid == 182: 
        texture = data & 7
    else: # data > 8 are special double slabs
        texture = data

    if blockid == 44 or blockid == 43:
        if texture== 0: # stone slab
            top = self.load_image_texture("assets/minecraft/textures/blocks/stone_slab_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/stone_slab_side.png")
        elif texture== 1: # sandstone slab
            top = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_normal.png")
        elif texture== 2: # wooden slab
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png")
        elif texture== 3: # cobblestone slab
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/cobblestone.png")
        elif texture== 4: # brick
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/brick.png")
        elif texture== 5: # stone brick
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick.png")
        elif texture== 6: # nether brick slab
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/nether_brick.png")
        elif texture== 7: #quartz        
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_side.png")
        elif texture== 8: # special stone double slab with top texture only
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/stone_slab_top.png")
        elif texture== 9: # special sandstone double slab with top texture only
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_top.png")
        else:
            return None

    elif blockid == 182: # single red sandstone slab
        if texture == 0:
            top = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_normal.png")
        else:
            return None

    elif blockid == 181: # double red sandstone slab
        if texture == 0: # red sandstone
            top = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_normal.png")
        elif texture == 8: # 'full' red sandstone (smooth)
            top = side = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_top.png");
        else:
            return None
    elif blockid == 204 or blockid == 205: # purpur slab (single=205 double=204)
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/purpur_block.png");

    
    if blockid == 43 or blockid == 181 or blockid == 204: # double slab
        return self.build_block(top, side)
    
    return self.build_slab_block(top, side, data & 8 == 8);

# brick block
block(blockid=45, top_image="assets/minecraft/textures/blocks/brick.png")
# TNT
block(blockid=46, top_image="assets/minecraft/textures/blocks/tnt_top.png", side_image="assets/minecraft/textures/blocks/tnt_side.png", nospawn=True)
# bookshelf
block(blockid=47, top_image="assets/minecraft/textures/blocks/planks_oak.png", side_image="assets/minecraft/textures/blocks/bookshelf.png")
# moss stone
block(blockid=48, top_image="assets/minecraft/textures/blocks/cobblestone_mossy.png")
# obsidian
block(blockid=49, top_image="assets/minecraft/textures/blocks/obsidian.png")

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
        small = self.load_image_texture("assets/minecraft/textures/blocks/torch_on.png")
    elif blockid == 75: # off redstone torch
        small = self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_off.png")
    else: # on redstone torch
        small = self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_on.png")
        
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
block(blockid=52, top_image="assets/minecraft/textures/blocks/mob_spawner.png", transparent=True)

# wooden, cobblestone, red brick, stone brick, netherbrick, sandstone, spruce, birch, jungle, quartz, and red sandstone stairs.
@material(blockid=[53,67,108,109,114,128,134,135,136,156,163,164,180,203], data=range(128), transparent=True, solid=True, nospawn=True)
def stairs(self, blockid, data):
    # preserve the upside-down bit
    upside_down = data & 0x4

    # find solid quarters within the top or bottom half of the block
    #                   NW           NE           SE           SW
    quarters = [data & 0x8, data & 0x10, data & 0x20, data & 0x40]

    # rotate the quarters so we can pretend northdirection is always upper-left
    numpy.roll(quarters, [0,1,3,2][self.rotation])
    nw,ne,se,sw = quarters

    if blockid == 53: # wooden
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
    elif blockid == 67: # cobblestone
        texture = self.load_image_texture("assets/minecraft/textures/blocks/cobblestone.png").copy()
    elif blockid == 108: # red brick stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/brick.png").copy()
    elif blockid == 109: # stone brick stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick.png").copy()
    elif blockid == 114: # netherbrick stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/nether_brick.png").copy()
    elif blockid == 128: # sandstone stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_normal.png").copy()
    elif blockid == 134: # spruce wood stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png").copy()
    elif blockid == 135: # birch wood  stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png").copy()
    elif blockid == 136: # jungle good stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png").copy()
    elif blockid == 156: # quartz block stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_side.png").copy()
    elif blockid == 163: # acacia wood stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png").copy()
    elif blockid == 164: # dark oak stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png").copy()
    elif blockid == 180: # red sandstone stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_normal.png").copy()
    elif blockid == 203: # purpur stairs
        texture = self.load_image_texture("assets/minecraft/textures/blocks/purpur_block.png").copy()

    outside_l = texture.copy()
    outside_r = texture.copy()
    inside_l = texture.copy()
    inside_r = texture.copy()

    # sandstone, red sandstone, and quartz stairs have special top texture
    if blockid == 128:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/sandstone_top.png").copy()
    elif blockid == 156:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_top.png").copy()
    elif blockid == 180:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/red_sandstone_top.png").copy()


    slab_top = texture.copy()

    push = 8 if upside_down else 0

    def rect(tex,coords):
        ImageDraw.Draw(tex).rectangle(coords,outline=(0,0,0,0),fill=(0,0,0,0))

    # cut out top or bottom half from inner surfaces
    rect(inside_l, (0,8-push,15,15-push))
    rect(inside_r, (0,8-push,15,15-push))

    # cut out missing or obstructed quarters from each surface
    if not nw:
        rect(outside_l, (0,push,7,7+push))
        rect(texture, (0,0,7,7))
    if not nw or sw:
        rect(inside_r, (8,push,15,7+push)) # will be flipped
    if not ne:
        rect(texture, (8,0,15,7))
    if not ne or nw:
        rect(inside_l, (0,push,7,7+push))
    if not ne or se:
        rect(inside_r, (0,push,7,7+push)) # will be flipped
    if not se:
        rect(outside_r, (0,push,7,7+push)) # will be flipped
        rect(texture, (8,8,15,15))
    if not se or sw:
        rect(inside_l, (8,push,15,7+push))
    if not sw:
        rect(outside_l, (8,push,15,7+push))
        rect(outside_r, (8,push,15,7+push)) # will be flipped
        rect(texture, (0,8,7,15))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    if upside_down:
        # top should have no cut-outs after all
        texture = slab_top
    else:
        # render the slab-level surface
        slab_top = self.transform_image_top(slab_top)
        alpha_over(img, slab_top, (0,6))

    # render inner left surface
    inside_l = self.transform_image_side(inside_l)
    # Darken the vertical part of the second step
    sidealpha = inside_l.split()[3]
    # darken it a bit more than usual, looks better
    inside_l = ImageEnhance.Brightness(inside_l).enhance(0.8)
    inside_l.putalpha(sidealpha)
    alpha_over(img, inside_l, (6,3))

    # render inner right surface
    inside_r = self.transform_image_side(inside_r).transpose(Image.FLIP_LEFT_RIGHT)
    # Darken the vertical part of the second step
    sidealpha = inside_r.split()[3]
    # darken it a bit more than usual, looks better
    inside_r = ImageEnhance.Brightness(inside_r).enhance(0.7)
    inside_r.putalpha(sidealpha)
    alpha_over(img, inside_r, (6,3))

    # render outer surfaces
    alpha_over(img, self.build_full_block(texture, None, None, outside_l, outside_r))

    return img

# normal, locked (used in april's fool day), ender and trapped chest
# NOTE:  locked chest used to be id95 (which is now stained glass)
@material(blockid=[54,130,146], data=range(30), transparent = True)
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
    
    if blockid == 130 and not data in [2,3,4,5]: return None
        # iterate.c will only return the ancil data (without pseudo 
        # ancil data) for locked and ender chests, so only 
        # ancilData = 2,3,4,5 are used for this blockids
    
    if data & 24 == 0:
        if blockid == 130: t = self.load_image("ender.png")
        else:
            try:
                t = self.load_image("normal.png")
            except (TextureException, IOError):
                t = self.load_image("chest.png")

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
        t = self.load_image("normal_double.png")
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
        redstone_wire_t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_dust_line0.png").rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(255,0,0))

        redstone_cross_t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_dust_dot.png")
        redstone_cross_t = self.tint_texture(redstone_cross_t,(255,0,0))

        
    else: # unpowered redstone wire
        redstone_wire_t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_dust_line0.png").rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(48,0,0))
        
        redstone_cross_t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_dust_dot.png")
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

    # see iterate.c for where these masks come from
    has_x = (data & 0b1010) > 0
    has_z = (data & 0b0101) > 0
    if has_x and has_z:
        bottom = redstone_cross_t.copy()
        if has_x:
            alpha_over(bottom, redstone_wire_t.copy())
        if has_z:
            alpha_over(bottom, redstone_wire_t.copy().rotate(90))

    else:
        if has_x:
            bottom = redstone_wire_t.copy()
        elif has_z:
            bottom = redstone_wire_t.copy().rotate(90)
        elif data & 0b1111 == 0: 
            bottom = redstone_cross_t.copy()

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
block(blockid=56, top_image="assets/minecraft/textures/blocks/diamond_ore.png")
# diamond block
block(blockid=57, top_image="assets/minecraft/textures/blocks/diamond_block.png")

# crafting table
# needs two different sides
@material(blockid=58, solid=True, nodata=True)
def crafting_table(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/crafting_table_top.png")
    side3 = self.load_image_texture("assets/minecraft/textures/blocks/crafting_table_side.png")
    side4 = self.load_image_texture("assets/minecraft/textures/blocks/crafting_table_front.png")
    
    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

# crops with 8 data values (like wheat)
@material(blockid=59, data=range(8), transparent=True, nospawn=True)
def crops8(self, blockid, data):
    raw_crop = self.load_image_texture("assets/minecraft/textures/blocks/wheat_stage_%d.png" % data)
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# farmland and grass path (15/16 blocks)
@material(blockid=[60,208], data=range(9), solid=True)
def farmland(self, blockid, data):
    if blockid == 60:
        side = self.load_image_texture("assets/minecraft/textures/blocks/dirt.png")
        top = self.load_image_texture("assets/minecraft/textures/blocks/farmland_wet.png")
        if data == 0:
            top = self.load_image_texture("assets/minecraft/textures/blocks/farmland_dry.png")
        # dirt.png is 16 pixels tall, so we need to crop it before building full block
        side = side.crop((0, 1, 16, 16))
        return self.build_full_block((top, 1), side, side, side, side)
			
    else:
        top = self.load_image_texture("assets/minecraft/textures/blocks/grass_path_top.png")
        side = self.load_image_texture("assets/minecraft/textures/blocks/grass_path_side.png")
        # side already has 1 transparent pixel at the top, so it doesn't need to be modified
        # just shift the top image down 1 pixel
        return self.build_full_block((top, 1), side, side, side, side)


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

    texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
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
    texture_stick = self.load_image_texture("assets/minecraft/textures/blocks/log_oak.png")
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
@material(blockid=[64,71,193,194,195,196,197], data=range(32), transparent=True)
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
        if blockid == 64: # classic wood door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_wood_upper.png")
        elif blockid == 71: # iron door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_iron_upper.png")
        elif blockid == 193: # spruce door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_spruce_upper.png")
        elif blockid == 194: # birch door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_birch_upper.png")
        elif blockid == 195: # jungle door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_jungle_upper.png")
        elif blockid == 196: # acacia door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_acacia_upper.png")
        elif blockid == 197: # dark_oak door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_dark_oak_upper.png")
    else: # bottom of the door
        if blockid == 64:
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_wood_lower.png")
        elif blockid == 71: # iron door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_iron_lower.png")
        elif blockid == 193: # spruce door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_spruce_lower.png")
        elif blockid == 194: # birch door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_birch_lower.png")
        elif blockid == 195: # jungle door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_jungle_lower.png")
        elif blockid == 196: # acacia door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_acacia_lower.png")
        elif blockid == 197: # dark_oak door
            raw_door = self.load_image_texture("assets/minecraft/textures/blocks/door_dark_oak_lower.png")
    
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
    raw_texture = self.load_image_texture("assets/minecraft/textures/blocks/ladder.png")

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

    texture = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
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
    t_base = self.load_image_texture("assets/minecraft/textures/blocks/stone.png").copy()

    ImageDraw.Draw(t_base).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # generate the texture for the stick
    stick = self.load_image_texture("assets/minecraft/textures/blocks/lever.png").copy()
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

# wooden and stone pressure plates, and weighted pressure plates
@material(blockid=[70, 72,147,148], data=[0,1], transparent=True)
def pressure_plate(self, blockid, data):
    if blockid == 70: # stone
        t = self.load_image_texture("assets/minecraft/textures/blocks/stone.png").copy()
    elif blockid == 72: # wooden
        t = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
    elif blockid == 147: # light golden
        t = self.load_image_texture("assets/minecraft/textures/blocks/gold_block.png").copy()
    else: # blockid == 148: # heavy iron
        t = self.load_image_texture("assets/minecraft/textures/blocks/iron_block.png").copy()
    
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
block(blockid=[73, 74], top_image="assets/minecraft/textures/blocks/redstone_ore.png")

# stone a wood buttons
@material(blockid=(77,143), data=range(16), transparent=True)
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

    if blockid == 77:
        t = self.load_image_texture("assets/minecraft/textures/blocks/stone.png").copy()
    else:
        t = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()

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
    
    tex = self.load_image_texture("assets/minecraft/textures/blocks/snow.png")
    
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
block(blockid=80, top_image="assets/minecraft/textures/blocks/snow.png")

# cactus
@material(blockid=81, data=range(15), transparent=True, solid=True, nospawn=True)
def cactus(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/cactus_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/cactus_side.png")

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
block(blockid=82, top_image="assets/minecraft/textures/blocks/clay.png")

# sugar cane
@material(blockid=83, data=range(16), transparent=True)
def sugar_cane(self, blockid, data):
    tex = self.load_image_texture("assets/minecraft/textures/blocks/reeds.png")
    return self.build_sprite(tex)

# jukebox
@material(blockid=84, data=range(16), solid=True)
def jukebox(self, blockid, data):
    return self.build_block(self.load_image_texture("assets/minecraft/textures/blocks/jukebox_top.png"), self.load_image_texture("assets/minecraft/textures/blocks/noteblock.png"))

# nether and normal fences
# uses pseudo-ancildata found in iterate.c
@material(blockid=[85, 188, 189, 190, 191, 192, 113], data=range(16), transparent=True, nospawn=True)
def fence(self, blockid, data):
    # no need for rotations, it uses pseudo data.
    # create needed images for Big stick fence
    if blockid == 85: # normal fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
    elif blockid == 188: # spruce fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png").copy()
    elif blockid == 189: # birch fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png").copy()
    elif blockid == 190: # jungle fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png").copy()
    elif blockid == 191: # big/dark oak fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png").copy()
    elif blockid == 192: # acacia oak fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png").copy()
    else: # netherbrick fence
        fence_top = self.load_image_texture("assets/minecraft/textures/blocks/nether_brick.png").copy()
        fence_side = self.load_image_texture("assets/minecraft/textures/blocks/nether_brick.png").copy()
        fence_small_side = self.load_image_texture("assets/minecraft/textures/blocks/nether_brick.png").copy()

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
    top = self.load_image_texture("assets/minecraft/textures/blocks/pumpkin_top.png")
    frontName = "assets/minecraft/textures/blocks/pumpkin_face_off.png" if blockid == 86 else "assets/minecraft/textures/blocks/pumpkin_face_on.png"
    front = self.load_image_texture(frontName)
    side = self.load_image_texture("assets/minecraft/textures/blocks/pumpkin_side.png")

    if data == 0: # pointing west
        img = self.build_full_block(top, None, None, side, front)

    elif data == 1: # pointing north
        img = self.build_full_block(top, None, None, front, side)

    else: # in any other direction the front can't be seen
        img = self.build_full_block(top, None, None, side, side)

    return img

# netherrack
block(blockid=87, top_image="assets/minecraft/textures/blocks/netherrack.png")

# soul sand
block(blockid=88, top_image="assets/minecraft/textures/blocks/soul_sand.png")

# glowstone
block(blockid=89, top_image="assets/minecraft/textures/blocks/glowstone.png")

# portal
@material(blockid=90, data=[1, 2, 4, 5, 8, 10], transparent=True)
def portal(self, blockid, data):
    # no rotations, uses pseudo data
    portaltexture = self.load_portal()
    img = Image.new("RGBA", (24,24), self.bgcolor)

    side = self.transform_image_side(portaltexture)
    otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

    if data in (1,4,5):
        alpha_over(img, side, (5,4), side)

    if data in (2,8,10):
        alpha_over(img, otherside, (5,4), otherside)

    return img

# cake!
@material(blockid=92, data=range(6), transparent=True, nospawn=True)
def cake(self, blockid, data):
    
    # cake textures
    top = self.load_image_texture("assets/minecraft/textures/blocks/cake_top.png").copy()
    side = self.load_image_texture("assets/minecraft/textures/blocks/cake_side.png").copy()
    fullside = side.copy()
    inside = self.load_image_texture("assets/minecraft/textures/blocks/cake_inner.png")
    
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
    top = self.load_image_texture("assets/minecraft/textures/blocks/repeater_off.png") if blockid == 93 else self.load_image_texture("assets/minecraft/textures/blocks/repeater_on.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/stone_slab_side.png")
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
    t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_off.png").copy() if blockid == 93 else self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_on.png").copy()
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

# redstone comparator (149 is inactive, 150 is active)
@material(blockid=[149,150], data=range(16), transparent=True, nospawn=True)
def comparator(self, blockid, data):

    # rotation
    # add self.rotation to the lower 2 bits,  mod 4
    data = data & 0b1100 | (((data & 0b11) + self.rotation) % 4)


    top = self.load_image_texture("assets/minecraft/textures/blocks/comparator_off.png") if blockid == 149 else self.load_image_texture("assets/minecraft/textures/blocks/comparator_on.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/stone_slab_side.png")
    increment = 13

    if (data & 0x3) == 0: # pointing north
        pass
        static_torch = (-3,-1)
        torch = ((0,2),(6,-1))
    
    if (data & 0x3) == 1: # pointing east
        top = top.rotate(270)
        static_torch = (5,-1)
        torch = ((-4,-1),(0,2))

    if (data & 0x3) == 2: # pointing south
        top = top.rotate(180)
        static_torch = (5,3)
        torch = ((0,-4),(-4,-1))

    if (data & 0x3) == 3: # pointing west
        top = top.rotate(90)
        static_torch = (-3,3)
        torch = ((1,-4),(6,-1))


    def build_torch(active):
        # compose a "3d" redstone torch
        t = self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_off.png").copy() if not active else self.load_image_texture("assets/minecraft/textures/blocks/redstone_torch_on.png").copy()
        torch = Image.new("RGBA", (24,24), self.bgcolor)
        
        t_crop = t.crop((2,2,14,14))
        slice = t_crop.copy()
        ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))
        
        alpha_over(torch, slice, (6,4))
        alpha_over(torch, t_crop, (5,5))
        alpha_over(torch, t_crop, (6,5))
        alpha_over(torch, slice, (6,6))

        return torch
    
    active_torch = build_torch(True)
    inactive_torch = build_torch(False)
    back_torch = active_torch if (blockid == 150 or data & 0b1000 == 0b1000) else inactive_torch
    static_torch_img = active_torch if (data & 0b100 == 0b100) else inactive_torch 

    img = self.build_full_block( (top, increment), None, None, side, side)

    alpha_over(img, static_torch_img, static_torch, static_torch_img) 
    alpha_over(img, back_torch, torch[0], back_torch) 
    alpha_over(img, back_torch, torch[1], back_torch) 
    return img
    
    
# trapdoor
# the trapdoor is looks like a sprite when opened, that's not good
@material(blockid=[96,167], data=range(16), transparent=True, nospawn=True)
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
    if blockid == 96:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/trapdoor.png")
    else:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/iron_trapdoor.png")
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
        if data & 0x8 == 0x8: # is a top trapdoor
            img = Image.new("RGBA", (24,24), self.bgcolor)
            t = self.build_full_block((texture, 12), None, None, texture, texture)
            alpha_over(img, t, (0,-9),t)
        else: # is a bottom trapdoor
            img = self.build_full_block((texture, 12), None, None, texture, texture)
    
    return img

# block with hidden silverfish (stone, cobblestone and stone brick)
@material(blockid=97, data=range(3), solid=True)
def hidden_silverfish(self, blockid, data):
    if data == 0: # stone
        t = self.load_image_texture("assets/minecraft/textures/blocks/stone.png")
    elif data == 1: # cobblestone
        t = self.load_image_texture("assets/minecraft/textures/blocks/cobblestone.png")
    elif data == 2: # stone brick
        t = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick.png")
    
    img = self.build_block(t, t)
    
    return img

# stone brick
@material(blockid=98, data=range(4), solid=True)
def stone_brick(self, blockid, data):
    if data == 0: # normal
        t = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick.png")
    elif data == 1: # mossy
        t = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick_mossy.png")
    elif data == 2: # cracked
        t = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick_cracked.png")
    elif data == 3: # "circle" stone brick
        t = self.load_image_texture("assets/minecraft/textures/blocks/stonebrick_carved.png")

    img = self.build_full_block(t, None, None, t, t)

    return img

# huge brown and red mushroom
@material(blockid=[99,100], data= range(11) + [14,15], solid=True)
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
        cap = self.load_image_texture("assets/minecraft/textures/blocks/mushroom_block_skin_brown.png")
    else: # red
        cap = self.load_image_texture("assets/minecraft/textures/blocks/mushroom_block_skin_red.png")

    stem = self.load_image_texture("assets/minecraft/textures/blocks/mushroom_block_skin_stem.png")
    porous = self.load_image_texture("assets/minecraft/textures/blocks/mushroom_block_inside.png")
    
    if data == 0: # fleshy piece
        img = self.build_full_block(porous, None, None, porous, porous)

    if data == 1: # north-east corner
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 2: # east side
        img = self.build_full_block(cap, None, None, porous, cap)

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

    if data == 14: # all cap
        img = self.build_block(cap,cap)

    if data == 15: # all stem
        img = self.build_block(stem,stem)

    return img

# iron bars and glass pane
# TODO glass pane is not a sprite, it has a texture for the side,
# at the moment is not used
@material(blockid=[101,102, 160], data=range(256), transparent=True, nospawn=True)
def panes(self, blockid, data):
    # no rotation, uses pseudo data
    if blockid == 101:
        # iron bars
        t = self.load_image_texture("assets/minecraft/textures/blocks/iron_bars.png")
    elif blockid == 160:
        t = self.load_image_texture("assets/minecraft/textures/blocks/glass_%s.png" % color_map[data & 0xf])
    else:
        # glass panes
        t = self.load_image_texture("assets/minecraft/textures/blocks/glass.png")
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

    # the lower 4 bits encode color, the upper 4 encode adjencies
    data = data >> 4

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
block(blockid=103, top_image="assets/minecraft/textures/blocks/melon_top.png", side_image="assets/minecraft/textures/blocks/melon_side.png", solid=True)

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
    t = self.load_image_texture("assets/minecraft/textures/blocks/melon_stem_disconnected.png").copy()
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
    raw_texture = self.load_image_texture("assets/minecraft/textures/blocks/vine.png")
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
@material(blockid=[107, 183, 184, 185, 186, 187], data=range(8), transparent=True, nospawn=True)
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
    if blockid == 107: # Oak
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png").copy()
    elif blockid == 183: # Spruce
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png").copy()
    elif blockid == 184: # Birch
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png").copy()
    elif blockid == 185: # Jungle
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png").copy()
    elif blockid == 186: # Dark Oak
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png").copy()
    elif blockid == 187: # Acacia
        gate_side = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png").copy()
    else:
        return None

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
block(blockid=110, top_image="assets/minecraft/textures/blocks/mycelium_top.png", side_image="assets/minecraft/textures/blocks/mycelium_side.png")

# lilypad
# At the moment of writing this lilypads has no ancil data and their
# orientation depends on their position on the map. So it uses pseudo
# ancildata.
@material(blockid=111, data=range(4), transparent=True)
def lilypad(self, blockid, data):
    t = self.load_image_texture("assets/minecraft/textures/blocks/waterlily.png").copy()
    if data == 0:
        t = t.rotate(180)
    elif data == 1:
        t = t.rotate(270)
    elif data == 2:
        t = t
    elif data == 3:
        t = t.rotate(90)

    return self.build_full_block(None, None, None, None, None, t)

# nether brick
block(blockid=112, top_image="assets/minecraft/textures/blocks/nether_brick.png")

# nether wart
@material(blockid=115, data=range(4), transparent=True)
def nether_wart(self, blockid, data):
    if data == 0: # just come up
        t = self.load_image_texture("assets/minecraft/textures/blocks/nether_wart_stage_0.png")
    elif data in (1, 2):
        t = self.load_image_texture("assets/minecraft/textures/blocks/nether_wart_stage_1.png")
    else: # fully grown
        t = self.load_image_texture("assets/minecraft/textures/blocks/nether_wart_stage_2.png")
    
    # use the same technic as tall grass
    img = self.build_billboard(t)

    return img

# enchantment table
# TODO there's no book at the moment
@material(blockid=116, transparent=True, nodata=True)
def enchantment_table(self, blockid, data):
    # no book at the moment
    top = self.load_image_texture("assets/minecraft/textures/blocks/enchanting_table_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/enchanting_table_side.png")
    img = self.build_full_block((top, 4), None, None, side, side)

    return img

# brewing stand
# TODO this is a place holder, is a 2d image pasted
@material(blockid=117, data=range(5), transparent=True)
def brewing_stand(self, blockid, data):
    base = self.load_image_texture("assets/minecraft/textures/blocks/brewing_stand_base.png")
    img = self.build_full_block(None, None, None, None, None, base)
    t = self.load_image_texture("assets/minecraft/textures/blocks/brewing_stand.png")
    stand = self.build_billboard(t)
    alpha_over(img,stand,(0,-2))
    return img

# cauldron
@material(blockid=118, data=range(4), transparent=True)
def cauldron(self, blockid, data):
    side = self.load_image_texture("assets/minecraft/textures/blocks/cauldron_side.png")
    top = self.load_image_texture("assets/minecraft/textures/blocks/cauldron_top.png")
    bottom = self.load_image_texture("assets/minecraft/textures/blocks/cauldron_inner.png")
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

# end portal and end_gateway
@material(blockid=[119,209], transparent=True, nodata=True)
def end_portal(self, blockid, data):
    img = Image.new("RGBA", (24,24), self.bgcolor)
    # generate a black texure with white, blue and grey dots resembling stars
    t = Image.new("RGBA", (16,16), (0,0,0,255))
    for color in [(155,155,155,255), (100,255,100,255), (255,255,255,255)]:
        for i in range(6):
            x = randint(0,15)
            y = randint(0,15)
            t.putpixel((x,y),color)
    if blockid == 209: # end_gateway
        return  self.build_block(t, t)
        
    t = self.transform_image_top(t)
    alpha_over(img, t, (0,0), t)

    return img

# end portal frame (data range 8 to get all orientations of filled)
@material(blockid=120, data=range(8), transparent=True)
def end_portal_frame(self, blockid, data):
    # The bottom 2 bits are oritation info but seems there is no
    # graphical difference between orientations
    top = self.load_image_texture("assets/minecraft/textures/blocks/endframe_top.png")
    eye_t = self.load_image_texture("assets/minecraft/textures/blocks/endframe_eye.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/endframe_side.png")
    img = self.build_full_block((top, 4), None, None, side, side)
    if data & 0x4 == 0x4: # ender eye on it
        # generate the eye
        eye_t = self.load_image_texture("assets/minecraft/textures/blocks/endframe_eye.png").copy()
        eye_t_s = self.load_image_texture("assets/minecraft/textures/blocks/endframe_eye.png").copy()
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
block(blockid=121, top_image="assets/minecraft/textures/blocks/end_stone.png")

# dragon egg
# NOTE: this isn't a block, but I think it's better than nothing
block(blockid=122, top_image="assets/minecraft/textures/blocks/dragon_egg.png")

# inactive redstone lamp
block(blockid=123, top_image="assets/minecraft/textures/blocks/redstone_lamp_off.png")

# active redstone lamp
block(blockid=124, top_image="assets/minecraft/textures/blocks/redstone_lamp_on.png")

# daylight sensor.  
@material(blockid=151, transparent=True)
def daylight_sensor(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/daylight_detector_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/daylight_detector_side.png")

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
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, side, (0,12), side)
    alpha_over(img, otherside, (12,12), otherside)
    alpha_over(img, top, (0,6), top)
    
    return img


# wooden double and normal slabs
# these are the new wooden slabs, blockids 43 44 still have wooden
# slabs, but those are unobtainable without cheating
@material(blockid=[125, 126], data=range(16), transparent=(44,), solid=True)
def wooden_slabs(self, blockid, data):
    texture = data & 7
    if texture== 0: # oak 
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_oak.png")
    elif texture== 1: # spruce
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_spruce.png")
    elif texture== 2: # birch
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_birch.png")
    elif texture== 3: # jungle
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_jungle.png")
    elif texture== 4: # acacia
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_acacia.png")
    elif texture== 5: # dark wood
        top = side = self.load_image_texture("assets/minecraft/textures/blocks/planks_big_oak.png")
    else:
        return None
    
    if blockid == 125: # double slab
        return self.build_block(top, side)
    
    return self.build_slab_block(top, side, data & 8 == 8);

# emerald ore
block(blockid=129, top_image="assets/minecraft/textures/blocks/emerald_ore.png")

# emerald block
block(blockid=133, top_image="assets/minecraft/textures/blocks/emerald_block.png")

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
        t = self.load_image_texture("assets/minecraft/textures/blocks/cocoa_stage_2.png")
        c_left = (0,3)
        c_right = (8,3)
        c_top = (5,2)
    elif size == 4: # normal
        t = self.load_image_texture("assets/minecraft/textures/blocks/cocoa_stage_1.png")
        c_left = (-2,2)
        c_right = (8,2)
        c_top = (5,2)
    elif size == 0: # small
        t = self.load_image_texture("assets/minecraft/textures/blocks/cocoa_stage_0.png")
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

# command block
@material(blockid=[137,210,211], solid=True, nodata=True)
def command_block(self, blockid, data):
    if blockid == 210:
        front = self.load_image_texture("assets/minecraft/textures/blocks/repeating_command_block_front.png")
        side = self.load_image_texture("assets/minecraft/textures/blocks/repeating_command_block_side.png")
        back = self.load_image_texture("assets/minecraft/textures/blocks/repeating_command_block_back.png")
    elif blockid == 211:
        front = self.load_image_texture("assets/minecraft/textures/blocks/chain_command_block_front.png")
        side = self.load_image_texture("assets/minecraft/textures/blocks/chain_command_block_side.png")
        back = self.load_image_texture("assets/minecraft/textures/blocks/chain_command_block_back.png")
    else:
        front = self.load_image_texture("assets/minecraft/textures/blocks/command_block_front.png")
        side = self.load_image_texture("assets/minecraft/textures/blocks/command_block_side.png")
        back = self.load_image_texture("assets/minecraft/textures/blocks/command_block_back.png")
    return self.build_full_block(side, side, back, front, side)

# beacon block
# at the moment of writing this, it seems the beacon block doens't use
# the data values
@material(blockid=138, transparent=True, nodata = True)
def beacon(self, blockid, data):
    # generate the three pieces of the block
    t = self.load_image_texture("assets/minecraft/textures/blocks/glass.png")
    glass = self.build_block(t,t)
    t = self.load_image_texture("assets/minecraft/textures/blocks/obsidian.png")
    obsidian = self.build_full_block((t,12),None, None, t, t)
    obsidian = obsidian.resize((20,20), Image.ANTIALIAS)
    t = self.load_image_texture("assets/minecraft/textures/blocks/beacon.png")
    crystal = self.build_block(t,t)
    crystal = crystal.resize((16,16),Image.ANTIALIAS)
    
    # compose the block
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, obsidian, (2, 4), obsidian)
    alpha_over(img, crystal, (4,3), crystal)
    alpha_over(img, glass, (0,0), glass)
    
    return img

# cobblestone and mossy cobblestone walls, chorus plants
# one additional bit of data value added for mossy and cobblestone
@material(blockid=[139, 199], data=range(32), transparent=True, nospawn=True)
def cobblestone_wall(self, blockid, data):
    # chorus plants
    if blockid == 199:
        t = self.load_image_texture("assets/minecraft/textures/blocks/chorus_plant.png").copy()
    # no rotation, uses pseudo data
    elif data & 0b10000 == 0:
        # cobblestone
        t = self.load_image_texture("assets/minecraft/textures/blocks/cobblestone.png").copy()
    else:
        # mossy cobblestone
        t = self.load_image_texture("assets/minecraft/textures/blocks/cobblestone_mossy.png").copy()

    wall_pole_top = t.copy()
    wall_pole_side = t.copy()
    wall_side_top = t.copy()
    wall_side = t.copy()
    # _full is used for walls without pole
    wall_side_top_full = t.copy()
    wall_side_full = t.copy()

    # generate the textures of the wall
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(wall_pole_side).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_side).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the pole
    wall_pole_side = self.transform_image_side(wall_pole_side)
    wall_pole_other_side = wall_pole_side.transpose(Image.FLIP_LEFT_RIGHT)
    wall_pole_top = self.transform_image_top(wall_pole_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_pole_side.split()[3]
    wall_pole_side = ImageEnhance.Brightness(wall_pole_side).enhance(0.8)
    wall_pole_side.putalpha(sidealpha)
    othersidealpha = wall_pole_other_side.split()[3]
    wall_pole_other_side = ImageEnhance.Brightness(wall_pole_other_side).enhance(0.7)
    wall_pole_other_side.putalpha(othersidealpha)

    # Compose the wall pole
    wall_pole = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(wall_pole,wall_pole_side, (3,4),wall_pole_side)
    alpha_over(wall_pole,wall_pole_other_side, (9,4),wall_pole_other_side)
    alpha_over(wall_pole,wall_pole_top, (0,0),wall_pole_top)

    # create the sides and the top of a wall attached to a pole
    ImageDraw.Draw(wall_side).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    # full version, without pole
    ImageDraw.Draw(wall_side_full).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # compose the sides of a wall atached to a pole
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side = self.transform_image_side(wall_side)
    wall_side_top = self.transform_image_top(wall_side_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side.split()[3]
    wall_side = ImageEnhance.Brightness(wall_side).enhance(0.7)
    wall_side.putalpha(sidealpha)

    alpha_over(tmp,wall_side, (0,0),wall_side)
    alpha_over(tmp,wall_side_top, (-5,3),wall_side_top)
    wall_side = tmp
    wall_other_side = wall_side.transpose(Image.FLIP_LEFT_RIGHT)

    # compose the sides of the full wall
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side_full = self.transform_image_side(wall_side_full)
    wall_side_top_full = self.transform_image_top(wall_side_top_full.rotate(90))

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side_full.split()[3]
    wall_side_full = ImageEnhance.Brightness(wall_side_full).enhance(0.7)
    wall_side_full.putalpha(sidealpha)

    alpha_over(tmp,wall_side_full, (4,0),wall_side_full)
    alpha_over(tmp,wall_side_top_full, (3,-4),wall_side_top_full)
    wall_side_full = tmp
    wall_other_side_full = wall_side_full.transpose(Image.FLIP_LEFT_RIGHT)

    # Create img to compose the wall
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # Position wall imgs around the wall bit stick
    pos_top_left = (-5,-2)
    pos_bottom_left = (-8,4)
    pos_top_right = (5,-3)
    pos_bottom_right = (7,4)
    
    # +x axis points top right direction
    # +y axis points bottom right direction
    # There are two special cases for wall without pole.
    # Normal case: 
    # First compose the walls in the back of the image, 
    # then the pole and then the walls in the front.
    if (data == 0b1010) or (data == 0b11010):
        alpha_over(img, wall_other_side_full,(0,2), wall_other_side_full)
    elif (data == 0b0101) or (data == 0b10101):
        alpha_over(img, wall_side_full,(0,2), wall_side_full)
    else:
        if (data & 0b0001) == 1:
            alpha_over(img,wall_side, pos_top_left,wall_side)                # top left
        if (data & 0b1000) == 8:
            alpha_over(img,wall_other_side, pos_top_right,wall_other_side)    # top right

        alpha_over(img,wall_pole,(0,0),wall_pole)
            
        if (data & 0b0010) == 2:
            alpha_over(img,wall_other_side, pos_bottom_left,wall_other_side)      # bottom left    
        if (data & 0b0100) == 4:
            alpha_over(img,wall_side, pos_bottom_right,wall_side)                  # bottom right
    
    return img

# carrots, potatoes
@material(blockid=[141,142], data=range(8), transparent=True, nospawn=True)
def crops4(self, blockid, data):
    # carrots and potatoes have 8 data, but only 4 visual stages
    stage = {0:0,
             1:0,
             2:1,
             3:1,
             4:2,
             5:2,
             6:2,
             7:3}[data]
    if blockid == 141: # carrots
        raw_crop = self.load_image_texture("assets/minecraft/textures/blocks/carrots_stage_%d.png" % stage)
    else: # potatoes
        raw_crop = self.load_image_texture("assets/minecraft/textures/blocks/potatoes_stage_%d.png" % stage)
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# anvils
@material(blockid=145, data=range(12), transparent=True)
def anvil(self, blockid, data):
    
    # anvils only have two orientations, invert it for rotations 1 and 3
    orientation = data & 0x1
    if self.rotation in (1,3):
        if orientation == 1:
            orientation = 0
        else:
            orientation = 1

    # get the correct textures
    # the bits 0x4 and 0x8 determine how damaged is the anvil
    if (data & 0xc) == 0: # non damaged anvil
        top = self.load_image_texture("assets/minecraft/textures/blocks/anvil_top_damaged_0.png")
    elif (data & 0xc) == 0x4: # slightly damaged
        top = self.load_image_texture("assets/minecraft/textures/blocks/anvil_top_damaged_1.png")
    elif (data & 0xc) == 0x8: # very damaged
        top = self.load_image_texture("assets/minecraft/textures/blocks/anvil_top_damaged_2.png")
    # everything else use this texture
    big_side = self.load_image_texture("assets/minecraft/textures/blocks/anvil_base.png").copy()
    small_side = self.load_image_texture("assets/minecraft/textures/blocks/anvil_base.png").copy()
    base = self.load_image_texture("assets/minecraft/textures/blocks/anvil_base.png").copy()
    small_base = self.load_image_texture("assets/minecraft/textures/blocks/anvil_base.png").copy()
    
    # cut needed patterns
    ImageDraw.Draw(big_side).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((0,0,2,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((13,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(base).rectangle((0,0,15,15),outline=(0,0,0,0))
    ImageDraw.Draw(base).rectangle((1,1,14,14),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((0,0,15,15),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((1,1,14,14),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((2,2,13,13),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((3,3,12,12),outline=(0,0,0,0))
    
    # check orientation and compose the anvil
    if orientation == 1: # bottom-left top-right
        top = top.rotate(90)
        left_side = small_side
        left_pos = (1,7)
        right_side = big_side
        right_pos = (10,5)
    else: # top-left bottom-right
        right_side = small_side
        right_pos = (12,7)
        left_side = big_side
        left_pos = (3,5)
    
    img = Image.new("RGBA", (24,24), self.bgcolor)
    
    # darken sides
    alpha = big_side.split()[3]
    big_side = ImageEnhance.Brightness(big_side).enhance(0.8)
    big_side.putalpha(alpha)
    alpha = small_side.split()[3]
    small_side = ImageEnhance.Brightness(small_side).enhance(0.9)
    small_side.putalpha(alpha)
    alpha = base.split()[3]
    base_d = ImageEnhance.Brightness(base).enhance(0.8)
    base_d.putalpha(alpha)
    
    # compose
    base = self.transform_image_top(base)
    base_d = self.transform_image_top(base_d)
    small_base = self.transform_image_top(small_base)
    top = self.transform_image_top(top)
    
    alpha_over(img, base_d, (0,12), base_d)
    alpha_over(img, base_d, (0,11), base_d)
    alpha_over(img, base_d, (0,10), base_d)
    alpha_over(img, small_base, (0,10), small_base)
    
    alpha_over(img, top, (0,0), top)
    
    left_side = self.transform_image_side(left_side)
    right_side = self.transform_image_side(right_side).transpose(Image.FLIP_LEFT_RIGHT)
    
    alpha_over(img, left_side, left_pos, left_side)
    alpha_over(img, right_side, right_pos, right_side)
    
    return img


# block of redstone
block(blockid=152, top_image="assets/minecraft/textures/blocks/redstone_block.png")

# nether quartz ore
block(blockid=153, top_image="assets/minecraft/textures/blocks/quartz_ore.png")

# block of quartz
@material(blockid=155, data=range(5), solid=True)
def quartz_block(self, blockid, data):
    
    if data in (0,1): # normal and chiseled quartz block
        if data == 0:
            top = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_side.png")
        else:
            top = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_chiseled_top.png")
            side = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_chiseled.png")    
        return self.build_block(top, side)
    
    # pillar quartz block with orientation
    top = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_lines_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/quartz_block_lines.png").copy()
    if data == 2: # vertical
        return self.build_block(top, side)
    elif data == 3: # north-south oriented
        if self.rotation in (0,2):
            return self.build_full_block(side, None, None, top, side.rotate(90))
        return self.build_full_block(side.rotate(90), None, None, side.rotate(90), top)
        
    elif data == 4: # east-west oriented
        if self.rotation in (0,2):
            return self.build_full_block(side.rotate(90), None, None, side.rotate(90), top)
        return self.build_full_block(side, None, None, top, side.rotate(90))
    
# hopper
@material(blockid=154, data=range(4), transparent=True)
def hopper(self, blockid, data):
    #build the top
    side = self.load_image_texture("assets/minecraft/textures/blocks/hopper_outside.png")
    top = self.load_image_texture("assets/minecraft/textures/blocks/hopper_top.png")
    bottom = self.load_image_texture("assets/minecraft/textures/blocks/hopper_inside.png")
    hop_top = self.build_full_block((top,10), side, side, side, side, side)

    #build a solid block for mid/top
    hop_mid = self.build_full_block((top,5), side, side, side, side, side)
    hop_bot = self.build_block(side,side)

    hop_mid = hop_mid.resize((17,17),Image.ANTIALIAS)
    hop_bot = hop_bot.resize((10,10),Image.ANTIALIAS)
    
    #compose the final block
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, hop_bot, (7,14), hop_bot)
    alpha_over(img, hop_mid, (3,3), hop_mid)
    alpha_over(img, hop_top, (0,-6), hop_top)

    return img

# slime block
block(blockid=165, top_image="assets/minecraft/textures/blocks/slime.png")

# prismarine block
@material(blockid=168, data=range(3), solid=True)
def prismarine_block(self, blockid, data):

   if data == 0: # prismarine
       t = self.load_image_texture("assets/minecraft/textures/blocks/prismarine_rough.png")
   elif data == 1: # prismarine bricks
       t = self.load_image_texture("assets/minecraft/textures/blocks/prismarine_bricks.png")
   elif data == 2: # dark prismarine
       t = self.load_image_texture("assets/minecraft/textures/blocks/prismarine_dark.png")

   img = self.build_block(t, t)

   return img

# sea lantern 
block(blockid=169, top_image="assets/minecraft/textures/blocks/sea_lantern.png")

# hay block
@material(blockid=170, data=range(9), solid=True)
def hayblock(self, blockid, data):
    top = self.load_image_texture("assets/minecraft/textures/blocks/hay_block_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/hay_block_side.png")

    if self.rotation == 1:
        if data == 4: data = 8
        elif data == 8: data = 4
    elif self.rotation == 3:
        if data == 4: data = 8
        elif data == 8: data = 4

    # choose orientation and paste textures
    if data == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif data == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(90), top)
    else:
        return self.build_block(top, side)


# carpet - wool block that's small?
@material(blockid=171, data=range(16), transparent=True)
def carpet(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/wool_colored_%s.png" % color_map[data])

    return self.build_full_block((texture,15),texture,texture,texture,texture)

#clay block
block(blockid=172, top_image="assets/minecraft/textures/blocks/hardened_clay.png")

#stained hardened clay
@material(blockid=159, data=range(16), solid=True)
def stained_clay(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/hardened_clay_stained_%s.png" % color_map[data])

    return self.build_block(texture,texture)

#coal block
block(blockid=173, top_image="assets/minecraft/textures/blocks/coal_block.png")

# packed ice block
block(blockid=174, top_image="assets/minecraft/textures/blocks/ice_packed.png")

@material(blockid=175, data=range(16), transparent=True)
def flower(self, blockid, data):
    double_plant_map = ["sunflower", "syringa", "grass", "fern", "rose", "paeonia", "paeonia", "paeonia"]
    plant = double_plant_map[data & 0x7]

    if data & 0x8:
        part = "top"
    else:
        part = "bottom"

    png = "assets/minecraft/textures/blocks/double_plant_%s_%s.png" % (plant,part)
    texture = self.load_image_texture(png)
    img = self.build_billboard(texture)

    #sunflower top
    if data == 8:
        bloom_tex = self.load_image_texture("assets/minecraft/textures/blocks/double_plant_sunflower_front.png")
        alpha_over(img, bloom_tex.resize((14, 11), Image.ANTIALIAS), (5,5))

    return img

# chorus flower
@material(blockid=200, data=range(6), solid=True)
def chorus_flower(self, blockid, data):
    # aged 5, dead
    if data == 5:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/chorus_flower_dead.png")
    else:
        texture = self.load_image_texture("assets/minecraft/textures/blocks/chorus_flower.png")

    return self.build_block(texture,texture)

# purpur block
block(blockid=201, top_image="assets/minecraft/textures/blocks/purpur_block.png")

# purpur pilar
@material(blockid=202, data=range(12) , solid=True)
def purpur_pillar(self, blockid, data):
    pillar_orientation = data & 12
    top=self.load_image_texture("assets/minecraft/textures/blocks/purpur_pillar_top.png")
    side=self.load_image_texture("assets/minecraft/textures/blocks/purpur_pillar.png")
    if pillar_orientation == 0: # east-west orientation
        return self.build_block(top, side)
    elif pillar_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif pillar_orientation == 8: # north-south orientation

        return self.build_full_block(side, None, None, side.rotate(270), top)

# end brick
block(blockid=206, top_image="assets/minecraft/textures/blocks/end_bricks.png")

# frosted ice
@material(blockid=212, data=range(4), solid=True)
def frosted_ice(self, blockid, data):
    img = self.load_image_texture("assets/minecraft/textures/blocks/frosted_ice_%d.png" % data)
    return self.build_block(img, img)

# magma block
block(blockid=213, top_image="assets/minecraft/textures/blocks/magma.png")

# nether wart block
block(blockid=214, top_image="assets/minecraft/textures/blocks/nether_wart_block.png")

# red nether brick
block(blockid=215, top_image="assets/minecraft/textures/blocks/red_nether_brick.png")

@material(blockid=216, data=range(12), solid=True)
def boneblock(self, blockid, data):
    # extract orientation
    boneblock_orientation = data & 12
    if self.rotation == 1:
        if boneblock_orientation == 4: boneblock_orientation = 8
        elif boneblock_orientation == 8: boneblock_orientation = 4
    elif self.rotation == 3:
        if boneblock_orientation == 4: boneblock_orientation = 8
        elif boneblock_orientation == 8: boneblock_orientation = 4

    top = self.load_image_texture("assets/minecraft/textures/blocks/bone_block_top.png")
    side = self.load_image_texture("assets/minecraft/textures/blocks/bone_block_side.png")

    # choose orientation and paste textures
    if boneblock_orientation == 0:
        return self.build_block(top, side)
    elif boneblock_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif boneblock_orientation == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(270), top)

# observer
@material(blockid=218, data=range(6), solid=True, nospawn=True)
def observer(self, blockid, data):
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

    front = self.load_image_texture("assets/minecraft/textures/blocks/observer_front.png").copy()
    side = self.load_image_texture("assets/minecraft/textures/blocks/observer_side.png").copy()
    back = self.load_image_texture("assets/minecraft/textures/blocks/observer_back.png").copy()
    top = self.load_image_texture("assets/minecraft/textures/blocks/observer_top.png").copy()

    if data == 0: # down
        side = side.rotate(90)
        img = self.build_full_block(back, None, None, side, top)
    elif data == 1: # up
        side = side.rotate(90)
        img = self.build_full_block(front.rotate(180), None, None, side, top.rotate(180))
    elif data == 2: # east
        img = self.build_full_block(top.rotate(180), None, None, side, back)
    elif data == 3: # west
        img = self.build_full_block(top, None, None, side, front)
    elif data == 4: # north
        img = self.build_full_block(top.rotate(270), None, None, front, side)
    elif data == 5: # south
        img = self.build_full_block(top.rotate(90), None, None, back, side)

    return img

# shulker box
@material(blockid=range(219,235), data=range(6), solid=True, nospawn=True)
def shulker_box(self, blockid, data):
    # first, do the rotation if needed
    data = data & 7
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

    color = color_map[blockid - 219]
    shulker_t = self.load_image_texture("assets/minecraft/textures/entity/shulker/shulker_%s.png" % color).copy()
    w,h = shulker_t.size
    res = w / 4
    # Cut out the parts of the shulker texture we need for the box
    top = shulker_t.crop((res,0,res*2,res))
    bottom = shulker_t.crop((res*2,int(res*1.75),res*3,int(res*2.75)))
    side_top = shulker_t.crop((0,res,res,int(res*1.75)))
    side_bottom = shulker_t.crop((0,int(res*2.75),res,int(res*3.25)))
    side = Image.new('RGBA', (res,res))
    side.paste(side_top, (0,0), side_top)
    side.paste(side_bottom, (0,res/2), side_bottom)

    if data == 0: # down
        side = side.rotate(180)
        img = self.build_full_block(bottom, None, None, side, side)
    elif data == 1: # up
        img = self.build_full_block(top, None, None, side, side)
    elif data == 2: # east
        img = self.build_full_block(side, None, None, side.rotate(90), bottom)
    elif data == 3: # west
        img = self.build_full_block(side.rotate(180), None, None, side.rotate(270), top)
    elif data == 4: # north
        img = self.build_full_block(side.rotate(90), None, None, top, side.rotate(270))
    elif data == 5: # south
        img = self.build_full_block(side.rotate(270), None, None, bottom, side.rotate(90))

    return img

# structure block
@material(blockid=255, data=range(4), solid=True)
def structure_block(self, blockid, data):
    if data == 0:
        img = self.load_image_texture("assets/minecraft/textures/blocks/structure_block_save.png")
    elif data == 1:
        img = self.load_image_texture("assets/minecraft/textures/blocks/structure_block_load.png")
    elif data == 2:
        img = self.load_image_texture("assets/minecraft/textures/blocks/structure_block_corner.png")
    elif data == 3:
        img = self.load_image_texture("assets/minecraft/textures/blocks/structure_block_data.png")
    return self.build_block(img, img)

# beetroots
@material(blockid=207, data=range(4), transparent=True, nospawn=True)
def crops(self, blockid, data):
    raw_crop = self.load_image_texture("assets/minecraft/textures/blocks/beetroots_stage_%d.png" % data)
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# Concrete
@material(blockid=251, data=range(16), solid=True)
def concrete(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/concrete_%s.png" % color_map[data])
    return self.build_block(texture, texture)

# Concrete Powder
@material(blockid=252, data=range(16), solid=True)
def concrete(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/concrete_powder_%s.png" % color_map[data])
    return self.build_block(texture, texture)

# Glazed Terracotta
@material(blockid=range(235,251), data=range(8), solid=True)
def glazed_terracotta(self, blockid, data):
    texture = self.load_image_texture("assets/minecraft/textures/blocks/glazed_terracotta_%s.png" % color_map[blockid - 235])
    glazed_terracotta_orientation = data & 3
    
    # Glazed Terracotta rotations are need seperate handling for each render direction

    if self.rotation == 0: # rendering north upper-left
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))

    elif self.rotation == 1: # north upper-right
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_block(texture, texture)
            

    elif self.rotation == 2: # north lower-right
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
            
    elif self.rotation == 3: # north lower-left
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
    