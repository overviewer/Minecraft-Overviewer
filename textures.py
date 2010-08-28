import sys
import os
import os.path
import zipfile
from cStringIO import StringIO
import math

import numpy
from PIL import Image, ImageEnhance

def _get_terrain_image():
    if "win" in sys.platform:
        minecraftdir = os.environ['APPDATA']
    else:
        minecraftdir = os.environ['HOME']
    minecraftjar = zipfile.ZipFile(os.path.join(minecraftdir, ".minecraft", "bin", "minecraft.jar"))
    textures = minecraftjar.open("terrain.png")
    buffer = StringIO(textures.read())
    return Image.open(buffer)

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


def _build_texturemap():
    """"""
    t = terrain_images

    # Notes are for things I've left out or will probably have to make special
    # exception for
    top = [-1,1,0,2,16,4,15,17,205,205,237,237,18,19,32,33,
        34,21,52,48,49,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, # Cloths are left out
        -1,-1,-1,64,64,13,12,29,28,23,22,6,6,7,8,35, # Gold/iron blocks? Doublestep? TNT from above?
        36,37,80,-1,65,4,25,101,98,24,43,-1,86,1,1,-1, # Torch from above? leaving out fire. Redstone wire? Crops left out. sign post
        -1,-1,128,16,-1,-1,-1,-1,-1,51,51,-1,-1,1,66,67, # door,ladder left out. Minecart rail orientation
        66,69,72,-1,74 # clay?
        ]
    side = [-1,1,3,2,16,4,15,17,205,205,237,237,18,19,32,33,
        34,21,52,48,49,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,64,64,13,12,29,28,23,22,6,6,7,8,35,
        36,37,80,-1,65,4,25,101,98,24,43,-1,86,1,1,-1,
        -1,-1,128,16,-1,-1,-1,-1,-1,51,51,-1,-1,1,66,67,
        66,69,72,-1,74
        ]
    side[2] = 2

    return (
           [(t[x] if x != -1 else None) for x in top],
           [(_transform_image(t[x]) if x != -1 else None) for x in top],
           [(_transform_image_side(t[x]) if x != -1 else None) for x in side],
           )
# texturemap maps block ids to a 16x16 image that goes on the top face
# perspective_texturemap does the same, except the texture is rotated and shrunk
# shear_texturemap maps block ids to the image that goes on the side of the
# block, sheared appropriately
texturemap, perspective_texturemap, shear_texturemap = _build_texturemap()

def _render_sprite(img):
    """Takes a 16x16 sprite image, and returns a 22x22 image to go in the
    blockmap
    This is for rendering things that are sticking out of the ground, like
    flowers and such
    torches are drawn the same way, but torches that attach to walls are
    handled differently
    """
    pass

def _render_ground_image(img):
    """Takes a 16x16 sprite image and skews it to look like it's on the ground.
    This is for things like mine track and such

    """
    pass

def _build_blockimages():
    """Returns a mapping from blockid to an image of that block in perspective
    The values of the mapping are actually (image in RGB mode, alpha channel)"""
    # This maps block id to the texture that goes on the side of the block
    allimages = []
    for top, side in zip(perspective_texturemap, shear_texturemap):
        if not top or not side:
            allimages.append(None)
            continue
        img = Image.new("RGBA", (24,24))
        
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken the sides slightly. These methods also affect the alpha layer,
        # so save them first (we don't want to "darken" the alpha layer making
        # the block transparent)
        if 1:
            sidealpha = side.split()[3]
            side = ImageEnhance.Brightness(side).enhance(0.9)
            side.putalpha(sidealpha)
            othersidealpha = otherside.split()[3]
            otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
            otherside.putalpha(othersidealpha)

        # Copy on the left side
        img.paste(side, (0,6), side)
        # Copy on the other side
        img.paste(otherside, (12,6), otherside)
        # Copy on the top piece (last so it's on top)
        img.paste(top, (0,0), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the blocks perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))

        allimages.append((img.convert("RGB"), img.split()[3]))
    return allimages

# Maps block images to the appropriate texture on each side. This map is not
# appropriate for all block types
blockmap = _build_blockimages()

