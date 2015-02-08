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

from PIL import Image
import textures

"""The contents of this file are imported into the namespace of config files.
It also defines the render primitive objects, which are used by the C code.
Each render primitive has a corresponding section of C code, so both places
must be changed simultaneously if you want to make any changes.

"""

class RenderPrimitive(object):
    options = {}
    name = None
    def __init__(self, **kwargs):
        if self.name is None:
            raise RuntimeError("RenderPrimitive cannot be used directly")
        
        self.option_values = {}
        for key, val in kwargs.iteritems():
            if not key in self.options:
                raise ValueError("primitive `{0}' has no option `{1}'".format(self.name, key))
            self.option_values[key] = val
        
        # set up defaults
        for name, (description, default) in self.options.iteritems():
            if not name in self.option_values:
                self.option_values[name] = default

class Base(RenderPrimitive):
    name = "base"
    options = {
        "biomes": ("whether or not to use biomes", True),
    }

class NetherOld(RenderPrimitive):
    name = "netherold"

class Nether(RenderPrimitive):
    name = "nether"

class HeightFading(RenderPrimitive):
    name = "height-fading"
    options = {
        # 128 is *WRONG*, it should be 64. but we're grandfathered in for now
        "sealevel": ("target sea level", 128),
    }
    
    black_color = Image.new("RGB", (24,24), (0,0,0))
    white_color = Image.new("RGB", (24,24), (255,255,255))

class Depth(RenderPrimitive):
    name = "depth"
    options = {
        "min": ("lowest level of blocks to render", 0),
        "max": ("highest level of blocks to render", 255),
    }
    
class Exposed(RenderPrimitive):
    name = "exposed"
    options = {
        "mode": ("0 = exposed blocks only, 1 = unexposed blocks only", 0),
    }
    
class NoFluids(RenderPrimitive):
    name = "no-fluids"

class EdgeLines(RenderPrimitive):
    name = "edge-lines"
    options = {
        "opacity": ("darkness of the edge lines, from 0.0 to 1.0", 0.15),
    }

class Cave(RenderPrimitive):
    name = "cave"
    options = {
        "only_lit": ("only render lit caves", False),
    }

class DepthTinting(RenderPrimitive):
    name = "depth-tinting"
    
    @property
    def depth_colors(self):
        depth_colors = getattr(self, "_depth_colors", [])
        if depth_colors:
            return depth_colors
        r = 255
        g = 0
        b = 0
        for z in range(128):
            depth_colors.append(r)
            depth_colors.append(g)
            depth_colors.append(b)
            
            if z < 32:
                g += 7
            elif z < 64:
                r -= 7
            elif z < 96:
                b += 7
            else:
                g -= 7

        self._depth_colors = depth_colors
        return depth_colors

class Lighting(RenderPrimitive):
    name = "lighting"
    options = {
        "strength": ("how dark to make the shadows, from 0.0 to 1.0", 1.0),
        "night": ("whether to use nighttime skylight settings", False),
        "color": ("whether to use colored light", False),
    }

    @property
    def facemasks(self):
        facemasks = getattr(self, "_facemasks", None)
        if facemasks:
            return facemasks
        
        white = Image.new("L", (24,24), 255)
        
        top = Image.new("L", (24,24), 0)
        left = Image.new("L", (24,24), 0)
        whole = Image.new("L", (24,24), 0)
        
        toppart = textures.Textures.transform_image_top(white)
        leftpart = textures.Textures.transform_image_side(white)
        
        # using the real PIL paste here (not alpha_over) because there is
        # no alpha channel (and it's mode "L")
        top.paste(toppart, (0,0))
        left.paste(leftpart, (0,6))
        right = left.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Manually touch up 6 pixels that leave a gap, like in
        # textures._build_block()
        for x,y in [(13,23), (17,21), (21,19)]:
            right.putpixel((x,y), 255)
        for x,y in [(3,4), (7,2), (11,0)]:
            top.putpixel((x,y), 255)
    
        # special fix for chunk boundary stipple
        for x,y in [(13,11), (17,9), (21,7)]:
            right.putpixel((x,y), 0)
        
        self._facemasks = (top, left, right)
        return self._facemasks

class SmoothLighting(Lighting):
    name = "smooth-lighting"

class ClearBase(RenderPrimitive):
    name = "clear-base"

class Overlay(RenderPrimitive):
    name = "overlay"

    options = {
        'overlay_color' : ('a tuple of (r, g, b, a) for coloring the overlay', None),
    }

    @property
    def whitecolor(self):
        whitecolor = getattr(self, "_whitecolor", None)
        if whitecolor:
            return whitecolor
        white = Image.new("RGBA", (24,24), (255, 255, 255, 255))
        self._whitecolor = white
        return white
    
    @property
    def facemask_top(self):
        facemask_top = getattr(self, "_facemask_top", None)
        if facemask_top:
            return facemask_top
        
        white = Image.new("L", (24,24), 255)
        top = Image.new("L", (24,24), 0)
        toppart = textures.Textures.transform_image_top(white)
        top.paste(toppart, (0,0))
        for x,y in [(3,4), (7,2), (11,0)]:
            top.putpixel((x,y), 255)
        self._facemask_top = top
        return top

class SpawnOverlay(Overlay):
    name = "overlay-spawn"

class SlimeOverlay(Overlay):
    name = "overlay-slime"


class StructureOverlay(Overlay):
    name = "overlay-structure"
    options = {
        'structures': ('a list of ((((relx, rely, relz), blockid), ...), (r, g, b, a)) tuples for coloring minerals',
                       [(((0, 0, 0, 66),  (0, -1, 0, 4)), (255, 0,   0, 255)),
                        (((0, 0, 0, 27),  (0, -1, 0, 4)), (0,   255, 0, 255)),
                        (((0, 0, 0, 28),  (0, -1, 0, 4)), (255, 255, 0, 255)),
                        (((0, 0, 0, 157), (0, -1, 0, 4)), (255, 100, 0, 255)),
                       ]),
    }


class MineralOverlay(Overlay):
    name = "overlay-mineral"
    options = {
        'minerals' : ('a list of (blockid, (r, g, b)) tuples for coloring minerals', None),
    }

class BiomeOverlay(Overlay):
    name = "overlay-biomes"
    options = {
        'biomes' : ('a list of (biome, (r, g, b)) tuples for coloring biomes', None),
        'alpha'  : ('an integer value between 0 (transparent) and 255 (opaque)', None),
    }

class Hide(RenderPrimitive):
    name = "hide"
    options = {
        'blocks' : ('a list of blockids or (blockid, data) tuples of blocks to hide', []),
    }

# Built-in rendermodes for your convenience!
normal = [Base(), EdgeLines()]
lighting = [Base(), EdgeLines(), Lighting()]
smooth_lighting = [Base(), EdgeLines(), SmoothLighting()]
night = [Base(), EdgeLines(), Lighting(night=True)]
smooth_night = [Base(), EdgeLines(), SmoothLighting(night=True)]
netherold = [Base(), EdgeLines(), NetherOld()]
netherold_lighting = [Base(), EdgeLines(), NetherOld(), Lighting()]
netherold_smooth_lighting = [Base(), EdgeLines(), NetherOld(), SmoothLighting()]
nether = [Base(), EdgeLines(), Nether()]
nether_lighting = [Base(), EdgeLines(), Nether(), Lighting()]
nether_smooth_lighting = [Base(), EdgeLines(), Nether(), SmoothLighting()]
cave = [Base(), EdgeLines(), Cave(), DepthTinting()]
