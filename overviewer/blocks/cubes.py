"""
This module defines all simple cube blocks
"""

from collections import namedtuple

from overviewer import chunkrenderer
from overviewer.blockdefinitions import BlockModel, BlockDefinition

CubeTextures = namedtuple("CubeTextures", ["nx","nz","px","pz","ny","py"])

color_map = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
        "silver", "cyan", "purple", "blue", "brown", "green", "red", "black"]

# the Positive-Z face faces south
# the Negative-Z faces north (and is normally hidden from view)
# the Positive-X faces east (and is normally hidden from view)
# the Negative-X faces west 


# Height must be between 0 and 1.0
def make_box(tex, nx=(0, 0, 1, 1), px=(0, 0, 1, 1), ny=(0, 0, 1, 1), py=(0, 0, 1, 1), nz=(0, 0, 1, 1), pz=(0, 0, 1, 1), color=(255, 255, 255, 255), topcolor=None, bottom=0, height=1):
    if topcolor is None:
        topcolor = color
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)

    if not isinstance(tex, CubeTextures):
        if isinstance(tex, tuple):
            tex = CubeTextures._make(tex)
        else:
            tex = CubeTextures._make((tex,)*6)

    model = BlockModel()
    model.vertices = [
        # NX face
        ((0, bottom, 0), (nx[0], nx[1]), xcolor),
        ((0, bottom, 1), (nx[2], nx[1]), xcolor),
        ((0, height, 1), (nx[2], nx[3]), xcolor),
        ((0, height, 0), (nx[0], nx[3]), xcolor),
        
        # NZ face
        ((1, bottom, 0), (nz[0], nz[1]), zcolor),
        ((0, bottom, 0), (nz[2], nz[1]), zcolor),
        ((0, height, 0), (nz[2], nz[3]), zcolor),
        ((1, height, 0), (nz[0], nz[3]), zcolor),
        
        # PX face
        ((1, bottom, 1), (px[0], px[1]), xcolor),
        ((1, bottom, 0), (px[2], px[1]), xcolor),
        ((1, height, 0), (px[2], px[3]), xcolor),
        ((1, height, 1), (px[0], px[3]), xcolor),
        
        # PZ face
        ((0, bottom, 1), (pz[0], pz[1]), zcolor),
        ((1, bottom, 1), (pz[2], pz[1]), zcolor),
        ((1, height, 1), (pz[2], pz[3]), zcolor),
        ((0, height, 1), (pz[0], pz[3]), zcolor),
        
        # NY face
        ((0, bottom, 1), (ny[0], ny[1]), color),
        ((0, bottom, 0), (ny[2], ny[1]), color),
        ((1, bottom, 0), (ny[2], ny[3]), color),
        ((1, bottom, 1), (ny[0], ny[3]), color),
        
        # PY face
        ((0, height, 1), (py[0], py[1]), topcolor),
        ((1, height, 1), (py[2], py[1]), topcolor),
        ((1, height, 0), (py[2], py[3]), topcolor),
        ((0, height, 0), (py[0], py[3]), topcolor),
    ]
    model.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, tex.nx),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, tex.nz),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, tex.px),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, tex.pz),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, tex.ny),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, tex.py),
    ]
    return model

def make_simple(terrain, **kwargs):
    t = (0, 0, 1, 1)
    return make_box(terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs)

def make_custom(terrain, **kwargs):
    return make_box(terrain, **kwargs)

def add(bd):
    # grass
    side = "assets/minecraft/textures/blocks/grass_side.png"
    top = "assets/minecraft/textures/blocks/grass_top.png"
    bottom = "assets/minecraft/textures/blocks/dirt.png"
    bd.add(BlockDefinition(make_simple((side,side,side,side,bottom,top), topcolor=(0,255,0,255))), 2)
    
    # leaves
    leaves = BlockDefinition(transparent=True, datatype=chunkrenderer.BLOCK_DATA_MASKED, dataparameter=0x03)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_oak.png", color=(0, 150, 0, 255)), 0)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_spruce.png", color=(0, 150, 0, 255)), 1)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_birch.png", color=(0, 150, 0, 255)), 2)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_jungle.png", color=(0, 150, 0, 255)), 3)
    bd.add(leaves, 18)
