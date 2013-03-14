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

from itertools import tee, izip

import OIL
from . import chunkrenderer

"""

blockdefinitions.py contains the BlockDefinitions class, which acts as
a method of registering BlockDefinition objects to be used by the C
chunk renderer. BlockDefinitions objects must be compile()'d before
they can be used by the C renderer.

It also provides a default BlockDefinitions object pre-loaded with
vanilla Minecraft blocks.

"""

class BlockDefinition(object):
    """
    The C code expects this object to have the following attributes:
    
     * vertices: a list of (coordinate, texcoords, color) tuples,
       where coordinate is a 3-tuple of numbers, texcoords is a
       2-tuple, and color is a 4-tuple of integers between 0 and 255.
     
     * triangles: a list of ((i, j, k), type) tuples where i, j, k are
       indexes into the vertices list, and type is one of the
       chunkrenderer.FACE_TYPE_*
     
     * tex: an OIL.Image to use as the block texture.
     
     * transparent, solid, fluid, nospawn, and nodata: boolean values
    
    These are used by chunkrenderer.compile_block_definitions().
    
    Note that in this implementation, `triangles` is generated from
    `faces`, which is similarly-structured but may contain more than 3
    indexes per face.
    """
    
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.tex = None
        self.transparent = False
        self.solid = True
        self.fluid = False
        self.nospawn = False
        self.nodata = True

    @property
    def triangles(self):
        for indices, facetype in self.faces:
            first = indices[0]
            a, b = tee(indices[1:])
            b.next()
            for i, j in izip(a, b):
                yield ((first, i, j), facetype)

class BlockDefinitions(object):
    """
    This object is a container for BlockDefinition objects, which can
    be added by add() or the other convenience functions. The C code
    expects it to have the following attributes:
    
     * blocks: a dictionary mapping (blockid, data) tuples to
       BlockDefinition objects.
     
     * max_blockid, max_data: integers that are larger than all the
       currently-registered blockids or data values.
     
    These are used by chunkrenderer.compile_block_definitions().
    """
    
    def __init__(self):
        self.blocks = {}
        self.max_blockid = 0
        self.max_data = 0
        self.dirty = True

    def compile(self):
        if not self.dirty:
            return
        chunkrenderer.compile_block_definitions(self)
        self.dirty = False
    
    def add(self, blockdef, blockid, data=0):
        try:
            blockid = iter(blockid)
        except TypeError:
            blockid = [blockid]
        try:
            data = iter(data)
        except TypeError:
            data = [data]
        for b in blockid:
            for d in data:
                self.blocks[(b, d)] = blockdef
                self.max_blockid = max(self.max_blockid, b + 1)
                self.max_data = max(self.max_data, d + 1)

def make_box(tex, nx=(0, 0), px=(0, 0), ny=(0, 0), py=(0, 0), nz=(0, 0), pz=(0, 0), color=(255, 255, 255, 255), topcolor=None):
    if topcolor is None:
        topcolor = color
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)
    bd = BlockDefinition()
    bd.vertices = [
        ((0, 0, 0), (0.000000 + nx[0] / 16.0, 0.000000 + (15 - nx[1]) / 16.0), xcolor),
        ((0, 0, 1), (0.062500 + nx[0] / 16.0, 0.000000 + (15 - nx[1]) / 16.0), xcolor),
        ((0, 1, 1), (0.062500 + nx[0] / 16.0, 0.062500 + (15 - nx[1]) / 16.0), xcolor),
        ((0, 1, 0), (0.000000 + nx[0] / 16.0, 0.062500 + (15 - nx[1]) / 16.0), xcolor),
        
        ((0, 1, 0), (0.062500 + nz[0] / 16.0, 0.062500 + (15 - nz[1]) / 16.0), zcolor),
        ((1, 1, 0), (0.000000 + nz[0] / 16.0, 0.062500 + (15 - nz[1]) / 16.0), zcolor),
        ((1, 0, 0), (0.000000 + nz[0] / 16.0, 0.000000 + (15 - nz[1]) / 16.0), zcolor),
        ((0, 0, 0), (0.062500 + nz[0] / 16.0, 0.000000 + (15 - nz[1]) / 16.0), zcolor),
        
        ((1, 1, 0), (0.062500 + px[0] / 16.0, 0.062500 + (15 - px[1]) / 16.0), xcolor),
        ((1, 1, 1), (0.000000 + px[0] / 16.0, 0.062500 + (15 - px[1]) / 16.0), xcolor),
        ((1, 0, 1), (0.000000 + px[0] / 16.0, 0.000000 + (15 - px[1]) / 16.0), xcolor),
        ((1, 0, 0), (0.062500 + px[0] / 16.0, 0.000000 + (15 - px[1]) / 16.0), xcolor),
        
        ((0, 0, 1), (0.000000 + pz[0] / 16.0, 0.000000 + (15 - pz[1]) / 16.0), zcolor),
        ((1, 0, 1), (0.062500 + pz[0] / 16.0, 0.000000 + (15 - pz[1]) / 16.0), zcolor),
        ((1, 1, 1), (0.062500 + pz[0] / 16.0, 0.062500 + (15 - pz[1]) / 16.0), zcolor),
        ((0, 1, 1), (0.000000 + pz[0] / 16.0, 0.062500 + (15 - pz[1]) / 16.0), zcolor),
        
        ((0, 0, 1), (0.000000 + ny[0] / 16.0, 0.062500 + (15 - ny[1]) / 16.0), color),
        ((0, 0, 0), (0.000000 + ny[0] / 16.0, 0.000000 + (15 - ny[1]) / 16.0), color),
        ((1, 0, 0), (0.062500 + ny[0] / 16.0, 0.000000 + (15 - ny[1]) / 16.0), color),
        ((1, 0, 1), (0.062500 + ny[0] / 16.0, 0.062500 + (15 - ny[1]) / 16.0), color),
        
        ((1, 1, 1), (0.062500 + py[0] / 16.0, 0.000000 + (15 - py[1]) / 16.0), topcolor),
        ((1, 1, 0), (0.062500 + py[0] / 16.0, 0.062500 + (15 - py[1]) / 16.0), topcolor),
        ((0, 1, 0), (0.000000 + py[0] / 16.0, 0.062500 + (15 - py[1]) / 16.0), topcolor),
        ((0, 1, 1), (0.000000 + py[0] / 16.0, 0.000000 + (15 - py[1]) / 16.0), topcolor),
    ]
    bd.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY),
    ]
    bd.tex = tex
    return bd

def make_simple(terrain, tx, ty, **kwargs):
    t = (tx, ty)
    return make_box(terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs)

def get_default():
    terrain = OIL.Image.load("terrain.png")
    bd = BlockDefinitions()
    
    # stone
    bd.add(make_simple(terrain, 1, 0), 1)
    
    # grass
    sides = (3, 0)
    top = (0, 0)
    bottom = (2, 0)
    bd.add(make_box(terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides, topcolor=(150, 255, 150, 255)), 2)
    
    # dirt
    bd.add(make_simple(terrain, 2, 0), 3)
    # cobblestone
    bd.add(make_simple(terrain, 0, 1), 4)
    # wood planks
    bd.add(make_simple(terrain, 4, 0), 5)
    # bedrock
    bd.add(make_simple(terrain, 1, 1), 7)
    # sand
    bd.add(make_simple(terrain, 2, 1), 12)
    # gravel
    bd.add(make_simple(terrain, 3, 1), 13)
    # gold ore
    bd.add(make_simple(terrain, 0, 2), 14)
    # copper ore
    bd.add(make_simple(terrain, 1, 2), 15)
    # coal
    bd.add(make_simple(terrain, 2, 2), 16)
    
    # logs
    sides = (4, 1)
    top = (5, 1)
    bottom = (5, 1)
    bd.add(make_box(terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides), 17)
    
    # leaves
    defn = make_simple(terrain, 4, 3, color=(0, 150, 0, 255))
    defn.transparent = True
    bd.add(defn, 18)
    
    return bd
