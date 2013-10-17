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

from overviewer import chunkrenderer

"""

blockdefinitions.py contains the BlockDefinitions class, which acts as
a method of registering BlockDefinition objects to be used by the C
chunk renderer.

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
       indexes into the vertices list, and type is a bitmask made from
       chunkrenderer.FACE_TYPE_* constants.
     
     * tex: a texture path (as a string, uses textures.Textures
       eventually)
     
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

def make_box(tex, nx=(0, 0, 1, 1), px=(0, 0, 1, 1), ny=(0, 0, 1, 1), py=(0, 0, 1, 1), nz=(0, 0, 1, 1), pz=(0, 0, 1, 1), color=(255, 255, 255, 255), topcolor=None):
    if topcolor is None:
        topcolor = color
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)
    bd = BlockDefinition()
    bd.vertices = [
        ((0, 0, 0), (nx[0], nx[1]), xcolor),
        ((0, 0, 1), (nx[2], nx[1]), xcolor),
        ((0, 1, 1), (nx[2], nx[3]), xcolor),
        ((0, 1, 0), (nx[0], nx[3]), xcolor),
        
        ((0, 1, 0), (nz[0], nz[1]), zcolor),
        ((1, 1, 0), (nz[2], nz[1]), zcolor),
        ((1, 0, 0), (nz[2], nz[3]), zcolor),
        ((0, 0, 0), (nz[0], nz[3]), zcolor),
        
        ((1, 1, 0), (px[0], px[1]), xcolor),
        ((1, 1, 1), (px[2], px[1]), xcolor),
        ((1, 0, 1), (px[2], px[3]), xcolor),
        ((1, 0, 0), (px[0], px[3]), xcolor),
        
        ((0, 0, 1), (pz[0], pz[1]), zcolor),
        ((1, 0, 1), (pz[2], pz[1]), zcolor),
        ((1, 1, 1), (pz[2], pz[3]), zcolor),
        ((0, 1, 1), (pz[0], pz[3]), zcolor),
        
        ((0, 0, 1), (ny[0], ny[1]), color),
        ((0, 0, 0), (ny[2], ny[1]), color),
        ((1, 0, 0), (ny[2], ny[3]), color),
        ((1, 0, 1), (ny[0], ny[3]), color),
        
        ((1, 1, 1), (py[0], py[1]), topcolor),
        ((1, 1, 0), (py[2], py[1]), topcolor),
        ((0, 1, 0), (py[2], py[3]), topcolor),
        ((0, 1, 1), (py[0], py[3]), topcolor),
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

def make_simple(terrain, **kwargs):
    t = (0, 0, 1, 1)
    return make_box(terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs)

def get_default():
    bd = BlockDefinitions()
    
    # stone
    bd.add(make_simple("textures/blocks/stone.png"), 1)
    
    # grass
    #sides = (3, 0)
    #top = (0, 0)
    #bottom = (2, 0)
    #bd.add(make_box(terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides, topcolor=(150, 255, 150, 255)), 2)
    
    # dirt
    bd.add(make_simple("textures/blocks/dirt.png"), 3)
    # cobblestone
    bd.add(make_simple("textures/blocks/stonebrick.png"), 4)
    # wood planks
    bd.add(make_simple("textures/blocks/wood.png"), 5)
    # bedrock
    bd.add(make_simple("textures/blocks/bedrock.png"), 7)
    # sand
    bd.add(make_simple("textures/blocks/sand.png"), 12)
    # gravel
    bd.add(make_simple("textures/blocks/gravel.png"), 13)
    # gold ore
    bd.add(make_simple("textures/blocks/oreGold.png"), 14)
    # iron ore
    bd.add(make_simple("textures/blocks/oreIron.png"), 15)
    # coal
    bd.add(make_simple("textures/blocks/oreCoal.png"), 16)
    
    # logs
    #sides = (4, 1)
    #top = (5, 1)
    #bottom = (5, 1)
    #bd.add(make_box(terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides), 17)
    
    # leaves
    defn = make_simple("textures/blocks/leaves.png", color=(0, 150, 0, 255))
    defn.transparent = True
    bd.add(defn, 18)
    
    return bd
