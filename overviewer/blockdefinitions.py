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

class BlockModel(object):
    """This represents a mesh, or a description of how to draw a model. It
    includes the vertices, textures, and face definitions.

    The C code expects this object to have the following attributes:
    
     * vertices: a list of (coordinate, texcoords, color) tuples,
       where coordinate is a 3-tuple of numbers, texcoords is a
       2-tuple, and color is a 4-tuple of integers between 0 and 255.

     * faces: a list of ((pt, ...), type, texture) where pt is at least 3
       integers corresponding to indexes into the vertices array. type is a
       bitmask made from chunkrenderer.FACE_TYPE_* constants. texture is a
       texture name or path.
     
    Note that the C code actually expects `triangles`, not `faces`. In this
    implementation, `triangles` is generated from `faces`, which is
    similarly-structured but may contain more than 3 indexes per face.

    """
    def __init__(self):
        self.vertices = []
        self.faces = []

    @property
    def triangles(self):
        for indices, facetype, tex in self.faces:
            first = indices[0]
            a, b = tee(indices[1:])
            b.next()
            for i, j in izip(a, b):
                yield ((first, i, j), facetype, tex)

class BlockDefinition(object):
    """
    This represents the definition for a block, holding one or more BlockModel
    objects. A block definition may include more than one block model for
    different scenarios according to various conditions, including the data
    value of the block, or properties of surrounding worlds.

    The block definition defines a function for determining which model to use.
    This is how blocks with "pseudo data" are supported: blocks that use data
    from neighboring blocks define an appropriate function for the "data"
    attribute. Blocks that only depend on their own data value will use a
    simple passthrough data function.

    The datatype attribute specifies how to determine this block's data. Data
    types are defined in the C file blockdata.c. This attribute should be a
    pointer to one of the chunkrenderer.BLOCK_DATA_* values.

    Some data types take a parameter. For such data types, set the `dataparameter`
    attribute.

    The transparent flag affects the occlusion algorithms of the renderer. This
    should be False unless the block completely occupies its cube and isn't
    transparent anywhere.
    
    solid, fluid, and nospawn are not currently used

    """
    
    def __init__(self, model_zero=None, **kwargs):
        self.models = []

        self.transparent = kwargs.get("transparent", False)
        self.solid = kwargs.get("solid", True)
        self.fluid = kwargs.get("fluid", False)
        self.nospawn = kwargs.get("nospawn", False)
        self.datatype = kwargs.get("datatype", chunkrenderer.BLOCK_DATA_NODATA)
        self.dataparameter = kwargs.get("dataparm", None)

        if model_zero:
            self.add(model_zero, 0)

    def add(self, blockmodel, datavalue):
        """Adds the given model to be used when the block has the given data value.
        
        """
        while datavalue >= len(self.models):
            self.models.append(None)
        self.models[datavalue] = blockmodel

class BlockDefinitions(object):
    """
    This object is a container for BlockDefinition objects, which can
    be added by add() or the other convenience functions. The C code
    expects it to have the following attributes:
    
     * blocks: a dictionary mapping blockid ints to
       BlockDefinition objects.
     
    """
    
    def __init__(self):
        self.blocks = {}

    @property
    def max_blockid(self):
        return max(self.blocks.iterkeys())

    def add(self, blockdef, blockid):
        """Adds a block definition as block id `blockid`.
        `blockid` may also be a list to add the same definition for several block ids.

        """
        try:
            blockid = iter(blockid)
        except TypeError:
            blockid = [blockid]
        for b in blockid:
            self.blocks[b] = blockdef

def make_box(tex, nx=(0, 0, 1, 1), px=(0, 0, 1, 1), ny=(0, 0, 1, 1), py=(0, 0, 1, 1), nz=(0, 0, 1, 1), pz=(0, 0, 1, 1), color=(255, 255, 255, 255), topcolor=None):
    if topcolor is None:
        topcolor = color
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)

    if isinstance(tex, str):
        tex = (tex,)*6

    model = BlockModel()
    model.vertices = [
        # NX face
        ((0, 0, 0), (nx[0], nx[1]), xcolor),
        ((0, 0, 1), (nx[2], nx[1]), xcolor),
        ((0, 1, 1), (nx[2], nx[3]), xcolor),
        ((0, 1, 0), (nx[0], nx[3]), xcolor),
        
        # NZ face
        ((1, 0, 0), (nz[0], nz[1]), zcolor),
        ((0, 0, 0), (nz[2], nz[1]), zcolor),
        ((0, 1, 0), (nz[2], nz[3]), zcolor),
        ((1, 1, 0), (nz[0], nz[3]), zcolor),
        
        # PX face
        ((1, 0, 1), (px[0], px[1]), xcolor),
        ((1, 0, 0), (px[2], px[1]), xcolor),
        ((1, 1, 0), (px[2], px[3]), xcolor),
        ((1, 1, 1), (px[0], px[3]), xcolor),
        
        # PZ face
        ((0, 0, 1), (pz[0], pz[1]), zcolor),
        ((1, 0, 1), (pz[2], pz[1]), zcolor),
        ((1, 1, 1), (pz[2], pz[3]), zcolor),
        ((0, 1, 1), (pz[0], pz[3]), zcolor),
        
        # NY face
        ((0, 0, 1), (ny[0], ny[1]), color),
        ((0, 0, 0), (ny[2], ny[1]), color),
        ((1, 0, 0), (ny[2], ny[3]), color),
        ((1, 0, 1), (ny[0], ny[3]), color),
        
        # PY face
        ((1, 1, 1), (py[0], py[1]), topcolor),
        ((1, 1, 0), (py[2], py[1]), topcolor),
        ((0, 1, 0), (py[2], py[3]), topcolor),
        ((0, 1, 1), (py[0], py[3]), topcolor),
    ]
    model.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, tex[0]),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, tex[1]),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, tex[2]),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, tex[3]),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, tex[4]),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, tex[5]),
    ]
    return model

def make_simple(terrain, **kwargs):
    t = (0, 0, 1, 1)
    return make_box(terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs)

def get_default():
    bd = BlockDefinitions()
    
    # stone
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/stone.png")), 1)
    
    # grass
    side = "assets/minecraft/textures/blocks/grass_side.png"
    top = "assets/minecraft/textures/blocks/grass_top.png"
    bottom = "assets/minecraft/textures/blocks/dirt.png"
    bd.add(BlockDefinition(make_simple((side,side,side,side,bottom,top), topcolor=(0,255,0,255))), 2)
    
    # dirt
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/dirt.png")), 3)
    # cobblestone
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/stonebrick.png")), 4)

    # wood planks
    wood_planks = BlockDefinition(datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_oak.png"), 0)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_spruce.png"), 1)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_birch.png"), 2)
    wood_planks.add(make_simple("assets/minecraft/textures/blocks/planks_jungle.png"), 3)
    bd.add(wood_planks, 5)

    # bedrock
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/bedrock.png")), 7)
    # sand
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/sand.png")), 12)
    # gravel
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/gravel.png")), 13)
    # gold ore
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/gold_ore.png")), 14)
    # iron ore
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/iron_ore.png")), 15)
    # coal
    bd.add(BlockDefinition(make_simple("assets/minecraft/textures/blocks/coal_ore.png")), 16)
    
    # logs
    #sides = (4, 1)
    #top = (5, 1)
    #bottom = (5, 1)
    #bd.add(make_box(terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides), 17)
    
    # leaves
    leaves = BlockDefinition(transparent=True, datatype=chunkrenderer.BLOCK_DATA_PASSTHROUGH)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_oak.png", color=(0, 150, 0, 255)), 0)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_spruce.png", color=(0, 150, 0, 255)), 1)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_birch.png", color=(0, 150, 0, 255)), 2)
    leaves.add(make_simple("assets/minecraft/textures/blocks/leaves_jungle.png", color=(0, 150, 0, 255)), 3)
    bd.add(leaves, 18)

    # cactus
    side = "assets/minecraft/textures/blocks/cactus_side.png"
    top = "assets/minecraft/textures/blocks/cactus_top.png"
    bottom = "assets/minecraft/textures/blocks/cactus_bottom.png"
    model = BlockModel()
    nx=px=nz=pz=ny=py=(0,0,1,1)
    model.vertices = [
        # NX face
        ((0+0.0625, 0, 0), (nx[0], nx[1]), (204,204,204,255)),
        ((0+0.0625, 0, 1), (nx[2], nx[1]), (204,204,204,255)),
        ((0+0.0625, 1, 1), (nx[2], nx[3]), (204,204,204,255)),
        ((0+0.0625, 1, 0), (nx[0], nx[3]), (204,204,204,255)),
        
        # NZ face
        ((1, 0, 0+0.0625), (nz[0], nz[1]), (229,229,229,255)),
        ((0, 0, 0+0.0625), (nz[2], nz[1]), (229,229,229,255)),
        ((0, 1, 0+0.0625), (nz[2], nz[3]), (229,229,229,255)),
        ((1, 1, 0+0.0625), (nz[0], nz[3]), (229,229,229,255)),
        
        # PX face
        ((1-0.0625, 0, 1), (px[0], px[1]), (204,204,204,255)),
        ((1-0.0625, 0, 0), (px[2], px[1]), (204,204,204,255)),
        ((1-0.0625, 1, 0), (px[2], px[3]), (204,204,204,255)),
        ((1-0.0625, 1, 1), (px[0], px[3]), (204,204,204,255)),
        
        # PZ face
        ((0, 0, 1-0.0625), (pz[0], pz[1]), (229,229,229,255)),
        ((1, 0, 1-0.0625), (pz[2], pz[1]), (229,229,229,255)),
        ((1, 1, 1-0.0625), (pz[2], pz[3]), (229,229,229,255)),
        ((0, 1, 1-0.0625), (pz[0], pz[3]), (229,229,229,255)),
        
        # NY face
        ((0, 0, 1), (ny[0], ny[1]), (255,255,255,255)),
        ((0, 0, 0), (ny[2], ny[1]), (255,255,255,255)),
        ((1, 0, 0), (ny[2], ny[3]), (255,255,255,255)),
        ((1, 0, 1), (ny[0], ny[3]), (255,255,255,255)),
        
        # PY face
        ((1, 1, 1), (py[0], py[1]), (255,255,255,255)),
        ((1, 1, 0), (py[2], py[1]), (255,255,255,255)),
        ((0, 1, 0), (py[2], py[3]), (255,255,255,255)),
        ((0, 1, 1), (py[0], py[3]), (255,255,255,255)),
    ]
    model.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, side),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, side),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, side),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, side),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, bottom),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, top),
    ]
    bd.add(BlockDefinition(model, transparent=True), 81)

    return bd
