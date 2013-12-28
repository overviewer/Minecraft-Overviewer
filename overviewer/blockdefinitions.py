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
        self.dataparameter = kwargs.get("dataparameter", None)

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

def get_default():
    bd = BlockDefinitions()
    
    from overviewer import blocks
    blocks.get_all(bd)

    return bd
