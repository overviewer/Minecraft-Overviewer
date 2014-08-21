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

import os.path
import json
import logging
import copy
from itertools import tee

from overviewer import chunkrenderer
from overviewer import util

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
    
    def copy(self):
        """Returns a copy of self that can be mutated freely without
        affecting the original model.
        """
        return copy.deepcopy(self)

    @property
    def triangles(self):
        for indices, facetype, tex in self.faces:
            first = indices[0]
            a, b = tee(indices[1:])
            next(b)
            for i, j in zip(a, b):
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

        self.transparent = kwargs.pop("transparent", False)
        self.solid = kwargs.pop("solid", True)
        self.fluid = kwargs.pop("fluid", False)
        self.nospawn = kwargs.pop("nospawn", False)
        self.datatype = kwargs.pop("datatype", chunkrenderer.BLOCK_DATA_NODATA)
        self.dataparameter = kwargs.pop("dataparameter", None)
        self.biomecolors = kwargs.pop("biomecolors", None)
        
        if kwargs:
            raise ValueError("unknown kwargs: {}".format(kwargs))

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
        return max(self.blocks.keys())

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

class BlockLoadError(Exception):
    def __init__(self, s, file=None):
        if file:
            s = "error loading block '{}': {}".format(file, s)
        else:
            s = "error loading block: {}".format(s)
        super(BlockLoadError, self).__init__(s)

model_types = {}
def model_type(typ):
    """decorator for the load function for a given model type, from JSON"""
    def wrapper(f):
        global model_types
        model_types[typ] = f
        return f
    return wrapper

transform_types = {}
def transform_type(typ):
    """decorator for the transform function for a given type, for JSON"""
    def wrapper(f):
        global transform_types
        transform_types[typ] = f
        return f
    return wrapper

@transform_type("texreplace")
def trans_texreplace(model, data):
    orig = data.pop("from")
    new = data.pop("to")
    
    newfaces = []
    for a, b, tex in model.faces:
        if orig in tex:
            tex = new
        newfaces.append((a, b, tex))
    model.faces = newfaces
    return model

@transform_type("biomecolor")
def trans_biomecolor(model, data):
    needle = data.pop("texture")
    
    newfaces = []
    for a, typ, tex in model.faces:
        if needle in tex:
            typ = typ | chunkrenderer.FACE_BIOME_COLORED
        newfaces.append((a, typ, tex))
    model.faces = newfaces
    return model

@model_type("cube")
def load_cube_model(model, path, label):
    """helper to load a cube-type model, from JSON"""
    texture = model.pop("texture", None)
    side = model.pop("side", texture)
    
    # Positive X is front
    # Positive Y is top
    # Positive Z is right
    tdefaults = {'top': texture, 'bottom': texture, 'left': side, 'right': side, 'front': side, 'back': side}
    
    t = {}
    for name, d in tdefaults.items():
        t[name] = model.pop(name, d)
    for name, tex in t.items():
        if not tex:
            raise RuntimeError("no texture specified for: {}".format(name))
    
    # tex coord rects
    nx = (0, 0, 1, 1)
    px = (0, 0, 1, 1)
    ny = (0, 0, 1, 1)
    py = (0, 0, 1, 1)
    nz = (0, 0, 1, 1)
    pz = (0, 0, 1, 1)
    
    # face colors
    color = (255, 255, 255, 255)
    xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
    zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)
    
    m = BlockModel()
    m.vertices = [
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
        ((0, 1, 1), (py[0], py[1]), color),
        ((1, 1, 1), (py[2], py[1]), color),
        ((1, 1, 0), (py[2], py[3]), color),
        ((0, 1, 0), (py[0], py[3]), color),
    ]

    m.faces = [
        ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX, t['back']),
        ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ, t['left']),
        ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX, t['front']),
        ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ, t['right']),
        ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY, t['bottom']),
        ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY, t['top']),
    ]
    
    return m


from overviewer import objparser
@model_type("obj")
def load_obj_model(model, path, label):
    objpath = os.path.splitext(path)[0] + ".obj"
    objpathmaybe = model.pop("file", None)
    if objpathmaybe:
        objpath = os.path.join(os.path.split(path)[0], objpathmaybe)
    if not os.path.exists(objpath):
        raise RuntimeError("file does not exist: '{}'".format(objpath))
    
    mtlfinder = objparser.SimpleMTLFinder(os.path.split(objpath)[0])
    with open(objpath) as f:
        objfile = objparser.OBJParser.parse(f, mtlfinder)
    
    if len(objfile) == 1:
        defaultname = list(objfile.keys())[0]
    else:
        defaultname = label
    
    name = model.pop("name", defaultname)
    candidates = [s for s in objfile.keys() if name in s]
    if len(candidates) > 1 or len(candidates) == 0:
        raise RuntimeError("object name is ambiguous: {} in '{}'".format(name, objpath))
    
    objmodel = objfile[candidates[0]]
    return objparser.obj_to_blockmodel(objmodel)

def load_model(model, path, label):
    """helper to load a model from a JSON model definition"""
    modeltype = model.pop("type", None)
    modelret = None
    if modeltype in model_types:
        modelret = model_types[modeltype](model, path, label)
    else:
        raise RuntimeError("unknown model type: {}".format(modeltype))
    
    transforms = model.pop("transforms", [])
    for trans in transforms:
        transtype = trans.pop("type", None)
        if not transtype in transform_types:
            raise RuntimeError("unknown transform type: {}".format(transtype))
        transf = transform_types[transtype]
        modelret = transf(modelret, trans)
        if trans:
            raise RuntimeError("unused transform data: {}".format(trans))
    
    if model:
        raise RuntimeError("unused model data: {}".format(model))
    return modelret

def add_from_path(bd, path, namemap={}):
    """add a block definition from a json file, given by path"""
    pathdir, pathfile = os.path.split(path)
    root, _ = os.path.splitext(pathfile)
    
    blockid = None
    name = None
    try:
        blockid, name = root.split('-', 1)
        blockid = int(blockid)
    except ValueError:
        pass
    
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError as e:
        raise BlockLoadError(e, file=path)
    
    name = data.pop("name", name)
    blockid = data.pop("blockid", blockid)
    if not name:
        raise BlockLoadError("definition has no name", file=path)
    if not blockid:
        raise BlockLoadError("definition has no blockid", file=path)
    
    blockdef_arg_names = ["transparent", "solid", "fluid", "nospawn", "datatype", "dataparameter", "biomecolors"]
    blockdef_args = {}
    for argname in blockdef_arg_names:
        if not argname in data:
            continue
        blockdef_args[argname] = data.pop(argname)
    
    if "datatype" in blockdef_args:
        try:
            blockdef_args["datatype"] = getattr(chunkrenderer, "BLOCK_DATA_" + blockdef_args["datatype"].upper())
        except AttributeError:
            raise BlockLoadError("unknown datatype: {}".format(blockdef_args["datatype"], file=path))
    
    bdef = BlockDefinition(**blockdef_args)
    try:
        if "type" in data:
            # inline model, only one
            model = load_model(data, path, "0")
            bdef.add(model, 0)
        else:
            # multiple models
            models = data.pop("models", {})
            for dataorig, model in models.items():
                dataval = int(dataorig)
                model = load_model(model, path, dataorig)
                bdef.add(model, dataval)
    except BlockLoadError:
        raise
    except Exception as e:
        raise BlockLoadError(e, file=path)
    
    if data:
        raise BlockLoadError("unused info: {}".format(data), file=path)
    
    bd.add(bdef, blockid)

def get_default():
    bd = BlockDefinitions()    
    blockspath = os.path.join(util.get_program_path(), "overviewer", "data", "blocks")
    
    for root, subdirs, files in os.walk(blockspath):
        del subdirs[:]
        
        for fname in files:
            if fname.startswith("."):
                continue
            if not fname.endswith(".json"):
                continue
            add_from_path(bd, os.path.join(root, fname))

    return bd
