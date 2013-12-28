import sys
import os.path

from .blockdefinitions import BlockModel, BlockDefinition

class MTLFinder(object):
    """searches for an mtl file by name"""
    def find(self, name):
        """returns a file-like object associated with this name"""
        raise NotImplementedError(self.__class__.__name__ + ".find")

class SimpleMTLFinder(MTLFinder):
    """a simple finder that looks in a single directory"""
    def __init__(self, path):
        self.path = path
    def find(self, name):
        return open(os.path.join(self.path, name))

class Object(object):
    """container for meshes loaded"""
    def __init__(self):
        # list of 3d tuples
        self.vertices = []
        # list of 2d tuples
        self.texcoords = []
        # list of materials
        self.materials = []
        # list of (materialindex, [vertices])
        # where vertices is a list of (vertid, texid, normalid)
        # each id can be None, which means there was none given
        # materialindex can also be None
        self.faces = []

class BaseParser(object):
    """a base parser for OBJ-like files"""
    def __init__(self):
        self.result = {}
    
    def get_emitable(self):
        """
        returns a name, value pair if there is an object to emit.
        otherwise, returns None.
        """
        raise NotImplementedError(self.__class__.__name__ + '.get_emitable')

    def reset_state(self):
        """resets internal parse state after emission"""
        raise NotImplementedError(self.__class__.__name__ + '.reset_state')
    
    @classmethod
    def parse(cls, f, *args, **kwargs):
        """a convenience method to create a parser and feed it a file"""
        o = cls(*args, **kwargs)
        o.reset_state()
        o.feed(f)
        o.emit()
        return o.result

    def feed(self, f):
        for line in f.readlines():
            # remove comments
            line = line.split('#', 1)[0]
            # strip whitespace
            line.strip()
            
            # split into components
            line = line.split()
            
            # ignore empty lines
            if not line:
                continue
                
            cmd = line[0]
            args = line[1:]
            
            f = getattr(self, cmd, None)
            if f is None:
                pass #print "unhandled", cmd, args
            else:
                f(*args)

    def emit(self):
        """used internally to emit an object"""
        e = self.get_emitable()
        if e:
            n, o = e
            self.result[n] = o
        self.reset_state()

class MTLParser(BaseParser):
    def get_emitable(self):
        if self.mtl and self.mname:
            return self.mname, self.mtl
        return None
    
    def reset_state(self):
        self.mname = None
        self.mtl = {}
    
    def newmtl(self, name):
        self.emit()
        self.mname = name
    
    def Kd(self, r, g, b):
        self.mtl['diffuse'] = tuple(float(x) for x in (r, g, b))
    
    def map_Kd(self, path):
        self.mtl['diffuse_map'] = path
    
class OBJParser(BaseParser):
    def __init__(self, mfinder):
        super(OBJParser, self).__init__()
        self.mfinder = mfinder
        self.mtls = {}

    # helper to parse a face vertex
    @staticmethod
    def parse_vert(s):
        parts = s.split('/')
        if len(parts) == 1:
            return (int(parts[0]) - 1, None, None)
        if len(parts) == 2:
            return (int(parts[0]) - 1, int(parts[1]) - 1, None)
        else:
            v, tc, n = parts
            v = int(v) - 1
            n = int(n) - 1
            if tc:
                tc = int(tc) - 1
            else:
                tc = None
            return (v, tc, n)
    
    def get_emitable(self):
        if self.obj.faces:
            return self.oname, self.obj
        return None
    
    def reset_state(self):
        self.oname = None
        self.fmtl = None
        self.obj = Object()
    
    def mtllib(self, lib):
        self.mtls.update(MTLParser.parse(self.mfinder.find(lib)))
    
    def o(self, name):
        self.emit()
        self.oname = name
    
    def v(self, x, y, z):
        v = tuple(float(t) for t in (x, y, z))
        self.obj.vertices.append(v)

    def vt(self, x, y):
        v = tuple(float(t) for t in (x, y))
        self.obj.texcoords.append(v)
    
    def usemtl(self, name):
        mtl = self.mtls.get(name)
        self.fmtl = len(self.obj.materials)
        self.obj.materials.append(mtl)
    
    def f(self, *args):
        verts = [self.parse_vert(v) for v in args]
        self.obj.faces.append((self.fmtl, verts))

def obj_to_blockmodel(obj):
    model = BlockModel()
    for mi, verts in obj.faces:
        mtl = obj.materials[mi]
        tex = mtl['diffuse_map']
        
        indices = []
        for (v, tc, n) in verts:
            index = len(model.vertices)
            v = obj.vertices[v]
            tc = obj.texcoords[tc]
            vertex = (v, tc, (255, 255, 255, 255))
            model.vertices.append(vertex)
            indices.append(index)
        model.faces.append((indices, 0, tex))
    return model

def add_from_path(bd, path, namemap={}):
    pathdir = os.path.split(path)[0]
    mfinder = SimpleMTLFinder(pathdir)
    with open(path) as f:
        objs = OBJParser.parse(f, mfinder)
    
    converted = {}
    
    for name, obj in objs.items():
        try:
            name = name.rsplit('_', 1)[-1]
            name = name.split(':', 1)
            if len(name) == 2:
                pre_name, data = name
            else:
                pre_name = name[0]
                data = 0
            data = int(data)
            name = namemap.get(pre_name, None)
            if name is None:
                name = int(pre_name)
        except ValueError:
            raise RuntimeError("invalid object name in: " + path)
        
        model = obj_to_blockmodel(obj)
        converted.setdefault(name, dict())[data] = model
    
    for blockid, defs in converted.items():
        if len(defs) == 1:
            model = defs[defs.keys()[0]]
            bd.add(BlockDefinition(model), blockid)
        else:
            bdef = BlockDefinition()
            for data, model in defs.items():
                bdef.add(model, data)
            bd.add(bdef, blockid)
        
