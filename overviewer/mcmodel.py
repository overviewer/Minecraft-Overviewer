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

from . import chunkrenderer

dirmap = {
    'up':    'PY',
    'down':  'NY',
    'north': 'NZ',
    'south': 'PZ',
    'east':  'PX',
    'west':  'NX',
}

class MCModelBuilder:
    def __init__(self, ap, namespace):
        self.ap = ap
        self.texturevars = {}
        self.namespace = namespace
        self.vi = 0
        self.vertices = []
        self.faces = []
    
    def add_texture(self, name, value):
        if value.startswith('#'):
            self.texturevars['#' + name] = value
        else:
            self.texturevars['#' + name] = 'assets/' + self.namespace + '/textures/' + value + '.png'
    
    def add_element(self, a, b, faces):
        ax, ay, az = (v / 16 for v in a)
        bx, by, bz = (v / 16 for v in b)
        # tex coord rects
        uv = {}
        for dm, da in dirmap.items():
            face = faces.get(dm, {})
            if 'uv' in face:
                uv[da] = [v / 16 for v in face['uv']]
            else:
                uv[da] = [0, 0, 1, 1]
        # face colors
        color = (255, 255, 255, 255)
        xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
        zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)

        vertices = [
            # NX face
            ((ax, ay, az), (uv['NX'][0], uv['NX'][1]), xcolor),
            ((ax, ay, bz), (uv['NX'][2], uv['NX'][1]), xcolor),
            ((ax, by, bz), (uv['NX'][2], uv['NX'][3]), xcolor),
            ((ax, by, az), (uv['NX'][0], uv['NX'][3]), xcolor),
            
            # NZ face
            ((bx, ay, az), (uv['NZ'][0], uv['NZ'][1]), zcolor),
            ((ax, ay, az), (uv['NZ'][2], uv['NZ'][1]), zcolor),
            ((ax, by, az), (uv['NZ'][2], uv['NZ'][3]), zcolor),
            ((bx, by, az), (uv['NZ'][0], uv['NZ'][3]), zcolor),
            
            # PX face
            ((bx, ay, bz), (uv['PX'][0], uv['PX'][1]), xcolor),
            ((bx, ay, az), (uv['PX'][2], uv['PX'][1]), xcolor),
            ((bx, by, az), (uv['PX'][2], uv['PX'][3]), xcolor),
            ((bx, by, bz), (uv['PX'][0], uv['PX'][3]), xcolor),
            
            # PZ face
            ((ax, ay, bz), (uv['PZ'][0], uv['PZ'][1]), zcolor),
            ((bx, ay, bz), (uv['PZ'][2], uv['PZ'][1]), zcolor),
            ((bx, by, bz), (uv['PZ'][2], uv['PZ'][3]), zcolor),
            ((ax, by, bz), (uv['PZ'][0], uv['PZ'][3]), zcolor),
            
            # NY face
            ((ax, ay, bz), (uv['NY'][0], uv['NY'][1]), color),
            ((ax, ay, az), (uv['NY'][2], uv['NY'][1]), color),
            ((bx, ay, az), (uv['NY'][2], uv['NY'][3]), color),
            ((bx, ay, bz), (uv['NY'][0], uv['NY'][3]), color),
            
            # PY face
            ((ax, by, bz), (uv['PY'][0], uv['PY'][1]), color),
            ((bx, by, bz), (uv['PY'][2], uv['PY'][1]), color),
            ((bx, by, az), (uv['PY'][2], uv['PY'][3]), color),
            ((ax, by, az), (uv['PY'][0], uv['PY'][3]), color),
        ]
        
        texes = {}
        types = {}
        for dm, da in dirmap.items():
            face = faces.get(dm, {})
            if 'cullface' in face:
                culla = dirmap[face['cullface']]
                types[da] = getattr(chunkrenderer, culla)
            else:
                types[da] = 0
            if 'texture' in face:
                texes[da] = face['texture']

        faces = []
        if 'NX' in texes:
            faces.append(([0, 1, 2, 3], types['NX'], texes['NX']))
        if 'NZ' in texes:
            faces.append(([4, 5, 6, 7], types['NZ'], texes['NZ']))
        if 'PX' in texes:
            faces.append(([8, 9, 10, 11], types['PX'], texes['PX']))
        if 'PZ' in texes:
            faces.append(([12, 13, 14, 15], types['PZ'], texes['PZ']))
        if 'NY' in texes:
            faces.append(([16, 17, 18, 19], types['NY'], texes['NY']))
        if 'PY' in texes:
            faces.append(([20, 21, 22, 23], types['PY'], texes['PY']))
        
        self.vertices += vertices
        self.faces += [([vi + self.vi for vi in vs], ty, tx) for vs, ty, tx in faces]
        self.vi += len(vertices)
    
    def finalize(self):
        def resolve_tex(tx):
            while tx.startswith('#') and tx in self.texturevars:
                tx = self.texturevars[tx]
            return self.ap.open_texture(tx)
        faces = [(vs, ty, resolve_tex(tx)) for vs, ty, tx in self.faces]
        return chunkrenderer.BlockModel(self.vertices, faces)

def load_model(ap, name, namespace="minecraft"):
    def assemble_builder(rawname):
        data = ap.read_json('assets', namespace, 'models', rawname + '.json')
        if 'parent' in data:
            builder = assemble_builder(data['parent'])
        else:
            builder = MCModelBuilder(ap, namespace)
        
        for el in data.get('elements', []):
            a = el.get('from', [0, 0, 0])
            b = el.get('to', [16, 16, 16])
            faces = el.get('faces', {})
            builder.add_element(a, b, faces)
        
        for name, val in data.get('textures', {}).items():
            builder.add_texture(name, val)
        
        return builder
    builder = assemble_builder('block/' + name)
    return builder.finalize()

if __name__ == "__main__":
    import sys
    from . import assetpack
    from . import blockviewer
    
    ap = assetpack.get_default()
    block = load_model(ap, sys.argv[1])
    blockviewer.render(block)
