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

from math import ceil
from itertools import product, tee, izip
from operator import itemgetter

import OIL
from .canvas import Renderer
from . import chunkrenderer

"""

isometricrenderer.py contains the IsometricRenderer class, a Renderer
implementation that renders RegionSet objects in an isometric
perspective.

It uses the chunkrenderer module, a generalized (and C-based) 3D chunk
renderer.

"""

class BlockDefinition(object):
    def __init__(self, tex, nx=(0, 0), px=(0, 0), ny=(0, 0), py=(0, 0), nz=(0, 0), pz=(0, 0), color=(255, 255, 255, 255), topcolor=None):
        if topcolor is None:
            topcolor = color
        xcolor = tuple(int(c * 0.8) for c in color[:3]) + (color[3],)
        zcolor = tuple(int(c * 0.9) for c in color[:3]) + (color[3],)
        self.vertices = [
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
        self.faces = [
            ([0, 1, 2, 3], chunkrenderer.FACE_TYPE_NX),
            ([4, 5, 6, 7], chunkrenderer.FACE_TYPE_NZ),
            ([8, 9, 10, 11], chunkrenderer.FACE_TYPE_PX),
            ([12, 13, 14, 15], chunkrenderer.FACE_TYPE_PZ),
            ([16, 17, 18, 19], chunkrenderer.FACE_TYPE_NY),
            ([20, 21, 22, 23], chunkrenderer.FACE_TYPE_PY),
        ]
        self.tex = tex

    @property
    def triangles(self):
        for indices, facetype in self.faces:
            first = indices[0]
            a, b = tee(indices[1:])
            b.next()
            for i, j in izip(a, b):
                yield ((first, i, j), facetype)

class BlockDefinitions(object):
    def __init__(self):
        self.blocks = {}
        self.max_blockid = 0
        self.max_data = 0
        
        self.terrain = OIL.Image.load("terrain.png")
        
        # stone
        self.add_simple(1, 1, 0)
        
        # grass
        sides = (3, 0)
        top = (0, 0)
        bottom = (2, 0)
        self.add(2, BlockDefinition(self.terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides, topcolor=(150, 255, 150, 255)))
        
        # dirt
        self.add_simple(3, 2, 0)
        
        # cobblestone
        self.add_simple(4, 0, 1)
        
        # wood planks
        self.add_simple(5, 4, 0)
        
        # bedrock
        self.add_simple(7, 1, 1)
        
        # sand
        self.add_simple(12, 2, 1)
        
        # gravel
        self.add_simple(13, 3, 1)
        
        # gold ore
        self.add_simple(14, 0, 2)
        
        # copper ore
        self.add_simple(15, 1, 2)
        
        # coal
        self.add_simple(16, 2, 2)
        
        # logs
        sides = (4, 1)
        top = (5, 1)
        bottom = (5, 1)
        self.add(17, BlockDefinition(self.terrain, nx=sides, px=sides, ny=bottom, py=top, nz=sides, pz=sides))
        
        # leaves
        self.add_simple(18, 4, 3, color=(0, 150, 0, 255))
        
        chunkrenderer.compile_block_definitions(self)
    
    def add_simple(self, blockid, tx, ty, **kwargs):
        t = (tx, ty)
        self.add(blockid, BlockDefinition(self.terrain, nx=t, px=t, ny=t, py=t, nz=t, pz=t, **kwargs))
    
    def add(self, blockid, blockdef, data=0):
        self.blocks[(blockid, data)] = blockdef
        self.max_blockid = max(self.max_blockid, blockid + 1)
        self.max_data = max(self.max_data, data + 1)

class IsometricRenderer(Renderer):
    sections_per_chunk = 16
    def __init__(self, world, regionset, blockdefs, matrix):
        self.world = world
        self.regionset = regionset
        self.blockdefs = blockdefs
        self.matrix = OIL.Matrix().scale(1, -1, 1) * matrix
        self.inverse = self.matrix.inverse
        
        # computed handy things, relative vectors
        # assumes the matrix is affine
        # so: only scales, rotations, and translations!
        
        self.origin = self.matrix.transform(0, 0, 0)
        self.invorigin = self.inverse.transform(0, 0, 0)
        
        self.chunkbox = self._makebox((16, 0, 0), (0, 16 * self.sections_per_chunk, 0), (0, 0, 16))
        self.sectionbox = self._makebox((16, 0, 0), (0, 16, 0), (0, 0, 16))
        self.sectionvec = self._transformrel(0, 16, 0)
        self.viewvec = self._transformrel(0, 0, -1, inverse=True)
        assert self.viewvec[1] != 0
    
    def _transformrel(self, x, y, z, inverse=False):
        mat = self.matrix
        origin = self.origin
        if inverse:
            mat = self.inverse
            origin = self.invorigin
        rel = mat.transform(x, y, z)
        return (rel[0] - origin[0], rel[1] - origin[1], rel[2] - origin[2])
    
    def _makebox(self, v1, v2, v3):
        # helper to turn a paralellepiped into a 2d rect
        x1, y1, z1 = self._transformrel(*v1)
        x2, y2, z2 = self._transformrel(*v2)
        x3, y3, z3 = self._transformrel(*v3)
        
        allpts = ((0, 0, 0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x1 + x2, y1 + y2, z1 + z2), (x1 + x3, y1 + y3, z1 + z3), (x2 + x3, y2 + y3, z2 + z3), (x1 + x2 + x3, y1 + y2 + y3, z1 + z2 + z3))
        
        xs = [t[0] for t in allpts]
        ys = [t[1] for t in allpts]
        zs = [t[2] for t in allpts]
        
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def get_render_sources(self):
        return self.regionset.iterate_chunks()

    def get_render_source_mtime(self, source):
        return source[2]

    def get_rect_for_render_source(self, source):
        x, y, _ = self.matrix.transform(16 * source[0], 0, 16 * source[1])
        return (int(x + self.chunkbox[0][0]), int(y + self.chunkbox[0][1]), int(x + self.chunkbox[1][0]) + 1, int(y + self.chunkbox[1][1]) + 1)

    def get_render_sources_in_rect(self, rect):
        for _, _, _, cx, cy, mtime in self._get_chunks_in_rect(rect):
            yield (cx, cy, mtime)

    def get_full_rect(self):
        minx = miny = maxx = maxy = 0
        for cx, cz, _ in self.regionset.iterate_chunks():
            x, y, _ = self.matrix.transform(16 * cx, 0, 16 * cz)

            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)

        return (int(minx + self.chunkbox[0][0]), int(miny + self.chunkbox[0][1]), int(maxx + self.chunkbox[1][0]) + 1, int(maxy + self.chunkbox[1][1]) + 1)

    def _get_chunks_in_rect(self, rect):
        minx = rect[0] - self.chunkbox[1][0]
        miny = rect[1] - self.chunkbox[1][1]
        maxx = rect[2] - self.chunkbox[0][0] + 1
        maxy = rect[3] - self.chunkbox[0][1] + 1
        rect = (minx, miny, maxx, maxy)
        
        origin = self.inverse.transform(rect[0], rect[1], 0)
        vec1 = self._transformrel(rect[2] - rect[0], 0, 0, inverse=True)
        vec2 = self._transformrel(0, rect[3] - rect[1], 0, inverse=True)

        pt1 = tuple(origin)
        pt2 = tuple(origin[i] + vec1[i] for i in range(3))
        pt3 = tuple(origin[i] + vec2[i] for i in range(3))
        pt4 = tuple(origin[i] + vec1[i] + vec2[i] for i in range(3))
        
        # project these corners to the y=0 plane using the view vector
        pt1 = tuple(pt1[i] - self.viewvec[i] * (pt1[1] / self.viewvec[1]) for i in range(3))
        pt2 = tuple(pt2[i] - self.viewvec[i] * (pt2[1] / self.viewvec[1]) for i in range(3))
        pt3 = tuple(pt3[i] - self.viewvec[i] * (pt3[1] / self.viewvec[1]) for i in range(3))
        pt4 = tuple(pt4[i] - self.viewvec[i] * (pt4[1] / self.viewvec[1]) for i in range(3))
        
        minx = int(ceil(min([i[0] / 16 for i in (pt1, pt2, pt3, pt4)])))
        maxx = int(max([i[0] / 16 for i in (pt1, pt2, pt3, pt4)]))
        minz = int(ceil(min([i[2] / 16 for i in (pt1, pt2, pt3, pt4)])))
        maxz = int(max([i[2] / 16 for i in (pt1, pt2, pt3, pt4)]))
        for cx, cz in product(xrange(minx, maxx + 1), xrange(minz, maxz + 1)):
            x, y, z = self.matrix.transform(16 * cx, 0, 16 * cz)
            if not (x >= rect[0] and x < rect[2]) or not (y >= rect[1] and rect[3]):
                continue
            mtime = self.regionset.get_chunk_mtime(cx, cz)
            if mtime is None:
                continue
            yield (x, y, z, cx, cz, mtime)

    def _get_tile_chunks(self, origin, im):
        rect = (origin[0], origin[1], origin[0] + im.size[0], origin[1] + im.size[1])
        chunks_out = list(self._get_chunks_in_rect(rect))
        if not chunks_out:
            return (0, 0, [])
        chunks_out.sort(key=itemgetter(1))
        
        zs = map(itemgetter(2), chunks_out)
        minz = min(zs) + self.chunkbox[0][2]
        maxz = max(zs) + self.chunkbox[1][2]
        
        return (minz, maxz, chunks_out)

    def render(self, origin, im):
        minz, maxz, chunks = self._get_tile_chunks(origin, im)
        if not chunks:
            return
        
        im_matrix = OIL.Matrix().orthographic(origin[0], origin[0] + im.size[0], origin[1] + im.size[1], origin[1], minz, maxz)
        im_matrix *= self.matrix
        
        for x, y, _, chunkx, chunkz, _ in chunks:
            x -= origin[0];
            y -= origin[1];
            x += self.sectionvec[0] * self.sections_per_chunk
            y += self.sectionvec[1] * self.sections_per_chunk
            for chunky in xrange(self.sections_per_chunk - 1, -1, -1):
                x -= self.sectionvec[0]
                y -= self.sectionvec[1]
                if (x + self.sectionbox[1][0] < 0 or x + self.sectionbox[0][0] >= im.size[0] or y + self.sectionbox[1][1] < 0 or y + self.sectionbox[0][1] >= im.size[1]):
                    continue
                
                local_matrix = im_matrix * OIL.Matrix().translate(chunkx * 16, chunky * 16, chunkz * 16)
                chunkrenderer.render(self.world, self.regionset, chunkx, chunky, chunkz, im, local_matrix, self.blockdefs)
