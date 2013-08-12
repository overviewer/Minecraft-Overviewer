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
from itertools import product
from operator import itemgetter

from .oil import Image, Matrix
from .canvas import Renderer
from . import chunkrenderer

"""

isometricrenderer.py contains the IsometricRenderer class, a Renderer
implementation that renders RegionSet objects in an isometric
perspective.

It uses the chunkrenderer module, a generalized (and C-based) 3D chunk
renderer.

"""

class IsometricRenderer(Renderer):
    sections_per_chunk = 16
    def __init__(self, regionset, textures, blockdefs, matrix):
        self.regionset = regionset
        self.textures = textures
        self.blockdefs = blockdefs
        self.origmatrix = matrix
        self.matrix = Matrix().scale(1, -1, 1) * matrix
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
        
        # campile the block definitions
        self.compiled_blockdefs = chunkrenderer.compile_block_definitions(self.textures, self.blockdefs)
    
    def __getstate__(self):
        # a lot of our internal structure is not pickleable
        # so just send the args for __init__
        return (self.regionset, self.textures, self.blockdefs, self.origmatrix.data)
    
    def __setstate__(self, args):
        # turn the matrix back into an OIL Matrix
        mat = args[-1]
        mat = Matrix(mat)
        args = args[:-1] + (mat,)
        self.__init__(*args)
    
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
        
        im_matrix = Matrix().orthographic(origin[0], origin[0] + im.size[0], origin[1] + im.size[1], origin[1], minz, maxz)
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
                
                local_matrix = im_matrix * Matrix().translate(chunkx * 16, chunky * 16, chunkz * 16)
                chunkrenderer.render(self.regionset, chunkx, chunky, chunkz, im, local_matrix, self.compiled_blockdefs)
