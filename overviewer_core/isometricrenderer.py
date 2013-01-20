import numpy
from math import ceil
from itertools import product
from operator import itemgetter

from .c_overviewer import render_loop
from .canvas import Renderer

"""

isometricrenderer.py contains the IsometricRenderer class, a Renderer
implementation that renders RegionSet objects in an isometric
perspective.

"""

class IsometricRenderer(Renderer):
    # independent parameters
    chunk_section_height = 384
    chunk_section_add = 192
    chunk_width = 384
    sections_per_chunk = 16
    chunk_to_image = numpy.array(((192, 192),
                                  (-96, 96)))

    # computed
    chunk_height = chunk_section_height + (sections_per_chunk - 1) * chunk_section_add
    image_to_chunk = numpy.linalg.inv(chunk_to_image)

    def __init__(self, world, regionset, textures, rendermode):
        self.world = world
        self.regionset = regionset
        self.textures = textures
        self.rendermode = rendermode

    def get_render_sources(self):
        return self.regionset.iterate_chunks()

    def get_render_source_mtime(self, source):
        return source[2]

    def get_rect_for_render_source(self, source):
        x, y = self.chunk_to_image.dot(source[:2])
        return (x, y, x + self.chunk_width, y + self.chunk_height)

    def get_render_sources_in_rect(self, rect):
        for _, _, cx, cy, mtime in self._get_chunks_in_rect(rect):
            yield (cx, cy, mtime)

    def get_full_rect(self):
        minx = miny = maxx = maxy = 0
        for cx, cz, _ in self.regionset.iterate_chunks():
            x, y = self.chunk_to_image.dot((cx, cz))

            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)

        return (minx, miny, maxx + self.chunk_width, maxy + self.chunk_height)

    def _get_chunks_in_rect(self, rect):
        minx = rect[0] - self.chunk_width
        miny = rect[1] - self.chunk_height
        maxx = rect[2] + 1
        maxy = rect[3] + 1
        rect = (minx, miny, maxx, maxy)
        
        origin = self.image_to_chunk.dot(rect[:2])
        vec1 = self.image_to_chunk.dot((rect[2] - rect[0], 0))
        vec2 = self.image_to_chunk.dot((0, rect[3] - rect[1]))

        pt1 = tuple(origin)
        pt2 = tuple(origin + vec1)
        pt3 = tuple(origin + vec2)
        pt4 = tuple(origin + vec1 + vec2)

        minx = int(ceil(min([i[0] for i in (pt1, pt2, pt3, pt4)])))
        maxx = int(max([i[0] for i in (pt1, pt2, pt3, pt4)]))
        miny = int(ceil(min([i[1] for i in (pt1, pt2, pt3, pt4)])))
        maxy = int(max([i[1] for i in (pt1, pt2, pt3, pt4)]))
        for cx, cy in product(xrange(minx, maxx + 1), xrange(miny, maxy + 1)):
            x, y = self.chunk_to_image.dot((cx, cy))
            if not (x >= rect[0] and x < rect[2]) or not (y >= rect[1] and rect[3]):
                continue
            mtime = self.regionset.get_chunk_mtime(cx, cy)
            if mtime is None:
                continue
            yield (x, y, cx, cy, mtime)

    def _get_tile_chunks(self, origin, im):
        rect = (origin[0], origin[1], origin[0] + im.size[0], origin[1] + im.size[1])
        chunks_out = list(self._get_chunks_in_rect(rect))
        chunks_out.sort(key=itemgetter(1))
        return chunks_out

    def render(self, origin, im):
        for x, y, chunkx, chunkz, _ in self._get_tile_chunks(origin, im):
            x -= origin[0]
            y -= origin[1]
            y += self.chunk_section_add * self.sections_per_chunk
            for chunky in xrange(self.sections_per_chunk):
                y -= self.chunk_section_add
                if not (y + self.chunk_section_height > 0 and y < im.size[1]):
                    continue
                render_loop(self.world, self.regionset, chunkx, chunky, chunkz, im, x, y, self.rendermode, self.textures)
