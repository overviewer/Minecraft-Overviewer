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

from time import time
from math import ceil
import os
import numpy
from itertools import product

from overviewer.oil import Image
from overviewer.dispatcher import Worker

"""

canvas.py contains the Canvas and Renderer base classes, as well as
SimpleCanvas, a Canvas implementation that renders everything to one
large image.

The Canvas / Renderer interface lets you implement a Renderer that
turns arbitrary, opaque "render source" objects into images rendered
on a virtual canvas. The Renderer informs the Canvas of what render
sources are available, where they are located, and how big a virtual
canvas is needed. The Canvas object then uses this information to
decide how to render this virtual canvas, and provides the Renderer
with image objects to render to.

"""

class Renderer(object):
    """The base class for Renderer objects, which provides default
    implementations for some methods. Renderer objects are designed to
    be used in conjunction with Canvas objects."""
    
    def get_render_sources(self):
        """Return an iterator over all render sources. This are opaque
        objects that are passed back to the Renderer to decide how to
        proceed with rendering."""
        return []

    def get_render_source_mtime(self, source):
        """Return the last modified time of a render source. Canvas
        objects can use this information to do incremental updates."""
        raise NotImplementedError("Renderer.get_render_source_mtime")

    def get_rect_for_render_source(self, source):
        """Return a (xmin, ymin, xmax, ymax) tuple representing the
        rectangle on the virtual canvas this render source is
        contained inside."""
        raise NotImplementedError("Renderer.get_rect_for_render_source")

    def get_render_sources_in_rect(self, rect):
        """Return an iterator over all render sources that might
        possibly be contained inside the given (xmin, ymin, xmax,
        ymax) rect."""
        return []

    def get_full_rect(self):
        """Return a (xmin, ymin, xmax, ymax) tuple representing the
        entire virtual canvas this Renderer is requesting."""
        raise NotImplementedError("Renderer.get_full_rect")

    def render(self, origin, im):
        """Render to the given image object, assuming the origin of
        the image is located at the given origin on the virtual
        canvas."""
        raise NotImplementedError("Renderer.render")

class Canvas(Worker):
    """The base class for Canvas objects, which provides default
    implementations for some methods. Canvas objects also must
    implement the Worker interface. Canvas objects are designed to be
    used in conjunction with Renderer objects."""
    pass

class SingleImageCanvas(Canvas):
    """This class renders a single image of the given size at the given origin
    using the given renderer.
    
    """
    def __init__(self, origin, size, renderer, outputpath):
        self.size = size
        self.origin = origin
        self.renderer = renderer
        self.outputpath = outputpath

    def iterate_work_items(self, phase):
        return [(0, ())]

    def do_work(self, workobj):
        if workobj != 0:
            raise ValueError()

        image = Image(*self.size)
        self.renderer.render(self.origin, image)
        image.save(self.outputpath)

    def get_phase_length(self,_):
        return 1

class SimpleCanvas(Canvas):
    # how big the tiles should be
    region_size = 384 * 2

    def __init__(self, renderer, output, rect=None):
        self.renderer = renderer
        self.output = output
        self.started = time()

        if rect:
            self.rect = rect
        else:
            self.rect = renderer.get_full_rect()
        self.size = (self.rect[2] - self.rect[0], self.rect[3] - self.rect[1])
        self.numx = int(ceil(float(self.size[0]) / self.region_size))
        self.numy = int(ceil(float(self.size[1]) / self.region_size))

        try:
            out_mtime = os.path.getmtime(self.output)
        except os.error:
            out_mtime = 0

        self.needs_update = numpy.zeros((self.numx, self.numy), dtype=numpy.bool)
        for source in self.renderer.get_render_sources():
            if self.renderer.get_render_source_mtime(source) <= out_mtime:
                continue
            rect = self.renderer.get_rect_for_render_source(source)
            rect = (rect[0] - self.rect[0], rect[1] - self.rect[1], rect[2] - self.rect[0], rect[3] - self.rect[1])
            minx = rect[0] / self.region_size
            miny = rect[1] / self.region_size
            maxx = int(ceil(float(rect[2]) / self.region_size))
            maxy = int(ceil(float(rect[3]) / self.region_size))
            for x, y in product(xrange(minx, maxx), xrange(miny, maxy)):
                if x < 0 or x >= self.numx or y < 0 or y >= self.numy:
                    continue
                self.needs_update[x, y] = 1

    def get_num_phases(self):
        if self.needs_update.sum() == 0:
            return 0
        return 2

    def get_phase_length(self, phase):
        if phase == 0:
            return self.needs_update.sum()
        return 1

    def iterate_work_items(self, phase):
        if phase == 1:
            # do the one final thing
            yield (None, [])
        else:
            # do all the regions
            for x in xrange(self.numx):
                for y in xrange(self.numy):
                    if self.needs_update[x, y]:
                        yield ((x, y), [])

    def do_work(self, item):
        if item is None:
            try:
                bigim = Image.open(self.output)
            except IOError:
                bigim = Image.new("RGBA", self.size)
            for x in xrange(self.numx):
                for y in xrange(self.numy):
                    if self.needs_update[x, y]:
                        path = self.output + '.' + str(x) + '.' + str(y) + '.png'
                        im = Image.open(path)
                        bigim.paste(im, (x * self.region_size, y * self.region_size))
                        os.remove(path)
            bigim.save(self.output)
            os.utime(self.output, (self.started, self.started))
            return

        x, y = item
        origin = (self.rect[0] + x * self.region_size, self.rect[1] + y * self.region_size)
        size = tuple([min([self.rect[i + 2] - origin[i], self.region_size]) for i in range(2)])
        im = Image.new("RGBA", size)
        self.renderer.render(origin, im)
        im.save(self.output + '.' + str(x) + '.' + str(y) + '.png')
