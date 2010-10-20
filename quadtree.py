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

import multiprocessing
import itertools
import os
import os.path
import hashlib
import functools
import re
import shutil
import collections
import json
import logging
import util

from PIL import Image

from optimizeimages import optimize_image


"""
This module has routines related to generating a quadtree of tiles

"""

def iterate_base4(d):
    """Iterates over a base 4 number with d digits"""
    return itertools.product(xrange(4), repeat=d)

def catch_keyboardinterrupt(func):
    """Decorator that catches a keyboardinterrupt and raises a real exception
    so that multiprocessing will propagate it properly"""
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logging.error("Ctrl-C caught!")
            raise Exception("Exiting")
        except:
            import traceback
            traceback.print_exc()
            raise
    return newfunc

class QuadtreeGen(object):
    def __init__(self, worldobj, destdir, depth=None, imgformat=None, optimizeimg=None):
        """Generates a quadtree from the world given into the
        given dest directory

        worldobj is a world.WorldRenderer object that has already been processed

        If depth is given, it overrides the calculated value. Otherwise, the
        minimum depth that contains all chunks is calculated and used.

        """
        assert(imgformat)
        self.imgformat = imgformat
        self.optimizeimg = optimizeimg

        # Make the destination dir
        if not os.path.exists(destdir):
            os.mkdir(destdir)

        if depth is None:
            # Determine quadtree depth (midpoint is always 0,0)
            for p in xrange(15):
                # Will 2^p tiles wide and high suffice?

                # X has twice as many chunks as tiles, then halved since this is a
                # radius
                xradius = 2**p
                # Y has 4 times as many chunks as tiles, then halved since this is
                # a radius
                yradius = 2*2**p
                if xradius >= worldobj.maxcol and -xradius <= worldobj.mincol and \
                        yradius >= worldobj.maxrow and -yradius <= worldobj.minrow:
                    break
            else:
                raise ValueError("Your map is waaaay too big!")

            self.p = p
        else:
            self.p = depth
            xradius = 2**depth
            yradius = 2*2**depth

        # Make new row and column ranges
        self.mincol = -xradius
        self.maxcol = xradius
        self.minrow = -yradius
        self.maxrow = yradius

        self.world = worldobj
        self.destdir = destdir

    def print_statusline(self, complete, total, level, unconditional=False):
        if unconditional:
            pass
        elif complete < 100:
            if not complete % 25 == 0:
                return
        elif complete < 1000:
            if not complete % 100 == 0:
                return
        else:
            if not complete % 1000 == 0:
                return
        logging.info("{0}/{1} tiles complete on level {2}/{3}".format(
                complete, total, level, self.p))

    def write_html(self, skipjs=False):
        """Writes out index.html, marker.js, and region.js"""
        zoomlevel = self.p
        imgformat = self.imgformat
        templatepath = os.path.join(util.get_program_path(), "template.html")

        html = open(templatepath, 'r').read()
        html = html.replace(
                "{maxzoom}", str(zoomlevel))
        html = html.replace(
                "{imgformat}", str(imgformat))
                
        with open(os.path.join(self.destdir, "index.html"), 'w') as output:
            output.write(html)

        # Write a blank image
        blank = Image.new("RGBA", (1,1))
        tileDir = os.path.join(self.destdir, "tiles")
        if not os.path.exists(tileDir): os.mkdir(tileDir)
        blank.save(os.path.join(tileDir, "blank."+self.imgformat))

        if skipjs:
            return

        # write out the default marker table
        with open(os.path.join(self.destdir, "markers.js"), 'w') as output:
            output.write("var markerData=%s" % json.dumps(self.world.POI))

        # write out the default (empty, but documented) region table
        with open(os.path.join(self.destdir, "regions.js"), 'w') as output:
            output.write('var regionData=[\n')
            output.write('  // {"color": "#FFAA00", "opacity": 0.5, "closed": true, "path": [\n')
            output.write('  //   {"x": 0, "y": 0, "z": 0},\n')
            output.write('  //   {"x": 0, "y": 10, "z": 0},\n')
            output.write('  //   {"x": 0, "y": 0, "z": 10}\n')
            output.write('  // ]},\n')
            output.write('];')
        
    def _get_cur_depth(self):
        """How deep is the quadtree currently in the destdir? This glances in
        index.html to see what maxZoom is set to.
        returns -1 if it couldn't be detected, file not found, or nothing in
        index.html matched
        """
        indexfile = os.path.join(self.destdir, "index.html")
        if not os.path.exists(indexfile):
            return -1
        matcher = re.compile(r"maxZoom:\s*(\d+)")
        p = -1
        for line in open(indexfile, "r"):
            res = matcher.search(line)
            if res:
                p = int(res.group(1))
                break
        return p

    def _increase_depth(self):
        """Moves existing tiles into place for a larger tree"""
        getpath = functools.partial(os.path.join, self.destdir, "tiles")

        # At top level of the tree:
        # quadrant 0 is now 0/3
        # 1 is now 1/2
        # 2 is now 2/1
        # 3 is now 3/0
        # then all that needs to be done is to regenerate the new top level
        for dirnum in range(4):
            newnum = (3,2,1,0)[dirnum]

            newdir = "new" + str(dirnum)
            newdirpath = getpath(newdir)

            files = [str(dirnum)+"."+self.imgformat, str(dirnum)+".hash", str(dirnum)]
            newfiles = [str(newnum)+"."+self.imgformat, str(newnum)+".hash", str(newnum)]

            os.mkdir(newdirpath)
            for f, newf in zip(files, newfiles):
                p = getpath(f)
                if os.path.exists(p):
                    os.rename(p, getpath(newdir, newf))
            os.rename(newdirpath, getpath(str(dirnum)))

    def _decrease_depth(self):
        """If the map size decreases, or perhaps the user has a depth override
        in effect, re-arrange existing tiles for a smaller tree"""
        getpath = functools.partial(os.path.join, self.destdir, "tiles")

        # quadrant 0/3 goes to 0
        # 1/2 goes to 1
        # 2/1 goes to 2
        # 3/0 goes to 3
        # Just worry about the directories here, the files at the top two
        # levels are cheap enough to replace
        if os.path.exists(getpath("0", "3")):
            os.rename(getpath("0", "3"), getpath("new0"))
            shutil.rmtree(getpath("0"))
            os.rename(getpath("new0"), getpath("0"))

        if os.path.exists(getpath("1", "2")):
            os.rename(getpath("1", "2"), getpath("new1"))
            shutil.rmtree(getpath("1"))
            os.rename(getpath("new1"), getpath("1"))

        if os.path.exists(getpath("2", "1")):
            os.rename(getpath("2", "1"), getpath("new2"))
            shutil.rmtree(getpath("2"))
            os.rename(getpath("new2"), getpath("2"))

        if os.path.exists(getpath("3", "0")):
            os.rename(getpath("3", "0"), getpath("new3"))
            shutil.rmtree(getpath("3"))
            os.rename(getpath("new3"), getpath("3"))

    def _apply_render_worldtiles(self, pool):
        """Returns an iterator over result objects. Each time a new result is
        requested, a new task is added to the pool and a result returned.
        """
        for path in iterate_base4(self.p):
            # Get the range for this tile
            colstart, rowstart = self._get_range_by_path(path)
            colend = colstart + 2
            rowend = rowstart + 4

            # This image is rendered at:
            dest = os.path.join(self.destdir, "tiles", *(str(x) for x in path))

            # And uses these chunks
            tilechunks = self._get_chunks_in_range(colstart, colend, rowstart,
                    rowend)

            # Put this in the pool
            # (even if tilechunks is empty, render_worldtile will delete
            # existing images if appropriate)
            yield pool.apply_async(func=render_worldtile, args= (tilechunks,
                colstart, colend, rowstart, rowend, dest, self.imgformat,
                self.optimizeimg))

    def _apply_render_inntertile(self, pool, zoom):
        """Same as _apply_render_worltiles but for the inntertile routine.
        Returns an iterator that yields result objects from tasks that have
        been applied to the pool.
        """
        for path in iterate_base4(zoom):
            # This image is rendered at:
            dest = os.path.join(self.destdir, "tiles", *(str(x) for x in path[:-1]))
            name = str(path[-1])

            yield pool.apply_async(func=render_innertile, args= (dest, name, self.imgformat, self.optimizeimg))

    def go(self, procs):
        """Renders all tiles"""

        curdepth = self._get_cur_depth()
        if curdepth != -1:
            if self.p > curdepth:
                logging.warning("Your map seemes to have expanded beyond its previous bounds.")
                logging.warning( "Doing some tile re-arrangements... just a sec...")
                for _ in xrange(self.p-curdepth):
                    self._increase_depth()
            elif self.p < curdepth:
                logging.warning("Your map seems to have shrunk. Re-arranging tiles, just a sec...")
                for _ in xrange(curdepth - self.p):
                    self._decrease_depth()

        # Create a pool
        if procs == 1:
            pool = FakePool()
        else:
            pool = multiprocessing.Pool(processes=procs)

        # Render the highest level of tiles from the chunks
        results = collections.deque()
        complete = 0
        total = 4**self.p
        logging.info("Rendering highest zoom level of tiles now.")
        logging.info("There are {0} tiles to render".format(total))
        logging.info("There are {0} total levels to render".format(self.p))
        logging.info("Don't worry, each level has only 25% as many tiles as the last.")
        logging.info("The others will go faster")
        for result in self._apply_render_worldtiles(pool):
            results.append(result)
            if len(results) > 10000:
                # Empty the queue before adding any more, so that memory
                # required has an upper bound
                while len(results) > 500:
                    results.popleft().get()
                    complete += 1
                    self.print_statusline(complete, total, 1)

        # Wait for the rest of the results
        while len(results) > 0:
            results.popleft().get()
            complete += 1
            self.print_statusline(complete, total, 1)

        self.print_statusline(complete, total, 1, True)

        # Now do the other layers
        for zoom in xrange(self.p-1, 0, -1):
            level = self.p - zoom + 1
            assert len(results) == 0
            complete = 0
            total = 4**zoom
            logging.info("Starting level {0}".format(level))
            for result in self._apply_render_inntertile(pool, zoom):
                results.append(result)
                if len(results) > 10000:
                    while len(results) > 500:
                        results.popleft().get()
                        complete += 1
                        self.print_statusline(complete, total, level)
            # Empty the queue
            while len(results) > 0:
                results.popleft().get()
                complete += 1
                self.print_statusline(complete, total, level)

            self.print_statusline(complete, total, level, True)

            logging.info("Done")

        pool.close()
        pool.join()

        # Do the final one right here:
        render_innertile(os.path.join(self.destdir, "tiles"), "base", self.imgformat, self.optimizeimg)

    def _get_range_by_path(self, path):
        """Returns the x, y chunk coordinates of this tile"""
        x, y = self.mincol, self.minrow
        
        xsize = self.maxcol
        ysize = self.maxrow

        for p in path:
            if p in (1, 3):
                x += xsize
            if p in (2, 3):
                y += ysize
            xsize //= 2
            ysize //= 2

        return x, y

    def _get_chunks_in_range(self, colstart, colend, rowstart, rowend):
        """Get chunks that are relevant to the tile rendering function that's
        rendering that range"""
        chunklist = []
        for row in xrange(rowstart-16, rowend+1):
            for col in xrange(colstart, colend+1):
                c = self.world.chunkmap.get((col, row), None)
                if c:
                    chunklist.append((col, row, c))
        return chunklist

@catch_keyboardinterrupt
def render_innertile(dest, name, imgformat, optimizeimg):
    """
    Renders a tile at os.path.join(dest, name)+".ext" by taking tiles from
    os.path.join(dest, name, "{0,1,2,3}.png")
    """
    imgpath = os.path.join(dest, name) + "." + imgformat
    hashpath = os.path.join(dest, name) + ".hash"

    if name == "base":
        q0path = os.path.join(dest, "0." + imgformat)
        q1path = os.path.join(dest, "1." + imgformat)
        q2path = os.path.join(dest, "2." + imgformat)
        q3path = os.path.join(dest, "3." + imgformat)
        q0hash = os.path.join(dest, "0.hash")
        q1hash = os.path.join(dest, "1.hash")
        q2hash = os.path.join(dest, "2.hash")
        q3hash = os.path.join(dest, "3.hash")
    else:
        q0path = os.path.join(dest, name, "0." + imgformat)
        q1path = os.path.join(dest, name, "1." + imgformat)
        q2path = os.path.join(dest, name, "2." + imgformat)
        q3path = os.path.join(dest, name, "3." + imgformat)
        q0hash = os.path.join(dest, name, "0.hash")
        q1hash = os.path.join(dest, name, "1.hash")
        q2hash = os.path.join(dest, name, "2.hash")
        q3hash = os.path.join(dest, name, "3.hash")

    # Check which ones exist
    if not os.path.exists(q0hash):
        q0path = None
        q0hash = None
    if not os.path.exists(q1hash):
        q1path = None
        q1hash = None
    if not os.path.exists(q2hash):
        q2path = None
        q2hash = None
    if not os.path.exists(q3hash):
        q3path = None
        q3hash = None

    # do they all not exist?
    if not (q0path or q1path or q2path or q3path):
        if os.path.exists(imgpath):
            os.unlink(imgpath)
        if os.path.exists(hashpath):
            os.unlink(hashpath)
        return
    
    # Now check the hashes
    hasher = hashlib.md5()
    if q0hash:
        hasher.update(open(q0hash, "rb").read())
    if q1hash:
        hasher.update(open(q1hash, "rb").read())
    if q2hash:
        hasher.update(open(q2hash, "rb").read())
    if q3hash:
        hasher.update(open(q3hash, "rb").read())
    if os.path.exists(hashpath):
        oldhash = open(hashpath, "rb").read()
    else:
        oldhash = None
    newhash = hasher.digest()

    if newhash == oldhash:
        # Nothing to do
        return

    # Create the actual image now
    img = Image.new("RGBA", (384, 384), (38,92,255,0))

    if q0path:
        try:
            quad0 = Image.open(q0path).resize((192,192), Image.ANTIALIAS)
            img.paste(quad0, (0,0))
        except Exception, e:
            logging.warning("Couldn't open %s. It may be corrupt, you may need to delete it. %s", q0path, e)
    if q1path:
        try:
            quad1 = Image.open(q1path).resize((192,192), Image.ANTIALIAS)
            img.paste(quad1, (192,0))
        except Exception, e:
            logging.warning("Couldn't open %s. It may be corrupt, you may need to delete it. %s", q1path, e)
    if q2path:
        try:
            quad2 = Image.open(q2path).resize((192,192), Image.ANTIALIAS)
            img.paste(quad2, (0, 192))
        except Exception, e:
            logging.warning("Couldn't open %s. It may be corrupt, you may need to delete it. %s", q2path, e)
    if q3path:
        try:
            quad3 = Image.open(q3path).resize((192,192), Image.ANTIALIAS)
            img.paste(quad3, (192, 192))
        except Exception, e:
            logging.warning("Couldn't open %s. It may be corrupt, you may need to delete it. %s", q3path, e)

    # Save it
    if imgformat == 'jpg':
        img.save(imgpath, quality=95, subsampling=0)
    else: # png
        img.save(imgpath)
        if optimizeimg:
            optimize_image(imgpath, imgformat, optimizeimg)

    with open(hashpath, "wb") as hashout:
        hashout.write(newhash)


@catch_keyboardinterrupt
def render_worldtile(chunks, colstart, colend, rowstart, rowend, path, imgformat, optimizeimg):
    """Renders just the specified chunks into a tile and save it. Unlike usual
    python conventions, rowend and colend are inclusive. Additionally, the
    chunks around the edges are half-way cut off (so that neighboring tiles
    will render the other half)

    chunks is a list of (col, row, filename) of chunk images that are relevant
    to this call

    The image is saved to path+".ext" and a hash is saved to path+".hash"

    If there are no chunks, this tile is not saved (if it already exists, it is
    deleted)

    If the hash file already exists, it is checked against the hash of each chunk.

    Standard tile size has colend-colstart=2 and rowend-rowstart=4

    There is no return value
    """
    # width of one chunk is 384. Each column is half a chunk wide. The total
    # width is (384 + 192*(numcols-1)) since the first column contributes full
    # width, and each additional one contributes half since they're staggered.
    # However, since we want to cut off half a chunk at each end (384 less
    # pixels) and since (colend - colstart + 1) is the number of columns
    # inclusive, the equation simplifies to:
    width = 192 * (colend - colstart)
    # Same deal with height
    height = 96 * (rowend - rowstart)

    # The standard tile size is 3 columns by 5 rows, which works out to 384x384
    # pixels for 8 total chunks. (Since the chunks are staggered but the grid
    # is not, some grid coordinates do not address chunks) The two chunks on
    # the middle column are shown in full, the two chunks in the middle row are
    # half cut off, and the four remaining chunks are one quarter shown.
    # The above example with cols 0-3 and rows 0-4 has the chunks arranged like this:
    #   0,0         2,0
    #         1,1
    #   0,2         2,2
    #         1,3
    #   0,4         2,4

    # Due to how the tiles fit together, we may need to render chunks way above
    # this (since very few chunks actually touch the top of the sky, some tiles
    # way above this one are possibly visible in this tile). Render them
    # anyways just in case). "chunks" should include up to rowstart-16

    # Before we render any tiles, check the hash of each image in this tile to
    # see if it's changed.
    hashpath = path + ".hash"
    imgpath = path + "." + imgformat

    if not chunks:
        # No chunks were found in this tile
        if os.path.exists(imgpath):
            os.unlink(imgpath)
        if os.path.exists(hashpath):
            os.unlink(hashpath)
        return None

    # Create the directory if not exists
    dirdest = os.path.dirname(path)
    if not os.path.exists(dirdest):
        try:
            os.makedirs(dirdest)
        except OSError, e:
            # Ignore errno EEXIST: file exists. Since this is multithreaded,
            # two processes could conceivably try and create the same directory
            # at the same time.
            import errno
            if e.errno != errno.EEXIST:
                raise

    imghash = hashlib.md5()
    for col, row, chunkfile in chunks:
        # Get the hash of this image and add it to our hash for this tile
        imghash.update(
                os.path.basename(chunkfile).split(".")[4]
                )
    digest = imghash.digest()

    if os.path.exists(hashpath):
        oldhash = open(hashpath, 'rb').read()
    else:
        oldhash = None

    if digest == oldhash:
        # All the chunks for this tile have not changed according to the hash
        return

    # Compile this image
    tileimg = Image.new("RGBA", (width, height), (38,92,255,0))

    # col colstart will get drawn on the image starting at x coordinates -(384/2)
    # row rowstart will get drawn on the image starting at y coordinates -(192/2)
    for col, row, chunkfile in chunks:
        try:
            chunkimg = Image.open(chunkfile)
            chunkimg.load()
        except Exception, e:
            # If for some reason the chunk failed to load (perhaps a previous
            # run was canceled and the file was only written half way,
            # corrupting it), then this could error.
            # Since we have no easy way of determining how this chunk was
            # generated, we need to just ignore it.
            logging.warning("Could not open chunk '{0}' ({1})".format(chunkfile,e))
            try:
                # Remove the file so that the next run will re-generate it.
                os.unlink(chunkfile)
            except OSError, e:
                import errno
                # Ignore if file doesn't exist, another task could have already
                # removed it.
                if e.errno != errno.ENOENT:
                    logging.warning("Could not remove chunk '{0}'!".format(chunkfile))
                    raise
            else:
                logging.warning("Removed the corrupt file")

            logging.warning("You will need to re-run the Overviewer to fix this chunk")
            continue

        xpos = -192 + (col-colstart)*192
        ypos = -96 + (row-rowstart)*96

        tileimg.paste(chunkimg.convert("RGB"), (xpos, ypos), chunkimg)

    # Save them
    tileimg.save(imgpath)

    if optimizeimg:
        optimize_image(imgpath, imgformat, optimizeimg)

    with open(hashpath, "wb") as hashout:
        hashout.write(digest)

class FakeResult(object):
    def __init__(self, res):
        self.res = res
    def get(self):
        return self.res
class FakePool(object):
    """A fake pool used to render things in sync. Implements a subset of
    multiprocessing.Pool"""
    def apply_async(self, func, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        result = func(*args, **kwargs)
        return FakeResult(result)
    def close(self):
        pass
    def join(self):
        pass
