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
import functools
import re
import shutil
import collections
import json
import logging
import cPickle
import stat
import errno 
import time
import random
from time import gmtime, strftime, sleep

from PIL import Image

from . import nbt
from . import chunk
from .optimizeimages import optimize_image
from c_overviewer import get_render_mode_inheritance
import composite


"""
This module has routines related to generating a quadtree of tiles

"""

def iterate_base4(d):
    """Iterates over a base 4 number with d digits"""
    return itertools.product(xrange(4), repeat=d)
   
class QuadtreeGen(object):
    def __init__(self, worldobj, destdir, bgcolor="#1A1A1A", depth=None, tiledir=None, forcerender=False, imgformat='png', imgquality=95, optimizeimg=None, rendermode="normal", rerender_prob=0.0):
        """Generates a quadtree from the world given into the
        given dest directory

        worldobj is a world.WorldRenderer object that has already been processed

        If depth is given, it overrides the calculated value. Otherwise, the
        minimum depth that contains all chunks is calculated and used.

        """
        self.forcerender = forcerender
        self.rerender_probability = rerender_prob
        self.imgformat = imgformat
        self.imgquality = imgquality
        self.optimizeimg = optimizeimg
        self.bgcolor = bgcolor
        self.rendermode = rendermode
        
        # force png renderformat if we're using an overlay mode
        if 'overlay' in get_render_mode_inheritance(rendermode):
            self.imgformat = "png"

        # Make the destination dir
        if not os.path.exists(destdir):
            os.makedirs(os.path.abspath(destdir))
        if tiledir is None:
            tiledir = rendermode
        self.tiledir = tiledir        
        
        if depth is None:
            # Determine quadtree depth (midpoint is always 0,0)
            for p in xrange(33):
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
            
            if p < 15:
                self.p = p
            else:
                raise ValueError("Your map is waaaay too big! Use the 'zoom' option in 'settings.py'. Overviewer is estimating %i zoom levels, but you probably want less." % (p,))

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
        self.full_tiledir = os.path.join(destdir, tiledir)
        
    def _get_cur_depth(self):
        """How deep is the quadtree currently in the destdir? This glances in
        config.js to see what maxZoom is set to.
        returns -1 if it couldn't be detected, file not found, or nothing in
        config.js matched
        """
        indexfile = os.path.join(self.destdir, "overviewerConfig.js")
        if not os.path.exists(indexfile):
            return -1
        matcher = re.compile(r"zoomLevels(?:\'|\")\s*:\s*(\d+)")
        p = -1
        for line in open(indexfile, "r"):
            res = matcher.search(line)
            if res:
                p = int(res.group(1))
                break
        return p

    def _increase_depth(self):
        """Moves existing tiles into place for a larger tree"""
        getpath = functools.partial(os.path.join, self.destdir, self.tiledir)

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

            files = [str(dirnum)+"."+self.imgformat, str(dirnum)]
            newfiles = [str(newnum)+"."+self.imgformat, str(newnum)]

            os.mkdir(newdirpath)
            for f, newf in zip(files, newfiles):
                p = getpath(f)
                if os.path.exists(p):
                    os.rename(p, getpath(newdir, newf))
            os.rename(newdirpath, getpath(str(dirnum)))

    def _decrease_depth(self):
        """If the map size decreases, or perhaps the user has a depth override
        in effect, re-arrange existing tiles for a smaller tree"""
        getpath = functools.partial(os.path.join, self.destdir, self.tiledir)

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

        # Delete the files in the top directory to make sure they get re-created.
        files = [str(num)+"."+self.imgformat for num in xrange(4)] + ["base." + self.imgformat]
        for f in files:
            try:
                os.unlink(getpath(f))
            except OSError, e:
                pass # doesn't exist maybe?

    def check_depth(self):
        """Ensure the current quadtree is the correct depth. If it's not,
        employ some simple re-arranging of tiles to save on computation.
        
        """

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
    
    
    def get_chunks_for_tile(self, tile):
        """Get chunks that are relevant to the given tile
        
        Returns a list of chunks where each item is
        (col, row, chunkx, chunky, regionobj)
        """

        chunklist = []

        unconvert_coords = self.world.unconvert_coords
        get_region = self.world.regionfiles.get

        # Cached region object for consecutive iterations
        regionx = None
        regiony = None
        c = None
        mcr = None

        rowstart = tile.row
        rowend = rowstart+4
        colstart = tile.col
        colend = colstart+2

        # Start 16 rows up from the actual tile's row, since chunks are that tall.
        # Also, every other tile doesn't exist due to how chunks are arranged. See
        # http://docs.overviewer.org/en/latest/design/designdoc/#chunk-addressing
        for row, col in itertools.product(
                xrange(rowstart-16, rowend+1),
                xrange(colstart, colend+1)
                ):
            if row % 2 != col % 2:
                continue
            
            chunkx, chunky = unconvert_coords(col, row)

            regionx_ = chunkx//32
            regiony_ = chunky//32
            if regionx_ != regionx or regiony_ != regiony:
                regionx = regionx_
                regiony = regiony_
                _, _, fname, mcr = get_region((regionx, regiony),(None,None,None,None))
                
            if fname is not None and mcr.chunkExists(chunkx,chunky):
                chunklist.append((col, row, chunkx, chunky, mcr))
                    
        return chunklist   
        
    def get_worldtiles(self):
        """Returns an iterator over the tiles of the most detailed layer that
        need to be rendered

        """
        # This quadtree object gets replaced by the caller in rendernode.py,
        # but we still have to let them know which quadtree this tile belongs
        # to. Hence returning both self and the tile.
        return ([self, tile] for tile in self.scan_chunks())
        
    def get_innertiles(self,zoom):
        """Same as get_worldtiles but for the inntertile routine.
        """    
        for path in iterate_base4(zoom):
            # This image is rendered at(relative to the worker's destdir):
            tilepath = [str(x) for x in path[:-1]]
            tilepath = os.sep.join(tilepath)
            name = str(path[-1])
  
            yield [self,tilepath, name]
    
    def render_innertile(self, dest, name):
        """
        Renders a tile at os.path.join(dest, name)+".ext" by taking tiles from
        os.path.join(dest, name, "{0,1,2,3}.png")
        """
        imgformat = self.imgformat 
        imgpath = os.path.join(dest, name) + "." + imgformat

        if name == "base":
            quadPath = [[(0,0),os.path.join(dest, "0." + imgformat)],[(192,0),os.path.join(dest, "1." + imgformat)], [(0, 192),os.path.join(dest, "2." + imgformat)],[(192,192),os.path.join(dest, "3." + imgformat)]]
        else:
            quadPath = [[(0,0),os.path.join(dest, name, "0." + imgformat)],[(192,0),os.path.join(dest, name, "1." + imgformat)],[(0, 192),os.path.join(dest, name, "2." + imgformat)],[(192,192),os.path.join(dest, name, "3." + imgformat)]]    
       
        #stat the tile, we need to know if it exists or it's mtime
        try:    
            tile_mtime =  os.stat(imgpath)[stat.ST_MTIME]
        except OSError, e:
            if e.errno != errno.ENOENT:
                raise
            tile_mtime = None
            
        #check mtimes on each part of the quad, this also checks if they exist
        needs_rerender = (tile_mtime is None) or self.forcerender
        quadPath_filtered = []
        for path in quadPath:
            try:
                quad_mtime = os.stat(path[1])[stat.ST_MTIME]; 
                quadPath_filtered.append(path)
                if quad_mtime > tile_mtime:     
                    needs_rerender = True            
            except OSError:
                # We need to stat all the quad files, so keep looping
                pass      
        # do they all not exist?
        if quadPath_filtered == []:
            if tile_mtime is not None:
                os.unlink(imgpath)
            return
        # quit now if we don't need rerender
        if not needs_rerender:
            return    
        #logging.debug("writing out innertile {0}".format(imgpath))

        # Create the actual image now
        img = Image.new("RGBA", (384, 384), self.bgcolor)
        
        # we'll use paste (NOT alpha_over) for quadtree generation because
        # this is just straight image stitching, not alpha blending
        
        for path in quadPath_filtered:
            try:
                quad = Image.open(path[1]).resize((192,192), Image.ANTIALIAS)
                img.paste(quad, path[0])
            except Exception, e:
                logging.warning("Couldn't open %s. It may be corrupt, you may need to delete it. %s", path[1], e)

        # Save it
        if self.imgformat == 'jpg':
            img.save(imgpath, quality=self.imgquality, subsampling=0)
        else: # png
            img.save(imgpath)
            
        if self.optimizeimg:
            optimize_image(imgpath, self.imgformat, self.optimizeimg)

    def render_worldtile(self, tile, check_tile=False):
        """Renders the given tile. All the other relevant information is
        already stored in this quadtree object or in self.world.

        This function is typically called in the child process. The tile is
        assumed to need rendering unless the check_tile flag is given.

        If check_tile is true, the mtimes of the chunk are compared with the
        mtime of this tile and the tile is conditionally rendered.

        The image is rendered and saved to disk in the place this quadtree is
        configured to store images.

        If there are no chunks, this tile is not saved. If this is the case but
        the tile exists, it is deleted

        There is no return value
        """    

        poi_queue = self.world.poi_q

        imgpath = tile.get_filepath(self.full_tiledir, self.imgformat)

        # Calculate which chunks are relevant to this tile
        chunks = self.get_chunks_for_tile(tile)

        world = self.world

        tile_mtime = None
        if check_tile:
            #stat the file, we need to know if it exists or it's mtime
            try:    
                tile_mtime =  os.stat(imgpath)[stat.ST_MTIME]
            except OSError, e:
                # ignore only if the error was "file not found"
                if e.errno != errno.ENOENT:
                    raise
            
        if not chunks:
            # No chunks were found in this tile
            if not check_tile:
                logging.warning("Tile %s was requested for render, but no chunks found! This may be a bug", tile)
            try:
                os.unlink(imgpath)
            except OSError, e:
                # ignore only if the error was "file not found"
                if e.errno != errno.ENOENT:
                    raise
            return

        # Create the directory if not exists
        dirdest = os.path.dirname(imgpath)
        if not os.path.exists(dirdest):
            try:
                os.makedirs(dirdest)
            except OSError, e:
                # Ignore errno EEXIST: file exists. Due to a race condition,
                # two processes could conceivably try and create the same
                # directory at the same time
                if e.errno != errno.EEXIST:
                    raise
        
        if check_tile:
            # Look at all the chunks that touch this tile and their mtimes to
            # determine if this tile actually needs rendering
            try:
                needs_rerender = False
                get_region_mtime = world.get_region_mtime
                
                for col, row, chunkx, chunky, region in chunks:

                    # don't even check if it's not in the regionlist
                    if self.world.regionlist and os.path.abspath(region._filename) not in self.world.regionlist:
                        continue

                    # bail early if forcerender is set
                    if self.forcerender:
                        needs_rerender = True
                        break
                    
                    # checking chunk mtime
                    if region.get_chunk_timestamp(chunkx, chunky) > tile_mtime:
                        needs_rerender = True
                        break
                
                # stochastic render check
                if not needs_rerender and self.rerender_probability > 0.0 and random.uniform(0, 1) < self.rerender_probability:
                    needs_rerender = True
                
                # if after all that, we don't need a rerender, return
                if not needs_rerender:
                    return
            except OSError:
                # couldn't get tile mtime, skip check and assume it does
                pass
        
        # We have all the necessary info and this tile has passed the checks
        # and should be rendered. So do it!

        #logging.debug("writing out worldtile {0}".format(imgpath))

        # Compile this image
        tileimg = Image.new("RGBA", (384, 384), self.bgcolor)

        rendermode = self.rendermode
        colstart = tile.col
        rowstart = tile.row
        # col colstart will get drawn on the image starting at x coordinates -(384/2)
        # row rowstart will get drawn on the image starting at y coordinates -(192/2)
        for col, row, chunkx, chunky, region in chunks:
            xpos = -192 + (col-colstart)*192
            ypos = -96 + (row-rowstart)*96

            # draw the chunk!
            try:
                a = chunk.ChunkRenderer((chunkx, chunky), world, rendermode, poi_queue)
                a.chunk_render(tileimg, xpos, ypos, None)
            except chunk.ChunkCorrupt:
                # an error was already printed
                pass
        
        # Save them
        if self.imgformat == 'jpg':
            tileimg.save(imgpath, quality=self.imgquality, subsampling=0)
        else: # png
            tileimg.save(imgpath)
        #Add tile to list of rendered tiles
        poi_queue.put(['rendered',imgpath])

        if self.optimizeimg:
            optimize_image(imgpath, self.imgformat, self.optimizeimg)

    def scan_chunks(self):
        """Scans the chunks of the world object and produce an iterator over
        the tiles that need to be rendered.

        """

        depth = self.p

        dirty = DirtyTiles(depth)

        # For each chunk, do this:
        #   For each tile that the chunk touches, do this:
        #       Compare the last modified time of the chunk and tile. If the
        #       tile is older, mark it in a DirtyTiles object as dirty.
        #
        # IDEA: check last render time against mtime of the region to short
        # circuit checking mtimes of all chunks in a region
        for chunkx, chunky, chunkmtime in self.world.iterate_chunk_metadata():

            chunkcol, chunkrow = self.world.convert_coords(chunkx, chunky)
            #logging.debug("Looking at chunk %s,%s", chunkcol, chunkrow)

            # find tile coordinates
            tilex = chunkcol - chunkcol % 2
            tiley = chunkrow - chunkrow % 4

            if chunkcol % 2 == 0:
                # This chunk is half-in one column and half-in another column.
                # tilex is the right one, also do tilex-2
                x_tiles = 2
            else:
                x_tiles = 1

            # The tile at tilex,tiley obviously contains chunk, but so do the
            # next 4 tiles down because chunks are very tall
            for i in xrange(x_tiles):
                for j in xrange(5):
                    tile = Tile.compute_path(tilex-2*i, tiley+4*j, depth)

                    tile_path = tile.get_filepath(self.full_tiledir, self.imgformat)
                    try:
                        tile_mtime = os.stat(tile_path)[stat.ST_MTIME]
                    except OSError, e:
                        if e.errno != errno.ENOENT:
                            raise
                        tile_mtime = 0
                    #logging.debug("tile %s(%s) vs chunk %s,%s (%s)",
                    #        tile, tile_mtime, chunkcol, chunkrow, chunkmtime)
                    if tile_mtime < chunkmtime:
                        dirty.set_dirty(tile.path)
                        #logging.debug("	Setting tile as dirty. Will render.")

        # Now that we know which tiles need rendering, return an iterator over them
        return (Tile.from_path(tpath) for tpath in dirty.iterate_dirty())


class DirtyTiles(object):
    """This tree holds which tiles need rendering.
    Each instance is a node, and the root of a subtree.

    Each node knows its "level", which corresponds to the zoom level where 0 is
    the inner-most (most zoomed in) tiles.

    Instances hold the clean/dirty state of their children. Leaf nodes are
    images and do not physically exist in the tree, level 1 nodes keep track of
    leaf image state. Level 2 nodes keep track of level 1 state, and so fourth.

    In attempt to keep things memory efficient, subtrees that are completely
    dirty are collapsed
    
    """
    def __init__(self, level):
        """Initialize a new node of the tree at the specified level

        """
        self.level = level

        # the self.children array holds the 4 children of this node. This
        # follows the same quadtree convention as elsewhere: children 0, 1, 2,
        # 3 are the upper-left, upper-right, lower-left, and lower-right
        # respectively
        # Values are:
        # False
        #   All children down this subtree are clean
        # True
        #   All children down this subtree are dirty
        # A DirtyTileTree instance
        #   the instance defines which children down that subtree are
        #   clean/dirty.
        # A node with level=1 cannot have a DirtyTileTree instance in its
        # children since its leaves are images, not more tree
        self.children = [False] * 4

    def set_dirty(self, path):
        """Marks the requested leaf node as "dirty".
        
        Path is an iterable of integers representing the path to the leaf node
        that is requested to be marked as dirty.
        
        """
        path = list(path)
        assert len(path) == self.level
        path.reverse()
        self._set_dirty_helper(path)

    def _set_dirty_helper(self, path):
        """Recursive call for set_dirty()

        Expects path to be a list in reversed order

        If *all* the nodes below this one are dirty, this function returns
        true. Otherwise, returns None.

        """

        if self.level == 1:
            # Base case
            self.children[path[0]] = True

            # Check to see if all children are dirty
            if all(self.children):
                return True
        else:
            # Recursive case

            childnum = path.pop()
            child = self.children[childnum]

            if child == False:
                # Create a new node
                child = self.__class__(self.level-1)
                child._set_dirty_helper(path)
                self.children[childnum] = child
            elif child == True:
                # Every child is already dirty. Nothing to do.
                return
            else:
                # subtree is mixed clean/dirty. Recurse
                ret = child._set_dirty_helper(path)
                if ret:
                    # Child says it's completely dirty, so we can purge the
                    # subtree and mark it as dirty. The subtree will be garbage
                    # collected when this method exits.
                    self.children[childnum] = True

                    # Since we've marked an entire sub-tree as dirty, we may be
                    # able to signal to our parent
                    if all(x is True for x in self.children):
                        return True

    def iterate_dirty(self):
        """Returns an iterator over every dirty tile in this subtree. Each item
        yielded is a sequence of integers representing the quadtree path to the
        dirty tile. Yielded sequences are of length self.level.

        """
        return (reversed(rpath) for rpath in self._iterate_dirty_helper())

    def _iterate_dirty_helper(self):
        if self.level == 1:
            # Base case
            if self.children[0]: yield [0]
            if self.children[1]: yield [1]
            if self.children[2]: yield [2]
            if self.children[3]: yield [3]

        else:
            # Higher levels:
            for c, child in enumerate(self.children):
                if child == True:
                    # All dirty down this subtree, iterate over every leaf
                    for x in iterate_base4(self.level-1):
                        x = list(x)
                        x.append(c)
                        yield x
                elif child != False:
                    # Mixed dirty/clean down this subtree, recurse
                    for path in child._iterate_dirty_helper():
                        path.append(c)
                        yield path

class Tile(object):
    """A simple container class that represents a single render-tile.

    A render-tile is a tile that is rendered, not a tile composed of other
    tiles.

    """
    __slots__ = ("col", "row", "path")
    def __init__(self, col, row, path):
        """Initialize the tile obj with the given parameters. It's probably
        better to use one of the other constructors though

        """
        self.col = col
        self.row = row
        self.path = tuple(path)

    def __repr__(self):
        return "%s(%r,%r,%r)" % (self.__class__.__name__, self.col, self.row, self.path)

    def __eq__(self,other):
        return self.col == other.col and self.row == other.row and tuple(self.path) == tuple(other.path)

    def __ne__(self, other):
        return not self == other

    def get_filepath(self, tiledir, imgformat):
        """Returns the path to this file given the directory to the tiles

        """
        path = os.path.join(tiledir, *(str(x) for x in self.path))
        imgpath = path + "." + imgformat
        return imgpath

    @classmethod
    def from_path(cls, path):
        """Constructor that takes a path and computes the col,row address of
        the tile and constructs a new tile object.

        """
        path = tuple(path)

        depth = len(path)

        # Radius of the world in chunk cols/rows
        # (Diameter in X is 2**depth, divided by 2 for a radius, multiplied by
        # 2 for 2 chunks per tile. Similarly for Y)
        xradius = 2**depth
        yradius = 2*2**depth

        col = -xradius
        row = -yradius
        xsize = xradius
        ysize = yradius

        for p in path:
            if p in (1,3):
                col += xsize
            if p in (2,3):
                row += ysize
            xsize //= 2
            ysize //= 2

        return cls(col, row, path)

    @classmethod
    def compute_path(cls, col, row, depth):
        """Constructor that takes a col,row of a tile and computes the path. 

        """
        assert col % 2 == 0
        assert row % 4 == 0

        xradius = 2**depth
        yradius = 2*2**depth

        colbounds = [-xradius, xradius]
        rowbounds = [-yradius, yradius]

        path = []

        for level in xrange(depth):
            # Strategy: Find the midpoint of this level, and determine which
            # quadrant this row/col is in. Then set the bounds to that level
            # and repeat

            xmid = (colbounds[1] + colbounds[0]) // 2
            ymid = (rowbounds[1] + rowbounds[0]) // 2

            if col < xmid:
                if row < ymid:
                    path.append(0)
                    colbounds[1] = xmid
                    rowbounds[1] = ymid
                else:
                    path.append(2)
                    colbounds[1] = xmid
                    rowbounds[0] = ymid
            else:
                if row < ymid:
                    path.append(1)
                    colbounds[0] = xmid
                    rowbounds[1] = ymid
                else:
                    path.append(3)
                    colbounds[0] = xmid
                    rowbounds[0] = ymid

        return cls(col, row, path)
