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
import util
import cPickle
import stat
import errno 
import time
from time import gmtime, strftime, sleep

from PIL import Image

import nbt
import chunk
from c_overviewer import get_render_mode_inheritance
from optimizeimages import optimize_image
import composite


"""
This module has routines related to generating a quadtree of tiles

"""

def iterate_base4(d):
    """Iterates over a base 4 number with d digits"""
    return itertools.product(xrange(4), repeat=d)
   
class QuadtreeGen(object):
    def __init__(self, worldobj, destdir, bgcolor, depth=None, tiledir=None, forcerender=False, imgformat=None, imgquality=95, optimizeimg=None, rendermode="normal"):
        """Generates a quadtree from the world given into the
        given dest directory

        worldobj is a world.WorldRenderer object that has already been processed

        If depth is given, it overrides the calculated value. Otherwise, the
        minimum depth that contains all chunks is calculated and used.

        """
        assert(imgformat)
        self.forcerender = forcerender
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
                raise ValueError("Your map is waaaay too big! Use the 'zoom' option in 'settings.py'.")

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
        
    def go(self, procs):
        """Processing before tile rendering"""

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
        
    def get_chunks_in_range(self, colstart, colend, rowstart, rowend):
        """Get chunks that are relevant to the tile rendering function that's
        rendering that range"""
        chunklist = []
        unconvert_coords = self.world.unconvert_coords
        #get_region_path = self.world.get_region_path
        get_region = self.world.regionfiles.get
        regionx = None
        regiony = None
        c = None
        mcr = None
        for row in xrange(rowstart-16, rowend+1):
            for col in xrange(colstart, colend+1):
                # due to how chunks are arranged, we can only allow
                # even row, even column or odd row, odd column
                # otherwise, you end up with duplicates!
                if row % 2 != col % 2:
                    continue
                
                chunkx, chunky = unconvert_coords(col, row)

                regionx_ = chunkx//32
                regiony_ = chunky//32
                if regionx_ != regionx or regiony_ != regiony:
                    regionx = regionx_
                    regiony = regiony_
                    _, _, c, mcr = get_region((regionx, regiony),(None,None,None,None))
                    
                if c is not None and mcr.chunkExists(chunkx,chunky):
                    chunklist.append((col, row, chunkx, chunky, c))
                    
        return chunklist   
        
    def get_worldtiles(self):
        """Returns an iterator over the tiles of the most detailed layer
        """
        for path in iterate_base4(self.p):
            # Get the range for this tile
            colstart, rowstart = self._get_range_by_path(path)
            colend = colstart + 2
            rowend = rowstart + 4   
            
            # This image is rendered at(relative to the worker's destdir):
            tilepath = [str(x) for x in path]
            tilepath = os.sep.join(tilepath)
            #logging.debug("this is rendered at %s", dest)
        
            # Put this in the batch to be submited to the pool     
            yield [self,colstart, colend, rowstart, rowend, tilepath]
        
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
            tile_mtime =  os.stat(imgpath)[stat.ST_MTIME];
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



    def render_worldtile(self, chunks, colstart, colend, rowstart, rowend, path, poi_queue=None):
        """Renders just the specified chunks into a tile and save it. Unlike usual
        python conventions, rowend and colend are inclusive. Additionally, the
        chunks around the edges are half-way cut off (so that neighboring tiles
        will render the other half)

        chunks is a list of (col, row, chunkx, chunky, filename) of chunk
        images that are relevant to this call (with their associated regions)

        The image is saved to path+"."+self.imgformat

        If there are no chunks, this tile is not saved (if it already exists, it is
        deleted)

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

        imgpath = path + "." + self.imgformat
        world = self.world
        #stat the file, we need to know if it exists or it's mtime
        try:    
            tile_mtime =  os.stat(imgpath)[stat.ST_MTIME];
        except OSError, e:
            if e.errno != errno.ENOENT:
                raise
            tile_mtime = None
            
        if not chunks:
            # No chunks were found in this tile
            if tile_mtime is not None:
                os.unlink(imgpath)
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
                if e.errno != errno.EEXIST:
                    raise
        
        # check chunk mtimes to see if they are newer
        try:
            needs_rerender = False
            get_region_mtime = world.get_region_mtime
            for col, row, chunkx, chunky, regionfile in chunks:
                region, regionMtime = get_region_mtime(regionfile)

                # don't even check if it's not in the regionlist
                if self.world.regionlist and os.path.abspath(region._filename) not in self.world.regionlist:
                    continue

                # bail early if forcerender is set
                if self.forcerender:
                    needs_rerender = True
                    break
                
                # check region file mtime first. 
                if regionMtime <= tile_mtime:
                    continue
               
                # checking chunk mtime
                if region.get_chunk_timestamp(chunkx, chunky) > tile_mtime:
                    needs_rerender = True
                    break
            
            # if after all that, we don't need a rerender, return
            if not needs_rerender:
                return None
        except OSError:
            # couldn't get tile mtime, skip check
            pass
        
        #logging.debug("writing out worldtile {0}".format(imgpath))

        # Compile this image
        tileimg = Image.new("RGBA", (width, height), self.bgcolor)

        world = self.world
        rendermode = self.rendermode
        # col colstart will get drawn on the image starting at x coordinates -(384/2)
        # row rowstart will get drawn on the image starting at y coordinates -(192/2)
        for col, row, chunkx, chunky, regionfile in chunks:
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
