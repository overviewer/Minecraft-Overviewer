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

import functools
import os
import os.path
import multiprocessing
import sys
import logging

import numpy

import chunk
import nbt

"""
This module has routines related to generating all the chunks for a world
and for extracting information about available worlds

"""

base36decode = functools.partial(int, base=36)


def _convert_coords(chunks):
    """Takes the list of (chunkx, chunky, chunkfile) where chunkx and chunky
    are in the chunk coordinate system, and figures out the row and column in
    the image each one should be.

    returns mincol, maxcol, minrow, maxrow, chunks_translated
    chunks_translated is a list of (col, row, filename)
    """
    chunks_translated = []
    # columns are determined by the sum of the chunk coords, rows are the
    # difference
    item = chunks[0]
    mincol = maxcol = item[0] + item[1]
    minrow = maxrow = item[1] - item[0]
    for c in chunks:
        col = c[0] + c[1]
        mincol = min(mincol, col)
        maxcol = max(maxcol, col)
        row = c[1] - c[0]
        minrow = min(minrow, row)
        maxrow = max(maxrow, row)
        chunks_translated.append((col, row, c[2]))

    return mincol, maxcol, minrow, maxrow, chunks_translated


def base36encode(number, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    '''
    Convert an integer to a base36 string.
    '''
    if not isinstance(number, (int, long)):
        raise TypeError('number must be an integer')
    
    newn = abs(number)
 
    # Special case for zero
    if number == 0:
        return '0'
 
    base36 = ''
    while newn != 0:
        newn, i = divmod(newn, len(alphabet))
        base36 = alphabet[i] + base36

    if number < 0:
        return "-" + base36
    return base36

class WorldRenderer(object):
    """Renders a world's worth of chunks.
    worlddir is the path to the minecraft world
    cachedir is the path to a directory that should hold the resulting images.
    It may be the same as worlddir (which used to be the default).
    
    If chunklist is given, it is assumed to be an iterator over paths to chunk
    files to update. If it includes a trailing newline, it is stripped, so you
    can pass in file handles just fine.
    """
    def __init__(self, worlddir, cachedir, chunklist=None, lighting=False, night=False):
        self.worlddir = worlddir
        self.caves = False
        self.lighting = lighting or night
        self.night = night
        self.cachedir = cachedir

        self.chunklist = chunklist

        #  stores Points Of Interest to be mapped with markers
        #  a list of dictionaries, see below for an example
        self.POI = []

    def _get_chunk_renderset(self):
        """Returns a set of (col, row) chunks that should be rendered. Returns
        None if all chunks should be rendered"""
        if not self.chunklist:
            return None
        
        # Get a list of the (chunks, chunky, filename) from the passed in list
        # of filenames
        chunklist = []
        for path in self.chunklist:
            if path.endswith("\n"):
                path = path[:-1]
            f = os.path.basename(path)
            if f and f.startswith("c.") and f.endswith(".dat"):
                p = f.split(".")
                chunklist.append((base36decode(p[1]), base36decode(p[2]),
                    path))

        if not chunklist:
            logging.error("No valid chunks specified in your chunklist!")
            logging.error("HINT: chunks are in your world directory and have names of the form 'c.*.*.dat'")
            sys.exit(1)

        # Translate to col, row coordinates
        _, _, _, _, chunklist = _convert_coords(chunklist)

        # Build a set from the col, row pairs
        inclusion_set = set()
        for col, row, filename in chunklist:
            inclusion_set.add((col, row))

        return inclusion_set
    
    def get_chunk_path(self, chunkX, chunkY):
        """Returns the path to the chunk file at (chunkX, chunkY), if
        it exists."""
        
        chunkFile = "%s/%s/c.%s.%s.dat" % (base36encode(chunkX % 64),
                                           base36encode(chunkY % 64),
                                           base36encode(chunkX),
                                           base36encode(chunkY))
        
        return os.path.join(self.worlddir, chunkFile)
    
    def findTrueSpawn(self):
        """Adds the true spawn location to self.POI.  The spawn Y coordinate
        is almost always the default of 64.  Find the first air block above
        that point for the true spawn location"""

        ## read spawn info from level.dat
        data = nbt.load(os.path.join(self.worlddir, "level.dat"))[1]
        spawnX = data['Data']['SpawnX']
        spawnY = data['Data']['SpawnY']
        spawnZ = data['Data']['SpawnZ']
   
        ## The chunk that holds the spawn location 
        chunkX = spawnX/16
        chunkY = spawnZ/16

        ## The filename of this chunk
        chunkFile = self.get_chunk_path(chunkX, chunkY)

        data=nbt.load(chunkFile)[1]
        level = data['Level']
        blockArray = numpy.frombuffer(level['Blocks'], dtype=numpy.uint8).reshape((16,16,128))

        ## The block for spawn *within* the chunk
        inChunkX = spawnX - (chunkX*16)
        inChunkZ = spawnZ - (chunkY*16)

        ## find the first air block
        while (blockArray[inChunkX, inChunkZ, spawnY] != 0):
            spawnY += 1


        self.POI.append( dict(x=spawnX, y=spawnY, z=spawnZ, msg="Spawn"))

    def go(self, procs):
        """Starts the render. This returns when it is finished"""
        
        logging.info("Scanning chunks")
        raw_chunks = self._find_chunkfiles()
        logging.debug("Done scanning chunks")

        # Translate chunks to our diagonal coordinate system
        mincol, maxcol, minrow, maxrow, chunks = _convert_coords(raw_chunks)
        del raw_chunks # Free some memory

        self.chunkmap = self._render_chunks_async(chunks, procs)

        self.mincol = mincol
        self.maxcol = maxcol
        self.minrow = minrow
        self.maxrow = maxrow

        self.findTrueSpawn()

    def _find_chunkfiles(self):
        """Returns a list of all the chunk file locations, and the file they
        correspond to.
        
        Returns a list of (chunkx, chunky, filename) where chunkx and chunky are
        given in chunk coordinates. Use convert_coords() to turn the resulting list
        into an oblique coordinate system.
        
        Usually this scans the given worlddir, but will use the chunk list
        given to the constructor if one was provided."""
        all_chunks = []

        for dirpath, dirnames, filenames in os.walk(self.worlddir):
            if not dirnames and filenames and "DIM-1" not in dirpath:
                for f in filenames:
                    if f.startswith("c.") and f.endswith(".dat"):
                        p = f.split(".")
                        all_chunks.append((base36decode(p[1]), base36decode(p[2]), 
                            os.path.join(dirpath, f)))

        if not all_chunks:
            logging.error("Error: No chunks found!")
            sys.exit(1)
        return all_chunks

    def _render_chunks_async(self, chunks, processes):
        """Starts up a process pool and renders all the chunks asynchronously.

        chunks is a list of (col, row, chunkfile)

        Returns a dictionary mapping (col, row) to the file where that
        chunk is rendered as an image
        """
        # The set of chunks to render, or None for all of them. The logic is
        # slightly more compliated than it should seem, since we still need to
        # build the results dict out of all chunks, even if they're not being
        # rendered.
        inclusion_set = self._get_chunk_renderset()

        results = {}
        if processes == 1:
            # Skip the multiprocessing stuff
            logging.debug("Rendering chunks synchronously since you requested 1 process")
            for i, (col, row, chunkfile) in enumerate(chunks):
                if inclusion_set and (col, row) not in inclusion_set:
                    # Skip rendering, just find where the existing image is
                    _, imgpath = chunk.ChunkRenderer(chunkfile,
                            self.cachedir, self).find_oldimage(False)
                    if imgpath:
                        results[(col, row)] = imgpath
                        continue

                result = chunk.render_and_save(chunkfile, self.cachedir, self, cave=self.caves)
                results[(col, row)] = result
                if i > 0:
                    if 1000 % i == 0 or i % 1000 == 0:
                        logging.info("{0}/{1} chunks rendered".format(i, len(chunks)))
        else:
            logging.debug("Rendering chunks in {0} processes".format(processes))
            pool = multiprocessing.Pool(processes=processes)
            asyncresults = []
            for col, row, chunkfile in chunks:
                if inclusion_set and (col, row) not in inclusion_set:
                    # Skip rendering, just find where the existing image is
                    _, imgpath = chunk.ChunkRenderer(chunkfile,
                            self.cachedir, self).find_oldimage(False)
                    if imgpath:
                        results[(col, row)] = imgpath
                        continue

                result = pool.apply_async(chunk.render_and_save,
                        args=(chunkfile,self.cachedir,self),
                        kwds=dict(cave=self.caves))
                asyncresults.append((col, row, result))

            pool.close()

            for i, (col, row, result) in enumerate(asyncresults):
                results[(col, row)] = result.get()
                if i > 0:
                    if 1000 % i == 0 or i % 1000 == 0:
                        logging.info("{0}/{1} chunks rendered".format(i, len(asyncresults)))

            pool.join()
        logging.info("Done!")

        return results

def get_save_dir():
    """Returns the path to the local saves directory
      * On Windows, at %APPDATA%/.minecraft/saves/
      * On Darwin, at $HOME/Library/Application Support/minecraft/saves/
      * at $HOME/.minecraft/saves/

    """
    
    savepaths = []
    if "APPDATA" in os.environ:
        savepaths += [os.path.join(os.environ['APPDATA'], ".minecraft", "saves")]
    if "HOME" in os.environ:
        savepaths += [os.path.join(os.environ['HOME'], "Library",
                "Application Support", "minecraft", "saves")]
        savepaths += [os.path.join(os.environ['HOME'], ".minecraft", "saves")]

    for path in savepaths:
        if os.path.exists(path):
            return path

def get_worlds():
    "Returns {world # : level.dat information}"
    ret = {}
    save_dir = get_save_dir()

    # No dirs found - most likely not running from inside minecraft-dir
    if save_dir is None:
        return None

    for dir in os.listdir(save_dir):
        if dir.startswith("World") and len(dir) == 6:
            world_n = int(dir[-1])
            info = nbt.load(os.path.join(save_dir, dir, "level.dat"))[1]
            info['Data']['path'] = os.path.join(save_dir, dir)
            ret[world_n] = info['Data']

    return ret
