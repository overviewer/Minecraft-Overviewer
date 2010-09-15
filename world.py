import functools
import string
import os
import os.path
import time
import multiprocessing
import hashlib

from PIL import Image

import chunk

"""
This module has routines related to generating all the chunks for a world

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

class WorldRenderer(object):
    """Renders a world's worth of chunks"""
    def __init__(self, worlddir):
        self.worlddir = worlddir
        self.caves = False

    def go(self, procs):
        """Starts the render. This returns when it is finished"""
        
        print "Scanning chunks"
        raw_chunks = self._find_chunkfiles()

        # Translate chunks to our diagonal coordinate system
        mincol, maxcol, minrow, maxrow, chunks = _convert_coords(raw_chunks)

        self.chunkmap = self._render_chunks_async(chunks, procs)

        self.mincol = mincol
        self.maxcol = maxcol
        self.minrow = minrow
        self.maxrow = maxrow

    def _find_chunkfiles(self):
        """Returns a list of all the chunk file locations, and the file they
        correspond to.
        
        Returns a list of (chunkx, chunky, filename) where chunkx and chunky are
        given in chunk coordinates. Use convert_coords() to turn the resulting list
        into an oblique coordinate system"""
        all_chunks = []
        for dirpath, dirnames, filenames in os.walk(self.worlddir):
            if not dirnames and filenames:
                for f in filenames:
                    if f.startswith("c.") and f.endswith(".dat"):
                        p = f.split(".")
                        all_chunks.append((base36decode(p[1]), base36decode(p[2]), 
                            os.path.join(dirpath, f)))
        return all_chunks

    def _render_chunks_async(self, chunks, processes):
        """Starts up a process pool and renders all the chunks asynchronously.

        chunks is a list of (col, row, chunkfile)

        Returns a dictionary mapping (col, row) to the file where that
        chunk is rendered as an image
        """
        results = {}
        if processes == 1:
            # Skip the multiprocessing stuff
            print "Rendering chunks synchronously since you requested 1 process"
            for i, (col, row, chunkfile) in enumerate(chunks):
                result = chunk.render_and_save(chunkfile, cave=self.caves)
                results[(col, row)] = result
                if i > 0:
                    if 1000 % i == 0 or i % 1000 == 0:
                        print "{0}/{1} chunks rendered".format(i, len(chunks))
        else:
            print "Rendering chunks in {0} processes".format(processes)
            pool = multiprocessing.Pool(processes=processes)
            asyncresults = []
            for col, row, chunkfile in chunks:
                result = pool.apply_async(chunk.render_and_save, args=(chunkfile,),
                        kwds=dict(cave=self.caves))
                asyncresults.append((col, row, result))

            pool.close()

            for i, (col, row, result) in enumerate(asyncresults):
                results[(col, row)] = result.get()
                if i > 0:
                    if 1000 % i == 0 or i % 1000 == 0:
                        print "{0}/{1} chunks rendered".format(i, len(chunks))

            pool.join()
        print "Done!"

        return results

