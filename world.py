import functools
import string
import os
import os.path
import time
import multiprocessing

from PIL import Image

import chunk

base36decode = functools.partial(int, base=36)

def base36encode(number):
    """String repr of a number in base 32"""
    if number==0: return '0'
    alphabet = string.digits + string.lowercase

    if number < 0:
        number = -number
        neg = True
    else:
        neg = False
    base36 = ''
    while number != 0:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36

    if neg:
        return "-"+base36
    else:
        return base36

def find_chunkfiles(worlddir):
    """Returns a list of all the chunk file locations, and the file they
    correspond to.
    
    Returns a list of (chunkx, chunky, filename) where chunkx and chunky are
    given in chunk coordinates. Use convert_coords() to turn the resulting list
    into an oblique coordinate system"""
    all_chunks = []
    for dirpath, dirnames, filenames in os.walk(worlddir):
        if not dirnames and filenames:
            for f in filenames:
                if f.startswith("c.") and f.endswith(".dat"):
                    p = f.split(".")
                    all_chunks.append((base36decode(p[1]), base36decode(p[2]), 
                        os.path.join(dirpath, f)))
    return all_chunks

def render_chunks_async(chunks, caves, processes):
    """Starts up a process pool and renders all the chunks asynchronously.

    caves is boolean passed to chunk.render_and_save()

    chunks is a list of (chunkx, chunky, chunkfile)

    Returns a dictionary mapping (chunkx, chunky) to a
    multiprocessing.pool.AsyncResult object
    """
    pool = multiprocessing.Pool(processes=processes)
    resultsmap = {}
    for chunkx, chunky, chunkfile in chunks:
        result = pool.apply_async(chunk.render_and_save, args=(chunkfile,),
                kwds=dict(cave=caves))
        resultsmap[(chunkx, chunky)] = result

    # Stick the pool object in the dict under the key "pool" so it isn't
    # garbage collected (which kills the subprocesses)
    resultsmap['pool'] = pool

    return resultsmap

def render_world(worlddir, cavemode=False, procs=2):
    print "Scanning chunks..."
    all_chunks = find_chunkfiles(worlddir)

    total = len(all_chunks)
    print "Done! {0} chunks found".format(total)
    if not total:
        return

    # Create an image big enough for all chunks
    # Each chunk is 384 pixels across. Each chunk is vertically 1728 pixels,
    # but are spaced only 16*12=192 pixels apart. (Staggered, it's half that)

    # Imagine a diagonal coordinate system to address the chunks where
    # increasing x goes up-right and increasing z goes down-right. This needs
    # to be embedded in a square. How big is this square?

    # Each column of chunks has a constant x+z sum of their coordinates, since
    # going from a chunk to the one below it involves adding 1 to z and
    # subtracting 1 from x.  Therefore, the leftmost column is the one that
    # minimizes x+z. The rightmost column maximizes x+z

    # This means the total width of the image is max sum - the min sum, times
    # the horizontal spacing between each neighboring chunk. Since the rows are
    # staggered, each row takes up half its actual width: 384/2

    # Similarly, each row of chunks has a constant difference between their x
    # and z coordinate, since going from from a chunk to the one to its right
    # involves an addition of 1 to both x and z.

    # So the total height of the image must be the max diff - the min diff,
    # times the vertical chunk spacing which is half of 16*12. Additionally,
    # 1536-8*12 must be added to the height for the rest of the bottom layer of
    # chunks.

    # Furthermore, the chunks with the minimum z-x are placed on the image at
    # y=0 (in image coordinates, not chunk coordinates). The chunks with the
    # minimum x+z are placed on the image at x=0.

    # I think I may have forgotten to account for the block heights, the image
    # may be short by 12 pixels or so. Not a huge deal.
    
    minsum, maxsum, mindiff, maxdiff, _ = convert_coords(all_chunks)

    width = (maxsum - minsum) * 384//2
    height = (maxdiff-mindiff) * 8*12 + (12*128-8*12)

    print "Final image will be {0}x{1}. (That's {2} bytes!)".format(
            width, height, width*height*4)
    print "Don't worry though, that's just the memory requirements"
    print "The final png will be much smaller"

    # Sort the chunks by their row, so when we loop through them it goes top to
    # bottom
    print "Sorting chunks..."
    all_chunks.sort(key=lambda x: x[1]-x[0])

    print "Starting up {0} chunk processors...".format(procs)
    resultsmap = render_chunks_async(all_chunks, cavemode, procs)

    # Oh god create a giant ass image
    print "Allocating memory for the giant image"
    worldimg = Image.new("RGBA", (width, height))

    print "Processing chunks!"
    processed = 0
    starttime = time.time()
    for chunkx, chunky, chunkfile in all_chunks:
        # Read in and render the chunk at world coordinates chunkx,chunky
        # Where should this chunk go on the image?
        column = chunkx + chunky - minsum
        row = chunky - chunkx - mindiff
        # col0 is at x=0. row0 is at y=0.
        # Each col adds 384/2. Each row adds 16*12/2
        imgx = 192 * column
        imgy = 96 * row

        print "Drawing chunk {0},{1} at pos {2},{3}".format(
                chunkx, chunky,
                imgx, imgy)
        print "It's in column {0} row {1}".format(column, row)
        
        # Read it and render
        result = resultsmap[(chunkx, chunky)]
        chunkimagefile = result.get()
        chunkimg = Image.open(chunkimagefile)
        # Draw the image sans alpha layer, using the alpha layer as a mask. (We
        # don't want the alpha layer actually drawn on the image, this pastes
        # it as if it was a layer)
        worldimg.paste(chunkimg.convert("RGB"), (imgx, imgy), chunkimg)

        processed += 1
        
        print "{0}/{1} chunks rendered. Avg {2}s per chunk".format(processed, total,
                (time.time()-starttime)/processed)

    print "All done!"
    print "Took {0} minutes".format((time.time()-starttime)/60)
    return worldimg

def convert_coords(chunks):
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

def render_worldtile(chunkmap, colstart, colend, rowstart, rowend):
    """Renders just the specified chunks into a tile. Unlike usual python
    conventions, rowend and colend are inclusive. Additionally, the chunks
    around the edges are half-way cut off (so that neighboring tiles will
    render the other half)

    chunkmap is a dictionary mapping (col, row) to an object whose .get()
    method returns a chunk filename path (a multiprocessing.pool.AsyncResult
    object) as returned from render_chunks_async()

    The image object is returned.
    """
    # width of one chunk is 384. Each column is half a chunk wide.
    width = 192 * (colend - colstart)
    # Same deal with height
    height = 96 * (rowend - rowstart)
    # I know those equations could be simplified. Left like that for clarity

    tileimg = Image.new("RGBA", (width, height))

    # col colstart will get drawn on the image starting at x coordinates -(384/2)
    # row rowstart will get drawn on the image starting at y coordinates -(192/2)
    # Due to how the tiles fit together, we may need to render chunks way above
    # this (since very few chunks actually touch the top of the sky, some tiles
    # way above this one are possibly visible in this tile). Render them
    # anyways just in case)
    for row in xrange(rowstart-16, rowend+1):
        for col in xrange(colstart, colend+1):
            chunkresult = chunkmap.get((col, row), None)
            if not chunkresult:
                continue
            chunkfile = chunkresult.get()
            chunkimg = Image.open(chunkfile)

            xpos = -192 + (col-colstart)*192
            ypos = -96 + (row-rowstart)*96

            print "Pasting chunk {0},{1} at {2},{3}".format(
                    col, row, xpos, ypos)

            tileimg.paste(chunkimg.convert("RGB"), (xpos, ypos), chunkimg)

    return tileimg
