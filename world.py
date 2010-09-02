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

def load_sort_and_process(worlddir):
    """Takes a directory to a world dir, and returns a mapping from (col, row)
    to result object"""
    all_chunks = find_chunkfiles(worlddir)
    mincol, maxcol, minrow, maxrow, translated_chunks = convert_coords(all_chunks)
    results = render_chunks_async(translated_chunks, caves=False, processes=5)
    return results

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

def generate_quadtree(chunkmap, colstart, colend, rowstart, rowend, prefix, quadrent="base"):
    """Recursive method that generates a quadtree.
    A single call generates, saves, and returns an image with the range
    specified by colstart,colend,rowstart, and rowend.

    The image is saved as os.path.join(prefix, quadrent+".png")

    If the requested range is larger than a certain threshold, this method will
    instead make 4 calls to itself to render the 4 quadrents of the image. The
    four pieces are then resized and pasted into one image that is saved and
    returned.

    If the requested range is not too large, it is generated with
    render_worldtile()

    The path "prefix" should be a directory where this call should save its
    image.

    quadrent is used in recursion. If it is "base", the image is saved in the
    directory named by prefix, and recursive calls will have quadrent set to
    "0" "1" "2" or "3" and prefix will remain unchanged.

    If quadrent is anything else, the tile will be saved just the same, but for
    recursive calls a directory named quadrent will be created (if it doesn't
    exist) and prefix will be set to os.path.join(prefix, quadrent)

    So the first call will have prefix "tiles" (e.g.) and quadrent "base" and
    will save its image as "tiles/base.png"
    The second call will have prefix "tiles" and quadrent "0" and will save its
    image as "tiles/0.png". It will create the directory "tiles/0/"
    The third call will have prefix "tiles/0", quadrent "0" and will save its image as
    "tile/0/0.png"

    Each tile outputted is always 384 by 384 pixels.
    """
    print "Called with {0},{1} {2},{3}".format(colstart, colend, rowstart, rowend)
    print "  prefix:", prefix
    print "  quadrent:", quadrent
    cols = colend - colstart
    rows = rowend - rowstart

    if cols == 3 and rows == 5:
        # base case: just render the image
        img = render_worldtile(chunkmap, colstart, colend, rowstart, rowend)
    elif cols < 3 or rows < 5:
        Exception("Something went wrong, this tile is too small. (Please send "
                "me the traceback so I can fix this)")
    else:
        # Recursively generate each quadrent for this tile
        img = Image.new("RGBA", (384, 384))

        # Find the midpoint
        colmid = (colstart + colend) // 2
        rowmid = (rowstart + rowend) // 2
        if quadrent == "base":
            # The first call has a special job. No matter the input, we need to
            # make sure that each recursive call splits both dimensions evenly
            # into a power of 2 * 384. (Since all tiles are 384x384 which is 3
            # cols by 5 rows)
            # Since the row of the final recursion needs to be 3, this split
            # needs to be sized into the void so that it is some number of rows
            # in the form 3*2^p. And columns must be in the form 5*2^p
            # They need to be the same power
            # In other words, I need to find the smallest power p such that
            # colmid + 3*2^p >= colend and rowmid + 5*2^p >= rowend
            for p in xrange(15): # That should be a high enough upper limit
                if colmid + 3*2**p >= colend and rowmid + 5*2**p >= rowend:
                    break
            else:
                raise Exception("Your map is waaaay to big")

            # Modify the lower and upper bounds to be sized correctly
            colstart = colmid - 3*2**p
            colend = colmid + 3*2**p
            rowstart = rowmid - 5*2**p
            rowend = rowmid + 5*2**p

            print "     power is", p
            print "     new bounds: {0},{1} {2},{3}".format(colstart, colend, rowstart, rowend)

            newprefix = prefix
        else:
            # Assert that the split in the center still leaves everything sized
            # exactly right by checking divisibility by the final row and
            # column sizes. This isn't sufficient, but is necessary for
            # success. (A better check would make sure the dimensions fit the
            # above equations for the same power of 2)
            assert (colmid - colstart) % 3 == 0
            assert (colend - colmid) % 3 == 0
            assert (rowmid - rowstart) % 5 == 0
            assert (rowend - rowmid) % 5 == 0

            newprefix = os.path.join(prefix, quadrent)
            if not os.path.exists(newprefix):
                os.mkdir(newprefix)

        # Recurse to generate each quadrent of images
        quad0file = generate_quadtree(chunkmap, 
                colstart, colmid, rowstart, rowmid,
                newprefix, "0")
        quad1file = generate_quadtree(chunkmap, 
                colmid, colend, rowstart, rowmid,
                newprefix, "1")
        quad2file = generate_quadtree(chunkmap, 
                colstart, colmid, rowmid, rowend,
                newprefix, "2")
        quad3file = generate_quadtree(chunkmap, 
                colmid, colend, rowmid, rowend,
                newprefix, "3")

        quad0 = Image.open(quad0file).resize((192,192))
        quad1 = Image.open(quad1file).resize((192,192))
        quad2 = Image.open(quad2file).resize((192,192))
        quad3 = Image.open(quad3file).resize((192,192))

        img.paste(quad0, (0,0))
        img.paste(quad1, (192,0))
        img.paste(quad2, (0, 192))
        img.paste(quad3, (192, 192))

    # Save the image
    path = os.path.join(prefix, quadrent+".png")
    img.save(path)

    # Return its location
    return path
