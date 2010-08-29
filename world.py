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
    correspond to"""
    all_chunks = []
    for dirpath, dirnames, filenames in os.walk(worlddir):
        if not dirnames and filenames:
            for f in filenames:
                if f.startswith("c.") and f.endswith(".dat"):
                    p = f.split(".")
                    all_chunks.append((base36decode(p[1]), base36decode(p[2]), 
                        os.path.join(dirpath, f)))
    return all_chunks

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
    
    # Find the max and min sum and difference. Start out by finding the sum and
    # diff of the first chunk
    item = all_chunks[0]
    minsum = maxsum = item[0] + item[1]
    mindiff = maxdiff = item[1] - item[0]

    for c in all_chunks:
        s = c[0] + c[1]
        minsum = min(minsum, s)
        maxsum = max(maxsum, s)
        d = c[1] - c[0]
        mindiff = min(mindiff, d)
        maxdiff = max(maxdiff, d)

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
    pool = multiprocessing.Pool(processes=procs)
    resultsmap = {}
    for chunkx, chunky, chunkfile in all_chunks:
        result = pool.apply_async(chunk.render_and_save, args=(chunkfile,),
                kwds=dict(cave=cavemode))
        resultsmap[(chunkx, chunky)] = result

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

