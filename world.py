import functools
import string
import os
import os.path
import time
import multiprocessing
import hashlib

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

    pool.close()

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

def render_worldtile(chunkmap, colstart, colend, rowstart, rowend, oldhash):
    """Renders just the specified chunks into a tile. Unlike usual python
    conventions, rowend and colend are inclusive. Additionally, the chunks
    around the edges are half-way cut off (so that neighboring tiles will
    render the other half)

    chunkmap is a dictionary mapping (col, row) to an object whose .get()
    method returns a chunk filename path (a multiprocessing.pool.AsyncResult
    object) as returned from render_chunks_async()

    Return value is (image object, hash) where hash is some string that depends
    on the image contents.
    
    If no tiles were found, (None, hash) is returned.

    oldhash is a hash value of an existing tile. The hash of this tile is
    computed before it is rendered, and if they match, rendering is skipped and
    (True, oldhash) is returned.
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
    # anyways just in case). That's the reason for the "rowstart-16" below.

    # Before we render any tiles, check the hash of each image in this tile to
    # see if it's changed.
    tilelist = []
    imghash = hashlib.md5()
    for row in xrange(rowstart-16, rowend+1):
        for col in xrange(colstart, colend+1):
            chunkresult = chunkmap.get((col, row), None)
            if not chunkresult:
                continue
            chunkfile = chunkresult.get()
            tilelist.append((col, row, chunkfile))
            # Get the hash of this image and add it to our hash for this tile
            imghash.update(
                    os.path.basename(chunkfile).split(".")[4]
                    )

    digest = imghash.digest()
    if not tilelist:
        # No chunks were found in this tile
        return None, digest
    if digest == oldhash:
        # All the chunks for this tile have not changed according to the hash
        return True, digest

    tileimg = Image.new("RGBA", (width, height))

    # col colstart will get drawn on the image starting at x coordinates -(384/2)
    # row rowstart will get drawn on the image starting at y coordinates -(192/2)
    for col, row, chunkfile in tilelist:
        chunkimg = Image.open(chunkfile)

        xpos = -192 + (col-colstart)*192
        ypos = -96 + (row-rowstart)*96

        #print "Pasting chunk {0},{1} at {2},{3}".format(
        #        col, row, xpos, ypos)

        tileimg.paste(chunkimg.convert("RGB"), (xpos, ypos), chunkimg)

    return tileimg, digest

def get_quadtree_depth(colstart, colend, rowstart, rowend):
    """Determines the zoom depth of a requested quadtree.
    
    Return value is an integer >= 0. Higher integers mean higher resolution
    maps.  This is one less than the maximum zoom (level 0 is a single tile,
    level 1 is 2 tiles wide by 2 tiles high, etc.)

    """
    # This determines how many zoom levels we need to encompass the entire map.
    # We need to make sure that each recursive call splits both dimensions
    # evenly into a power of 2 tiles wide and high, so this function determines
    # how many splits to make, and generate_quadtree() uses this to adjust the
    # row and column limits so that everything splits just right.
    #
    # This comment makes more sense if you consider it inlined in its call from
    # generate_quadtree()
    # Since a single tile has 3 columns of chunks and 5 rows of chunks,  this
    # split needs to be sized into the void so that it is some number of rows
    # in the form 2*2^p. And columns must be in the form 4*2^p
    # They need to be the same power
    # In other words, I need to find the smallest power p such that
    # colmid + 2*2^p >= colend and rowmid + 4*2^p >= rowend
    # I hope that makes some sense. I don't know how to explain this very well,
    # it was some trial and error.
    colmid = (colstart + colend) // 2
    rowmid = (rowstart + rowend) // 2
    for p in xrange(15): # That should be a high enough upper limit
        if colmid + 2*2**p >= colend and rowmid + 4*2**p >= rowend:
            break
    else:
        raise Exception("Your map is waaaay to big")
    
    return p

def generate_quadtree(chunkmap, colstart, colend, rowstart, rowend, prefix, procs):
    """Base call for quadtree_recurse. This sets up the recursion and generates
    a quadtree given a chunkmap and the ranges.

    """
    p = get_quadtree_depth(colstart, colend, rowstart, rowend);
    colmid = (colstart + colend) // 2
    rowmid = (rowstart + rowend) // 2

    # Modify the lower and upper bounds to be sized correctly. See comments in
    # get_quadtree_depth()
    colstart = colmid - 2*2**p
    colend = colmid + 2*2**p
    rowstart = rowmid - 4*2**p
    rowend = rowmid + 4*2**p

    #print "     power is", p
    #print "     new bounds: {0},{1} {2},{3}".format(colstart, colend, rowstart, rowend)

    # procs is -1 here since the main process always runs as well, only spawn
    # procs-1 /new/ processes
    sem = multiprocessing.BoundedSemaphore(procs-1)
    quadtree_recurse(chunkmap, colstart, colend, rowstart, rowend, prefix, "base", sem)

def quadtree_recurse(chunkmap, colstart, colend, rowstart, rowend, prefix, quadrant, sem):
    """Recursive method that generates a quadtree.
    A single call generates, saves, and returns an image with the range
    specified by colstart,colend,rowstart, and rowend.

    The image is saved as os.path.join(prefix, quadrant+".png")

    If the requested range is larger than a certain threshold, this method will
    instead make 4 calls to itself to render the 4 quadrants of the image. The
    four pieces are then resized and pasted into one image that is saved and
    returned.

    If the requested range is not too large, it is generated with
    render_worldtile()

    The path "prefix" should be a directory where this call should save its
    image.

    quadrant is used in recursion. If it is "base", the image is saved in the
    directory named by prefix, and recursive calls will have quadrant set to
    "0" "1" "2" or "3" and prefix will remain unchanged.

    If quadrant is anything else, the tile will be saved just the same, but for
    recursive calls a directory named quadrant will be created (if it doesn't
    exist) and prefix will be set to os.path.join(prefix, quadrant)

    So the first call will have prefix "tiles" (e.g.) and quadrant "base" and
    will save its image as "tiles/base.png"
    The second call will have prefix "tiles" and quadrant "0" and will save its
    image as "tiles/0.png". It will create the directory "tiles/0/"
    The third call will have prefix "tiles/0", quadrant "0" and will save its image as
    "tile/0/0.png"

    Each tile outputted is always 384 by 384 pixels.

    The last parameter, sem, should be a multiprocessing.Semaphore or
    BoundedSemaphore object. Before each recursive call, the semaphore is
    acquired without blocking. If the acquire is successful, the recursive call
    will spawn a new process. If it is not successful, the recursive call is
    run in the same thread. The semaphore is passed to each recursive call, so
    any call could spawn new processes if another one exits at some point.

    The return from this function is (path, hash) where path is the path to the
    file saved, and hash is a byte string that depends on the tile's contents.
    If the tile is blank, path will be None, but hash will still be valid.
    
    """
    #if 1 and prefix == "/tmp/testrender/2/1/0/1" and quadrant == "1":
    #    print "Called with {0},{1} {2},{3}".format(colstart, colend, rowstart, rowend)
    #    print "  prefix:", prefix
    #    print "  quadrant:", quadrant
    #    dbg = True
    #else:
    #    dbg = False
    cols = colend - colstart
    rows = rowend - rowstart

    # Get the tile's existing hash. Maybe it hasn't changed. Whether this
    # function invocation is destined to recurse, or whether we end up calling
    # render_worldtile(), the hash will help us short circuit a lot of pixel
    # copying.
    hashpath = os.path.join(prefix, quadrant+".hash")
    if os.path.exists(hashpath):
        oldhash = open(hashpath, "rb").read()
    else:
        # This method (should) never actually return None for a hash, this is
        # used so it will always compare unequal.
        oldhash = None

    if cols == 2 and rows == 4:
        # base case: just render the image
        img, newhash = render_worldtile(chunkmap, colstart, colend, rowstart, rowend, oldhash)
        # There are a few cases to handle here:
        # 1) img is None: the image doesn't exist (would have been blank, no
        #    chunks exist for that range.
        # 2) img is True: the image hasn't changed according to the hashes. The
        #    image object is not returned by render_worldtile, but we do need to
        #    return the path to it.
        # 3) img is a PIL.Image.Image object, a new tile was computed, we need
        #    to save it and its hash (newhash) to disk.

        if not img:
            # The image returned is blank, there should not be an image here.
            # If one does exist, from a previous world or something, it is not
            # deleted, but None is returned to indicate to our caller this tile
            # is blank.
            remove_tile(prefix, quadrant)
            return None, newhash
        if img is True:
            # No image was returned because the hashes matched. Return the path
            # to the image that already exists and is up to date according to
            # the hash
            path = os.path.join(prefix, quadrant+".png")
            if not os.path.exists(path):
                # Oops, the image doesn't actually exist. User must have
                # deleted it, or must be some bug?
                raise Exception("Error, this image should have existed according to the hashes, but didn't")
            return path, newhash

        # If img was not None or True, it is an image object. The image exists
        # and the hashes did not match, so it must have changed. Fall through
        # to the last part of this function which saves the image and its hash.
        assert isinstance(img, Image.Image)
    elif cols < 2 or rows < 4:
        raise Exception("Something went wrong, this tile is too small. (Please send "
                "me the traceback so I can fix this)")
    else:
        # Recursively generate each quadrant for this tile

        # Find the midpoint
        colmid = (colstart + colend) // 2
        rowmid = (rowstart + rowend) // 2

        # Assert that the split in the center still leaves everything sized
        # exactly right by checking divisibility by the final row and
        # column sizes. This isn't sufficient, but is necessary for
        # success. (A better check would make sure the dimensions fit the
        # above equations for the same power of 2)
        assert (colmid - colstart) % 2 == 0
        assert (colend - colmid) % 2 == 0
        assert (rowmid - rowstart) % 4 == 0
        assert (rowend - rowmid) % 4 == 0

        if quadrant == "base":
            newprefix = prefix
        else:
            # Make the directory for the recursive subcalls
            newprefix = os.path.join(prefix, quadrant)
            if not os.path.exists(newprefix):
                os.mkdir(newprefix)
        
        # Keep a hash of the concatenation of each returned hash. If it matches
        # oldhash from above, skip rendering this tile
        hasher = hashlib.md5()

        # Recurse to generate each quadrant of images
        # Quadrent 1:
        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess

        quad0result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colstart, colmid, rowstart, rowmid, newprefix, "0", sem)
                )
        quad0result.start()

        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess
        quad1result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colmid, colend, rowstart, rowmid, newprefix, "1", sem)
                )
        quad1result.start()

        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess
        quad2result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colstart, colmid, rowmid, rowend, newprefix, "2", sem)
                )
        quad2result.start()

        # 3rd quadrent always runs in this process, no need to spawn a new one
        # since we're just going to turn around and wait for it.
        quad3file, hash3 = quadtree_recurse(chunkmap, 
                colmid, colend, rowmid, rowend,
                newprefix, "3", sem)

        quad0file, hash0 = quad0result.get()
        quad1file, hash1 = quad1result.get()
        quad2file, hash2 = quad2result.get()

        #if dbg:
        #    print quad0file
        #    print repr(hash0)
        #    print quad1file
        #    print repr(hash1)
        #    print quad2file
        #    print repr(hash2)
        #    print quad3file
        #    print repr(hash3)

        # Check the hashes. This is checked even if the tile files returned
        # None, since that could happen if either the tile was blank or it
        # hasn't changed. So the hashes returned should tell us whether we need
        # to update this tile or not.
        hasher.update(hash0)
        hasher.update(hash1)
        hasher.update(hash2)
        hasher.update(hash3)
        newhash = hasher.digest()
        if newhash == oldhash:
            # Nothing left to do, this tile already exists and hasn't changed.
            #if dbg: print "hashes match, nothing to do"
            return os.path.join(prefix, quadrant+".png"), oldhash

        # Check here if this tile is actually blank. If all 4 returned quadrant
        # filenames are None, this tile should not be rendered. However, we
        # still need to return a valid hash for it, so that's why this check is
        # below the hash check.
        if not (bool(quad0file) or bool(quad1file) or bool(quad2file) or
                bool(quad3file)):
            remove_tile(prefix, quadrant)
            return None, newhash

        img = Image.new("RGBA", (384, 384))

        if quad0file:
            quad0 = Image.open(quad0file).resize((192,192), Image.ANTIALIAS)
            img.paste(quad0, (0,0))
        if quad1file:
            quad1 = Image.open(quad1file).resize((192,192), Image.ANTIALIAS)
            img.paste(quad1, (192,0))
        if quad2file:
            quad2 = Image.open(quad2file).resize((192,192), Image.ANTIALIAS)
            img.paste(quad2, (0, 192))
        if quad3file:
            quad3 = Image.open(quad3file).resize((192,192), Image.ANTIALIAS)
            img.paste(quad3, (192, 192))

    # At this point, if the tile hasn't change or is blank, the function should
    # have returned by now.
    assert bool(img)

    # Save the image
    path = os.path.join(prefix, quadrant+".png")
    img.save(path)

    print "Saving image", path

    # Save the hash
    with open(os.path.join(prefix, quadrant+".hash"), 'wb') as hashout:
        hashout.write(newhash)

    # Return the location and hash of this tile
    return path, newhash

def remove_tile(prefix, quadrent):
    """Called when a tile doesn't exist, this deletes an existing tile if it
    does
    """
    path = os.path.join(prefix, quadrent)
    img = path + ".png"
    hash = path + ".hash"

    if os.path.exists(img):
        print "removing", img
        os.unlink(img)
    if os.path.exists(hash):
        os.unlink(hash)

class ReturnableProcess(multiprocessing.Process):
    """Like the standard multiprocessing.Process class, but the return value of
    the target method is available by calling get().
    
    The given semaphore is released when the target finishes running"""
    def __init__(self, semaphore, *args, **kwargs):
        self.__sem = semaphore
        multiprocessing.Process.__init__(self, *args, **kwargs)

    def run(self):
        results = self._target(*self._args, **self._kwargs)
        self._respipe_in.send(results)
        self.__sem.release()

    def get(self):
        self.join()
        return self._respipe_out.recv()

    def start(self):
        self._respipe_out, self._respipe_in = multiprocessing.Pipe()
        multiprocessing.Process.start(self)

class FakeProcess(object):
    """Identical interface to the above class, but runs in the same thread.
    Used to make the code simpler in quadtree_recurse

    """
    def __init__(self, semaphore, target, args=None, kwargs=None):
        self._target = target
        self._args = args if args else ()
        self._kwargs = kwargs if kwargs else {}
    def start(self):
        self.ret = self._target(*self._args, **self._kwargs)
    def get(self):
        return self.ret
