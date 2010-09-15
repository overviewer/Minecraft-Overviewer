import multiprocessing
import itertools
import os
import os.path
import hashlib
import functools

from PIL import Image

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
            print "Ctrl-C caught!"
            raise Exception("Exiting")
        except:
            import traceback
            traceback.print_exc()
            raise
    return newfunc

class QuadtreeGen(object):
    def __init__(self, worldobj, destdir):
        """Generates a quadtree from the world given into the
        given dest directory

        worldobj is a world.WorldRenderer object that has already been processed

        """
        # Determine quadtree depth (midpoint is always 0,0)
        for p in xrange(15):
            xdiameter = 2*2**p
            ydiameter = 4*2**p
            if xdiameter >= worldobj.maxcol and -xdiameter <= worldobj.mincol and \
                    ydiameter >= worldobj.maxrow and -ydiameter <= worldobj.minrow:
                break
        else:
            raise ValueError("Your map is waaaay to big!")

        self.p = p

        # Make new row and column ranges
        self.mincol = -xdiameter
        self.maxcol = xdiameter
        self.minrow = -ydiameter
        self.maxrow = ydiameter

        self.world = worldobj
        self.destdir = destdir

    def go(self, procs):
        """Renders all tiles"""

        # Create a pool
        pool = multiprocessing.Pool(processes=procs)

        # Render the highest level of tiles from the chunks
        print "Computing the tile ranges and starting tile processers for inner-most tiles..."
        results = []
        for path in iterate_base4(self.p+1):
            # Get the range for this tile
            colstart, rowstart = self._get_range_by_path(path)
            colend = colstart + 2
            rowend = rowstart + 4

            # This image is rendered at:
            dest = os.path.join(self.destdir, *(str(x) for x in path))

            # The directory, create it if not exists
            dirdest = os.path.dirname(dest)
            if not os.path.exists(dirdest):
                os.makedirs(dirdest)

            # And uses these chunks
            tilechunks = self._get_chunks_in_range(colstart, colend, rowstart,
                    rowend)

            # Put this in the pool
            # (even if tilechunks is empty, render_worldtile will delete
            # existing images if appropriate)
            results.append(
                    pool.apply_async(func=render_worldtile, args=
                        (tilechunks, colstart, colend, rowstart, rowend, dest)
                        )
                    )

        # Wait for all results to finish
        print "Rendering inner most zoom level tiles now!"
        for i, result in enumerate(results):
            # get() instead of wait() so we can see errors
            result.get()
            if i > 0 and (i % 100 == 0 or 100 % i == 0):
                print "{0}/{1} tiles complete on level {2}/{3}".format(
                        i, len(results), 1, self.p+1)

        # Now do the other layers
        for zoom in xrange(self.p, 0, -1):
            level = self.p+2-zoom
            print "Preparing level", level

            results = []
            for path in iterate_base4(zoom):
                # This image is rendered at:
                dest = os.path.join(self.destdir, *(str(x) for x in path[:-1]))
                name = str(path[-1])

                print "Applying", path, dest, name
                results.append(
                        pool.apply_async(func=render_innertile, args=
                            (dest, name)
                            )
                        )

            print "Rendering level {0}/{1} now!".format(level, self.p+1)
            for i, result in enumerate(results):
                # get() instead of wait() so we can see errors
                result.get()
                if i > 0 and (i % 100 == 0 or 100 % i == 0):
                    print "{0}/{1} tiles complete on level {2}/{3}".format(
                            i, len(results), level, self.p+1)

        # Do the final one right here:
        render_innertile(self.destdir, "base")
        print "Done!"

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
def render_innertile(dest, name):
    """
    Renders a tile at os.path.join(dest, name)+".png" by taking tiles from
    os.path.join(dest, name, "{0,1,2,3}.png")
    """
    imgpath = os.path.join(dest, name) + ".png"
    hashpath = os.path.join(dest, name) + ".hash"

    if name == "base":
        q0path = os.path.join(dest, "0.png")
        q1path = os.path.join(dest, "1.png")
        q2path = os.path.join(dest, "2.png")
        q3path = os.path.join(dest, "3.png")
        q0hash = os.path.join(dest, "0.hash")
        q1hash = os.path.join(dest, "1.hash")
        q2hash = os.path.join(dest, "2.hash")
        q3hash = os.path.join(dest, "3.hash")
    else:
        q0path = os.path.join(dest, name, "0.png")
        q1path = os.path.join(dest, name, "1.png")
        q2path = os.path.join(dest, name, "2.png")
        q3path = os.path.join(dest, name, "3.png")
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
        print "Not generating due to non-existance of subtiles"
        print "\t", dest, name
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
        print "Not generating due to hash match"
        print "\t", dest, name
        return

    # Create the actual image now
    img = Image.new("RGBA", (384, 384))

    if q0path:
        quad0 = Image.open(q0path).resize((192,192), Image.ANTIALIAS)
        img.paste(quad0, (0,0))
    if q1path:
        quad1 = Image.open(q1path).resize((192,192), Image.ANTIALIAS)
        img.paste(quad1, (192,0))
    if q2path:
        quad2 = Image.open(q2path).resize((192,192), Image.ANTIALIAS)
        img.paste(quad2, (0, 192))
    if q3path:
        quad3 = Image.open(q3path).resize((192,192), Image.ANTIALIAS)
        img.paste(quad3, (192, 192))

    # Save it
    print "Saving", imgpath
    img.save(imgpath)
    with open(hashpath, "wb") as hashout:
        hashout.write(newhash)


@catch_keyboardinterrupt
def render_worldtile(chunks, colstart, colend, rowstart, rowend, path):
    """Renders just the specified chunks into a tile and save it. Unlike usual
    python conventions, rowend and colend are inclusive. Additionally, the
    chunks around the edges are half-way cut off (so that neighboring tiles
    will render the other half)

    chunks is a list of (col, row, filename) of chunk images that are relevant
    to this call

    The image is saved to path+".png" and a hash is saved to path+".hash"

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
    imgpath = path + ".png"

    if not chunks:
        # No chunks were found in this tile
        if os.path.exists(imgpath):
            os.unlink(imgpath)
        if os.path.exists(hashpath):
            os.unlink(hashpath)
        return None

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
    tileimg = Image.new("RGBA", (width, height))

    # col colstart will get drawn on the image starting at x coordinates -(384/2)
    # row rowstart will get drawn on the image starting at y coordinates -(192/2)
    for col, row, chunkfile in chunks:
        try:
            chunkimg = Image.open(chunkfile)
        except IOError, e:
            print "Error opening file", chunkfile
            print "Attempting to re-generate it"
            os.unlink(chunkfile)
            # Do some string manipulation to determine what the chunk file is
            # that goes with this image. Then call chunk.render_and_save
            dirname, imagename = os.path.split(chunkfile)
            parts = imagename.split(".")
            datafile = "c.{0}.{1}.dat".format(parts[1],parts[2])
            print "Chunk came from data file", datafile
            # XXX Don't forget to set cave mode here when it gets implemented!
            chunk.render_and_save(os.path.join(dirname, datafile), False)
            chunkimg = Image.open(chunkfile)
            print "Success"

        xpos = -192 + (col-colstart)*192
        ypos = -96 + (row-rowstart)*96

        tileimg.paste(chunkimg.convert("RGB"), (xpos, ypos), chunkimg)

    # Save them
    tileimg.save(imgpath)
    with open(hashpath, "wb") as hashout:
        hashout.write(digest)

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
        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess

        quad0result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colstart, colmid, rowstart, rowmid, newprefix, "0", sem)
                )

        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess
        quad1result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colmid, colend, rowstart, rowmid, newprefix, "1", sem)
                )

        if sem.acquire(False):
            Procobj = ReturnableProcess
        else:
            Procobj = FakeProcess
        quad2result = Procobj(sem, target=quadtree_recurse,
                args=(chunkmap, colstart, colmid, rowmid, rowend, newprefix, "2", sem)
                )
        
        # Start the processes. If one is a fakeprocess, it will do the
        # processing right here instead.
        quad0result.start()
        quad1result.start()
        quad2result.start()

        # 3rd quadrent always runs in this process, no need to spawn a new one
        # since we're just going to turn around and wait for it.
        quad3file, hash3 = quadtree_recurse(chunkmap, 
                colmid, colend, rowmid, rowend,
                newprefix, "3", sem)

        quad0file, hash0 = quad0result.get()
        quad1file, hash1 = quad1result.get()
        quad2file, hash2 = quad2result.get()

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
        try:
            results = self._target(*self._args, **self._kwargs)
        except BaseException, e:
            self._respipe_in.send(e)
        else:
            self._respipe_in.send(results)
        finally:
            self.__sem.release()

    def get(self):
        self.join()
        ret = self._respipe_out.recv()
        if isinstance(ret, BaseException):
            raise ret
        return ret

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
