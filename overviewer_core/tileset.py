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

import itertools
import logging
import os
import os.path
import shutil
import random
from collections import namedtuple

from .util import iterate_base4, convert_coords

"""

tileset.py contains the TileSet class, and in general, routines that manage a
set of output tiles corresponding to a requested rendermode for a world. In
general, there will be one TileSet object per world per rendermode requested by
the user.

The TileSet class implements the Worker interface. This interface has the
following methods:

do_preprocessing()
    This method is called before iterate_work_items(). It should do any work
    that needs to be done prior to iterate_work_items(). It is not called for
    instances that will not have iterate_work_items() called.

get_num_phases()
    This method returns an integer indicating how many phases of work this
    worker has to perform. Each phase of work is completed serially with the
    other phases... all work done by one phase is done before the next phase is
    started.

get_phase_length(phase)
    This method returns an integer indicating how many work items there are in
    this phase. This number is used for purely informational purposes. It can
    be exact, or an estimate. If there is no useful information on the size of
    a phase, return None.

iterate_work_items(phase)
    Takes a phase number (a non-negative integer). This method should return an
    iterator over work items and a list of dependencies i.e. (work_item, [d1,
    d2, ...]). The work items and dependencies can be any pickelable object;
    they are treated as opaque by the Dispatcher. The work item objects are
    passed back in to the do_work() method (perhaps in a different, identically
    configured instance).

    The dependency items are other work items that are compared for equality
    with work items that are already in the queue. The dispatcher guarantees
    that dependent items which are currently in the queue or in progress finish
    before the corresponding work item is started. Note that dependencies must
    have already been yielded as work items before they can be used as
    dependencies; the dispatcher requires this ordering or it cannot guarantee
    the dependencies are met.

do_work(workobj)
    Does the work for a given work object. This method is not expected to
    return anything, so the results of its work should be reflected on the
    filesystem or by sending signals.


"""

# A named tuple class storing the row and column bounds for the to-be-rendered
# world
Bounds = namedtuple("Bounds", ("mincol", "maxcol", "minrow", "maxrow"))

__all__ = ["TileSet"]
class TileSet(object):
    """The TileSet object manages the work required to produce a set of tiles
    on disk. It calculates the work that needs to be done and tells the
    dipatcher (through the Worker interface) this information. The Dispatcher
    then tells this object when and where to do the work of rendering the tiles.

    """

    def __init__(self, regionsetobj, assetmanagerobj, options, outputdir):
        """Construct a new TileSet object with the given configuration options
        dictionary.

        options is a dictionary of configuration parameters (strings mapping to
        values) that are interpreted by the rendering engine.

        regionsetobj is the RegionSet object that is used to render the tiles.

        assetmanagerobj is the AssetManager object that represents the
        destination directory where we'll put our tiles.

        outputdir is the absolute path to the tile output directory where the
        tiles are saved. It is assumed to exist already.
        TODO: This should probably be relative to the asset manager's output
        directory to avoid redundancy.

        
        Current valid options for the options dictionary are shown below. All
        the options must be specified unless they are not relevant. If the
        given options do not conform to the specifications, behavior is
        undefined (this class does not do any error checking and assumes items
        are given in the correct form).

        bgcolor
            A hex string specifying the background color for jpeg output.
            e.g.: "#1A1A1A". Not relevant unless rendering jpeg.

        renderchecks
            An integer indicating how to determine which tiles need updating
            and which don't. This is one of three levels:

            0
                Only render tiles that have chunks with a greater mtime than
                the last render timestamp, and their ancestors.
                
                In other words, only renders parts of the map that have changed
                since last render, nothing more, nothing less.
                
                This is the fastest option, but will not detect tiles that have
                e.g. been deleted from the directory tree, or pick up where a
                partial interrupted render left off.

            1
                For render-tiles, render all whose chunks have an mtime greater
                than the mtime of the tile on disk, and their upper-tile
                ancestors.
                
                Also check all other upper-tiles and render any that have
                children with more rencent mtimes than itself.
                
                This is slower due to stat calls to determine tile mtimes, but
                safe if the last render was interrupted.

            2
                Render all tiles unconditionally. This is a "forcerender" and
                is the slowest, but SHOULD be specified if this is the first
                render because the scan will forgo tile stat calls. It's also
                useful for changing texture packs or other options that effect
                the output.

        imgformat
            A string indicating the output format. Must be one of 'png' or
            'jpeg'

        imgquality
            An integer 1-100 indicating the quality of the jpeg output. Only
            relevant in jpeg mode.

        optimizeimg
            an integer indiating optimizations to perform on png outputs. 0
            indicates no optimizations. Only relevant in png mode.
            1 indicates pngcrush is run on all output images
            2 indicates pngcrush and advdef are run on all output images with advdef -z2
            3 indicates pngcrush and advdef are run on all output images with advdef -z4

        rendermode
            Perhaps the most important/relevant option: a string indicating the
            render mode to render. This rendermode must have already been
            registered with the C extension module. 

        rerender_prob
            A floating point number between 0 and 1 indicating the probability
            that a tile which is not marked for render by any mtime checks will
            be rendered anyways. 0 disables this option.

        """
        self.options = options
        self.regionset = regionsetobj
        self.am = assetmanagerobj

        # Throughout the class, self.outputdir is an absolute path to the
        # directory where we output tiles. It is assumed to exist.
        self.outputdir = os.path.abspath(outputdir)

        # Set the image format according to the options
        if self.options['imgformat'] == 'png':
            self.imgextension = 'png'
        elif self.options['imgformat'] == 'jpeg':
            self.imgextension = 'jpg'

    def do_preprocessing(self):
        """For the preprocessing step of the Worker interface, this does the
        chunk scan and stores the resulting tree as a private instance
        attribute for later use in iterate_work_items()

        """
        # REMEMBER THAT ATTRIBUTES ASSIGNED IN THIS METHOD ARE NOT AVAILABLE IN
        # THE do_work() METHOD

        # Calculate the min and max column over all the chunks.
        # This sets self.bounds to a Bounds namedtuple
        self.bounds = self._find_chunk_range()

        # Calculate the depth of the tree
        for p in xrange(1,33): # max 32
            # Will 2^p tiles wide and high suffice?

            # X has twice as many chunks as tiles, then halved since this is a
            # radius
            xradius = 2**p
            # Y has 4 times as many chunks as tiles, then halved since this is
            # a radius
            yradius = 2*2**p
            if xradius >= bounds.maxcol and -xradius <= bounds.mincol and \
                    yradius >= bounds.maxrow and -yradius <= bounds.minrow:
                break

        if p >= 15:
            logging.warning("Just letting you know, your map requries %s zoom levels. This is REALLY big!",
                    p)
        self.treedepth = p

        # Do any tile re-arranging if necessary
        self._rearrange_tiles()

        # Do the chunk scan here
        self.dirtytree = self._chunk_scan()


    def get_num_phases(self):
        """Returns the number of levels in the quadtree, which is equal to the
        number of phases of work that need to be done.

        """
        return 1
    
    def get_phase_length(self, phase):
        """Returns the number of work items in a given phase, or None if there
        is no good estimate.
        """
        return None

    def iterate_work_items(self, phase):
        """Iterates over the dirty tiles in the tree and return them in the
        appropriate order with the appropriate dependencies.

        This method returns an iterator over (obj, [dependencies, ...])
        """

        # With renderchecks set to 0 or 2, simply iterate over the dirty tiles
        # tree in post-traversal order and yield each tile path.
        
        # For 0 our caller has explicitly requested not to check mtimes on
        # disk, to speed things up. So the mode 0 chunk scan only looks at
        # chunk mtimes and the last render mtime, and has marked only the
        # render-tiles that need rendering. Mode 0 then iterates over all dirty
        # render-tiles and upper-tiles that depend on them. It does not check
        # mtimes of upper-tiles, so this is only a good option if the last
        # render was not interrupted.
        
        # For mode 2, this is a forcerender, the caller has requested we render
        # everything. The mode 2 chunk scan marks every tile as needing
        # rendering, and disregards mtimes completely. Mode 2 then iterates
        # over all render-tiles and upper-tiles that depend on them, which is
        # every tile that should exist.
        
        # In both 0 and 2 the render iteration is the same: the dirtytile tree
        # built is authoritive on every tile that needs rendering.

        # In mode 1, things are most complicated. The mode 2 chunk scan checks
        # every render tile's mtime for each chunk that touches it, so it can
        # determine accurately which tiles need rendering regardless of the
        # state on disk. The chunk scan also builds a RendertileSet of *every*
        # render-tile that exists.

        # The mode 1 render iteration then manually iterates over the set of
        # all render-tiles in a post-traversal order. When it visits a
        # render-node, it does the following:
        # * Checks the set of dirty render-tiles to see if the node needs
        #   rendering, and if so, renders it
        # * If the tile was rendered, set the mtime using os.utime() to the max
        #   of the chunk mtimes.
        # * If the tile was rendered, return (True, mtime).
        # * If the tile was not rendered, return (False, mtime)
        #
        # Then, for upper-tiles, it does the following:
        # * Gathers the return values of each child call.
        # * If any child returned True, render this tile.
        # * Otherwise, check this tile's mtime. If any child's mtime is greater
        #   than this tile's mtime, render this tile.
        # * If the tile was rendered, set the mtime using os.utime() to the max
        #   of the child mtimes.
        # * If the tile was rendered, return (True, mtime).
        # * If the tile was not rendered, return (False, mtime)

        pass

    def do_work(self, tileobj):
        """Renders the given tile.

        """

    def get_persistent_data(self):
        """Returns a dictionary representing the persistent data of this
        TileSet. Typically this is called by AssetManager

        """
        pass

    def _find_chunk_range(self):
        """Finds the chunk range in rows/columns and stores them in
        self.minrow, self.maxrow, self.mincol, self.maxcol

        """
        minrow = mincol = maxrow = maxcol = 0

        for c_x, c_z, _ in self.regionset.iterate_chunks():
            # Convert these coordinates to row/col
            col, row = convert_coords(c_x, c_z)

            minrow = min(minrow, row)
            maxrow = max(maxrow, row)
            mincol = min(mincol, col)
            maxcol = max(maxcol, col)

        return Bounds(mincol, maxcol, minrow, maxrow)

    def _rearrange_tiles(self):
        """If the target size of the tree is not the same as the existing size
        on disk, do some re-arranging

        """
        try:
            curdepth = get_dirdepth(self.outputdir)
        except Exception:
            logging.critical("Could not determine existing tile tree depth. Does it exist?")
            raise
        
        if self.treedepth != cur_depth:
            if self.treedepth > curdepth:
                logging.warning("Your map seems to have expanded beyond its previous bounds.")
                logging.warning( "Doing some tile re-arrangements... just a sec...")
                for _ in xrange(self.p-curdepth):
                    self._increase_depth()
            elif self.p < curdepth:
                logging.warning("Your map seems to have shrunk. Did you delete some chunks? No problem. Re-arranging tiles, just a sec...")
                for _ in xrange(curdepth - self.p):
                    self._decrease_depth()

    def _increase_depth(self):
        """Moves existing tiles into place for a larger tree"""
        getpath = functools.partial(os.path.join, self.outputdir)

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

            files = [str(dirnum)+"."+self.imgextension, str(dirnum)]
            newfiles = [str(newnum)+"."+self.imgextension, str(newnum)]

            os.mkdir(newdirpath)
            for f, newf in zip(files, newfiles):
                p = getpath(f)
                if os.path.exists(p):
                    os.rename(p, getpath(newdir, newf))
            os.rename(newdirpath, getpath(str(dirnum)))

    def _decrease_depth(self):
        """If the map size decreases, or perhaps the user has a depth override
        in effect, re-arrange existing tiles for a smaller tree"""
        getpath = functools.partial(os.path.join, self.outputdir)

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
        files = [str(num)+"."+self.imgextension for num in xrange(4)] + ["base." + self.imgextension]
        for f in files:
            try:
                os.unlink(getpath(f))
            except OSError, e:
                pass # doesn't exist maybe?

    def _chunk_scan(self):
        """Scans the chunks of this TileSet's world to determine which
        render-tiles need rendering. Returns a RendertileSet object.

        For rendercheck mode 0: only compares chunk mtimes against last render
        time of the map

        For rendercheck mode 1: compares chunk mtimes against the tile mtimes
        on disk

        For rendercheck mode 2: marks every tile, does not check any mtimes.

        """
        depth = self.treedepth

        dirty = RendertileSet(depth)

        chunkcount = 0
        stime = time.time()

        rendercheck = self.options['rendercheck']
        rerender_prob = self.options['rerender_prob']

        # XXX TODO:
        last_rendertime = 0 # TODO

        if rendercheck == 0:
            def compare_times(chunkmtime, tileobj):
                # Compare chunk mtime to last render time
                return chunkmtime > last_rendertime
        elif rendercheck == 1:
            def compare_times(chunkmtime, tileobj):
                # Compare chunk mtime to tile mtime on disk
                tile_path = tile.get_filepath(self.full_tiledir, self.imgformat)
                try:
                    tile_mtime = os.stat(tile_path)[stat.ST_MTIME]
                except OSError, e:
                    if e.errno != errno.ENOENT:
                        raise
                    # File doesn't exist. Render it.
                    return True

                return chunkmtime > tile_mtime
                

        # For each chunk, do this:
        #   For each tile that the chunk touches, do this:
        #       Compare the last modified time of the chunk and tile. If the
        #       tile is older, mark it in a RendertileSet object as dirty.

        for chunkx, chunkz, chunkmtime in self.regionset.iterate_chunks():

            chunkcount += 1
            
            # Convert to diagonal coordinates
            chunkcol, chunkrow = util.convert_coords(chunkx, chunkz)

            # find tile coordinates. Remember tiles are identified by the
            # address of the chunk in their upper left corner.
            tilecol = chunkcol - chunkcol % 2
            tilerow = chunkrow - chunkrow % 4

            # Determine if this chunk is in a column that spans two columns of
            # tiles, which are the even columns.
            if chunkcol % 2 == 0:
                x_tiles = 2
            else:
                x_tiles = 1

            # Loop over all tiles that this chunk potentially touches.
            # The tile at tilecol,tilerow obviously contains the chunk, but so
            # do the next 4 tiles down because chunks are very tall, and maybe
            # the next column over too.
            for i in xrange(x_tiles):
                for j in xrange(5):

                    # This loop iteration is for the tile at this column and
                    # row:
                    c = tilecol - 2*i
                    r = tilerow + 4*j

                    # Make sure the tile is in the range according to the given
                    # depth. This won't happen unless the user has given -z to
                    # render a smaller area of the map than everything
                    if (
                            c < self.bounds.mincol or
                            c >= self.bounds.maxcol or
                            r < self.bounds.minrow or
                            r >= self.bounds.maxrow
                            ):
                        continue

                    # Computes the path in the quadtree from the col,row coordinates
                    tile = RenderTile.compute_path(c, r, depth)

                    if rendercheck == 2:
                        # Skip all other checks, mark tiles as dirty unconditionally
                        dirty.add(tile.path)
                        continue

                    # Stochastic check. Since we're scanning by chunks and not
                    # by tiles, and the tiles get checked multiple times for
                    # each chunk, this is only an approximation. The given
                    # probability is for a particular tile that needs
                    # rendering, but since a tile gets touched up to 32 times
                    # (once for each chunk in it), divide the probability by
                    # 32.
                    if rerender_prob and rerender_prob/32 > random.random():
                        dirty.add(tile.path)
                        continue

                    # Check if this tile has already been marked dirty. If so,
                    # no need to do any of the below.
                    if dirty.query_path(tile.path):
                        continue

                    # Check mtimes and conditionally add tile to dirty set
                    if compare_mtimes(chunkmtime, tile):
                        dirty.add(tile.path)

        t = int(time.time()-stime)
        logging.debug("%s finished chunk scan. %s chunks scanned in %s second%s", 
                self, chunkcount, t,
                "s" if t != 1 else "")

        return dirty

    def __str__(self):
        return "<TileSet for %s>" % os.basename(self.outputdir)


def get_dirdepth(outputdir):
    """Returns the current depth of the tree on disk

    """
    # Traverses down the first directory until it reaches one with no
    # subdirectories. While all paths of the tree may not exist, all paths
    # of the tree should and are assumed to be the same depth

    # This function returns a list of all subdirectories of the given
    # directory. It's slightly more complicated than you'd think it should be
    # because one must turn directory names returned by os.listdir into
    # relative/absolute paths before they can be passed to os.path.isdir()
    getsubdirs = lambda directory: [
            abssubdir
            for abssubdir in
                (os.path.join(directory,subdir) for subdir in os.listdir(directory))
            if os.path.isdir(abssubdir)
            ]

    depth = 1
    subdirs = getsubdirs(outputdir)
    while subdirs:
        subdirs = getsubdirs(subdirs[0])
        depth += 1

    return depth

class RendertileSet(object):
    """This object holds a set of render-tiles using a quadtree data structure.
    It is typically used to hold tiles that need rendering. This implementation
    collapses subtrees that are completely in or out of the set to save memory.
    
    Each instance of this class is a node in the tree, and therefore each
    instance is the root of a subtree.

    Each node knows its "level", which corresponds to the zoom level where 0 is
    the inner-most (most zoomed in) tiles.

    Instances hold the state of their children (in or out of the set). Leaf
    nodes are images and do not physically exist in the tree as objects, but
    are represented as booleans held by the objects at the second-to-last
    level; level 1 nodes keep track of leaf image state. Level 2 nodes keep
    track of level 1 state, and so fourth.

    
    """
    __slots__ = ("depth", "children")
    def __init__(self, depth):
        """Initialize a new tree with the specified depth. This actually
        initializes a node, which is the root of a subtree, with `depth` levels
        beneath it.

        """
        # Stores the depth of the tree according to this node. This is not the
        # depth of this node, but rather the number of levels below this node
        # (including this node).
        self.depth = depth

        # the self.children array holds the 4 children of this node. This
        # follows the same quadtree convention as elsewhere: children 0, 1, 2,
        # 3 are the upper-left, upper-right, lower-left, and lower-right
        # respectively
        # Values are:
        # False
        #   All children down this subtree are not in the set
        # True
        #   All children down this subtree are in the set
        # A RendertileSet instance
        #   the instance defines which children down that subtree are in the
        #   set.
        # A node with depth=1 cannot have a RendertileSet instance in its
        # children since its children are leaves, representing images, not more
        # tree
        self.children = [False] * 4

    def posttraversal(self):
        """Returns an iterator over tile paths for every tile in the
        set, including the explictly marked render-tiles, as well as the
        implicitly marked ancestors of those render-tiles. Returns in
        post-traversal order, so that tiles with dependencies will always be
        yielded after their dependencies.

        """
        # XXX Implement Me!
        raise NotImplementedError()

    def add(self, path):
        """Marks the requested leaf node as in this set
        
        Path is an iterable of integers representing the path to the leaf node
        that is to be added to the set
        
        """
        path = list(path)
        assert len(path) == self.depth
        path.reverse()
        self._set_add_helper(path)

    def _set_add_helper(self, path):
        """Recursive helper for add()

        Expects path to be a list in reversed order

        If *all* the nodes below this one are in the set, this function returns
        true. Otherwise, returns None.

        """

        if self.depth == 1:
            # Base case
            self.children[path[0]] = True

            # Check to see if all children are in the set
            if all(self.children):
                return True
        else:
            # Recursive case

            childnum = path.pop()
            child = self.children[childnum]

            if child == False:
                # Create a new node and recurse.
                # (The use of __class__ is so possible subclasses of this class
                # work as expected)
                child = self.__class__(self.depth-1)
                child._set_add_helper(path)
                self.children[childnum] = child
            elif child == True:
                # Every child is already in the set and the subtree is already
                # collapsed. Nothing to do.
                return
            else:
                # subtree is mixed. Recurse to the already existing child node
                ret = child._set_add_helper(path)
                if ret:
                    # Child says every descendent is in the set, so we can
                    # purge the subtree and mark it as such. The subtree will
                    # be garbage collected when this method exits.
                    self.children[childnum] = True

                    # Since we've marked an entire sub-tree as in the set, we
                    # may be able to signal to our parent to do the same
                    if all(x is True for x in self.children):
                        return True

    def __iter__(self):
        return self.iterate()
    def iterate(self, level=None):
        """Returns an iterator over every tile in this set. Each item yielded
        is a sequence of integers representing the quadtree path to the tiles
        in the set. Yielded sequences are of length self.depth.

        If level is None, iterates over tiles of the highest level, i.e.
        worldtiles. If level is a value between 0 and the depth of this tree,
        this method iterates over tiles at that level. Zoom level 0 is zoomed
        all the way out, zoom level `depth` is all the way in.

        In other words, specifying level causes the tree to be iterated as if
        it was only that depth.

        """
        if level is None:
            todepth = 1
        else:
            if not (level > 0 and level <= self.depth):
                raise ValueError("Level parameter must be between 1 and %s" % self.depth)
            todepth = self.depth - level + 1

        return (tuple(reversed(rpath)) for rpath in self._iterate_helper(todepth))

    def _iterate_helper(self, todepth):
        if self.depth == todepth:
            # Base case
            if self.children[0]: yield [0]
            if self.children[1]: yield [1]
            if self.children[2]: yield [2]
            if self.children[3]: yield [3]

        else:
            # Higher levels:
            for c, child in enumerate(self.children):
                if child == True:
                    # All render-tiles are in the set down this subtree,
                    # iterate over every leaf using iterate_base4
                    for x in iterate_base4(self.depth-todepth):
                        x = list(x)
                        x.append(c)
                        yield x
                elif child != False:
                    # Mixed in/out of the set down this subtree, recurse
                    for path in child._iterate_helper(todepth):
                        path.append(c)
                        yield path

    def query_path(self, path):
        """Queries for the state of the given tile in the tree.

        Returns True for items in the set, False otherwise. Works for
        rendertiles as well as upper tiles (which are True if they have a
        descendent that is in the set)

        """
        # Traverse the tree down the given path. If the tree has been
        # collapsed, then just return the stored boolean. Otherwise, if we find
        # the specific tree node requested, return its state using the
        # __nonzero__ call.
        treenode = self
        for pathelement in path:
            treenode = treenode.children[pathelement]
            if not isinstance(treenode, RendertileSet):
                return treenode

        # If the method has not returned at this point, treenode is the
        # requested node, but it is an inner node with possibly mixed state
        # subtrees. If any of the children are True return True. This call
        # relies on the __nonzero__ method
        return bool(treenode)

    def __nonzero__(self):
        """Returns the boolean context of this particular node. If any
        descendent of this node is True return True. Otherwise, False.

        """
        # Any chilren that are True or are a RendertileSet that evaluate to
        # True
        # IDEA: look at all children for True before recursing Better idea:
        # every node except the root /must/ have a descendent in the set or it
        # wouldn't exist. This assumption is only valid as long as there is no
        # method to remove a tile from the set.
        return any(self.children)

    def count(self):
        """Returns the total number of render-tiles in this set.

        """
        # TODO: Make this more efficient (although for even the largest trees,
        # this takes only seconds)
        c = 0
        for _ in self.iterate():
            c += 1
        return c

class RenderTile(object):
    """A simple container class that represents a single render-tile.

    A render-tile is a tile that is rendered, not a tile composed of other
    tiles (composite-tile).

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
        # os.path.join would be the proper way to do this path concatenation,
        # but it is surprisingly slow, probably because it checks each path
        # element if it begins with a slash. Since we know these components are
        # all relative, just concatinate with os.path.sep
        pathcomponents = [tiledir]
        pathcomponents.extend(str(x) for x in self.path)
        path = os.path.sep.join(pathcomponents)
        imgpath = ".".join((path, imgformat))
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
