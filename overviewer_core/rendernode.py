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

from __future__ import division
import multiprocessing
import Queue
import os
import os.path
import functools
import collections
import logging
import time

from . import textures
from . import util
from . import quadtree
import c_overviewer

"""
This module has routines related to distributing the render job to multiple nodes

"""

def catch_keyboardinterrupt(func):
    """Decorator that catches a keyboardinterrupt and raises a real exception
    so that multiprocessing will propagate it properly"""
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            logging.error("Ctrl-C caught!")
            raise Exception("Exiting")
        except:
            import traceback
            traceback.print_exc()
            raise
    return newfunc
    
child_rendernode = None
def pool_initializer(rendernode):
    logging.debug("Child process {0}".format(os.getpid()))
    #stash the quadtree objects in a global variable after fork() for windows compat.
    global child_rendernode
    child_rendernode = rendernode
    
    # make sure textures are generated for this process
    # and initialize c_overviewer
    textures.generate(path=rendernode.options.get('textures_path', None),
            north_direction=rendernode.options.get('north_direction', None))
    c_overviewer.init_chunk_render()
    
    # setup c_overviewer rendermode customs / options
    for mode in rendernode.builtin_custom_rendermodes:
        c_overviewer.add_custom_render_mode(mode, rendernode.builtin_custom_rendermodes[mode])
    for mode in rendernode.options.custom_rendermodes:
        c_overviewer.add_custom_render_mode(mode, rendernode.options.custom_rendermodes[mode])
    for mode in rendernode.options.rendermode_options:
        c_overviewer.set_render_mode_options(mode, rendernode.options.rendermode_options[mode])
    
    # load biome data in each process, if needed
    for qtree in rendernode.quadtrees:
        if qtree.world.useBiomeData:
            # make sure we've at least *tried* to load the color arrays in this process...
            textures.prepareBiomeData(qtree.world.worlddir)
            if not textures.grasscolor or not textures.foliagecolor:
                raise Exception("Can't find grasscolor.png or foliagecolor.png")
            # only load biome data once
            break
                    
            
class RenderNode(object):
    def __init__(self, quadtrees, options):
        """Distributes the rendering of a list of quadtrees.
        
        This class name is slightly misleading: it does not represent a worker
        process, it coordinates the rendering of the given quadtrees across
        many worker processes.

        This class tries not to make any assumptions on whether the given
        quadtrees share the same world or whether the given quadtrees share the
        same depth/structure. However, those assumptions have not been checked;
        quadtrees right now always share the same depth, structure, and
        associated world objects. Beware of mixing and matching quadtrees from
        different worlds!
        
        """

        if not len(quadtrees) > 0:
            raise ValueError("there must be at least one quadtree to work on")    

        self.options = options
        # A list of quadtree.QuadTree objects representing each rendermode
        # requested
        self.quadtrees = quadtrees
        #List of changed tiles
        self.rendered_tiles = []

        #bind an index value to the quadtree so we can find it again
        #and figure out which worlds are where
        self.worlds = []
        for i, q in enumerate(quadtrees):
            q._render_index = i
            i += 1   
            if q.world not in self.worlds:
                self.worlds.append(q.world)            

        # queue for receiving interesting events from the renderer
        # (like the discovery of signs!)
        # stash into the world object like we stash an index into the quadtree
        #
        # TODO: Managers spawn a sub-process to manage their objects. If p=1,
        # fall back to a non-managed queue (like Queue.Queue). (While the
        # management process won't do much processing, part of the point of p=1
        # is to ease debugging and profiling by keeping everything in one
        # process/thread)
        manager = multiprocessing.Manager() 
        for world in self.worlds:
            world.poi_q = manager.Queue() 

        self._last_print_count = 0
        self._last_print_level = 0
        self._last_print_time = None

    def print_statusline(self, complete, total, level, unconditional=False):
        if unconditional:
            pass
        elif complete < 100:
            if not complete % 25 == 0:
                return
        elif complete < 1000:
            if not complete % 100 == 0:
                return
        else:
            if not complete % 1000 == 0:
                return
        logging.info("{0}/{1} ({4}%) tiles complete on level {2}/{3}".format(
                complete, total, level, self.max_p, '%.1f' % ( (100.0 * complete) / total) ))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            now = time.time()
            if self._last_print_level == level:
                deltacount = complete - self._last_print_count
                deltat = now - self._last_print_time
                avg = deltacount / deltat
                logging.debug("%i tiles rendered in %.1f seconds. Avg: %.1f tiles per sec",
                        deltacount, deltat, avg)

            self._last_print_level = level
            self._last_print_count = complete
            self._last_print_time = now
                
    def go(self, procs):
        """Renders all tiles"""
        
        # Signal to the quadtrees to scan the chunks and their respective tile
        # directories to find what needs to be rendered. We get from this the
        # total tiles that need to be rendered (at the highest level across all
        # quadtrees) as well as a list of [qtree, DirtyTiles object]
        total_rendertiles, dirty_list = self._get_dirty_tiles(procs)

        # Create a pool
        logging.debug("Parent process {0}".format(os.getpid()))
        if procs == 1:
            pool = FakePool()
            pool_initializer(self)
        else:
            pool_initializer(self)
            pool = multiprocessing.Pool(processes=procs,initializer=pool_initializer,initargs=(self,))
            
            #warm up the pool so it reports all the worker id's
            if logging.getLogger().level >= 10:
                pool.map(bool,xrange(multiprocessing.cpu_count()),1)
            else:
                pool.map_async(bool,xrange(multiprocessing.cpu_count()),1)
                
        # The list of quadtrees. There is 1 quadtree object per rendermode
        # requested
        quadtrees = self.quadtrees
        
        # Find the max zoom level (max_p). Even though each quadtree will
        # always have the same zoom level with the current implementation, this
        # bit of code does not make that assumption.
        # max_p is stored in the instance so self.print_statusline can see it
        max_p = 0
        for q in quadtrees:
            if q.p > max_p:
                max_p = q.p
        self.max_p = max_p

        # Set a reasonable batch size. Groups of tiles are sent to workers in
        # batches this large. It should be a multiple of the number of
        # quadtrees so that each worker gets corresponding tiles from each
        # quadtree in the typical case.
        batch_size = 4*len(quadtrees)
        while batch_size < 10:
            batch_size *= 2
        logging.debug("Will push tiles to worker processes in batches of %s", batch_size)

        # The next sections of code render the highest zoom level of tiles. The
        # section after render the other levels.
        logging.info("")
        logging.info("Rendering highest zoom level of tiles now.")
        logging.info("Rendering {0} rendermode{1}".format(len(quadtrees),'s' if len(quadtrees) > 1 else '' ))
        logging.info("Started {0} worker process{1}".format(
            procs, "es" if procs != 1 else ""))
        logging.info("There are {0} tiles to render at this level".format(total_rendertiles))        
        logging.info("There are {0} total levels".format(self.max_p))

        # results is a queue of multiprocessing.AsyncResult objects. They are
        # appended to the end and held in the queue until they are pop'd and
        # the results collected.
        # complete holds the tally of the number of tiles rendered. Each
        # results object returns the number of tiles rendered and is
        # accumulated in complete
        results = collections.deque()
        complete = 0

        # Iterate over _apply_render_worldtiles(). That generator method
        # dispatches batches of tiles to the workers and yields results
        # objects. multiprocessing.AsyncResult objects are lazy objects that
        # are used to access the values returned by the worker's function,
        # which in this case, is render_worldtile_batch()
        timestamp = time.time()
        for result in self._apply_render_worldtiles(dirty_list, pool, batch_size):
            results.append(result)               

            # The results objects are lazy. The workers will process an item in
            # the pool when they get to it, and when we call result.get() it
            # blocks until the result is ready. We dont' want to add *all* the
            # tiles to the pool becuse we'd have to hold every result object in
            # memory. So we add a few batches to the pool / result objects to
            # the results queue, then drain the results queue, and repeat.

            # every second drain some of the queue
            timestamp2 = time.time()
            if timestamp2 >= timestamp + 1:
                timestamp = timestamp2                
                count_to_remove = (1000//batch_size)

                # If there are less than count_to_remove items in the results
                # queue, drain the point of interest queue and count_to_remove
                # items from the results queue
                if count_to_remove < len(results):
                    # Drain the point of interest queue for each world
                    for world in self.worlds:
                        try:
                            while (1):
                                # an exception will break us out of this loop
                                item = world.poi_q.get(block=False)
                                if item[0] == "newpoi":
                                    if item[1] not in world.POI:
                                        #print "got an item from the queue!"
                                        world.POI.append(item[1])
                                elif item[0] == "removePOI":
                                    world.persistentData['POI'] = filter(
                                            lambda x: x['chunk'] != item[1],
                                            world.persistentData['POI']
                                            )

                                elif item[0] == "rendered":
                                    self.rendered_tiles.append(item[1])

                        except Queue.Empty:
                            pass
                    # Now drain the results queue. results has more than
                    # count_to_remove items in it (as checked above)
                    while count_to_remove > 0:
                        count_to_remove -= 1
                        complete += results.popleft().get()
                        self.print_statusline(complete, total_rendertiles, 1)  

            # If the results queue is getting too big, drain all but
            # 500//batch_size items from it
            if len(results) > (10000//batch_size):
                # Empty the queue before adding any more, so that memory
                # required has an upper bound
                while len(results) > (500//batch_size):
                    complete += results.popleft().get()
                    self.print_statusline(complete, total_rendertiles, 1)

            # Loop back to the top, add more items to the queue, and repeat

        # Added all there is to add to the workers. Wait for the rest of the
        # results to come in before continuing
        while len(results) > 0:
            complete += results.popleft().get()
            self.print_statusline(complete, total_rendertiles, 1)

        # Now drain the point of interest queues for each world
        for world in self.worlds:    
            try:
                while (1):
                    # an exception will break us out of this loop
                    item = world.poi_q.get(block=False)
                    if item[0] == "newpoi":
                        if item[1] not in world.POI:
                            #print "got an item from the queue!"
                            world.POI.append(item[1])
                    elif item[0] == "removePOI":
                        world.persistentData['POI'] = filter(lambda x: x['chunk'] != item[1], world.persistentData['POI'])
                    elif item[0] == "rendered":
                        self.rendered_tiles.append(item[1])

            except Queue.Empty:
                pass

        # Print the final status line almost unconditionally
        if total_rendertiles > 0:
            self.print_statusline(complete, total_rendertiles, 1, True)

        ##########################################
        # The highest zoom level has been rendered.
        # Now do the lower zoom levels, working our way down to level 1
        for zoom in xrange(self.max_p-1, 0, -1):
            # "level" counts up for the status output
            level = self.max_p - zoom + 1

            assert len(results) == 0

            # Reset these for this zoom level
            complete = 0
            total = 0

            # Count up the total tiles to render at this zoom level
            for q in quadtrees:
                if zoom <= q.p:
                    total += 4**zoom

            logging.info("Starting level {0}".format(level))
            timestamp = time.time()

            # Same deal as above. _apply_render_innertile adds tiles in batch
            # to the worker pool and yields result objects that return the
            # number of tiles rendered.
            #
            # XXX Some quadtrees may not have tiles at this zoom level if we're
            # not assuming they all have the same depth!!
            for result in self._apply_render_innertile(pool, zoom,batch_size):
                results.append(result)
                # every second drain some of the queue
                timestamp2 = time.time()
                if timestamp2 >= timestamp + 1:
                    timestamp = timestamp2                
                    count_to_remove = (1000//batch_size)
                    if count_to_remove < len(results):
                        while count_to_remove > 0:
                            count_to_remove -= 1
                            complete += results.popleft().get()
                            self.print_statusline(complete, total, level)
                if len(results) > (10000//batch_size):
                    while len(results) > (500//batch_size):
                        complete += results.popleft().get()
                        self.print_statusline(complete, total, level)
            # Empty the queue
            while len(results) > 0:
                complete += results.popleft().get()
                self.print_statusline(complete, total, level)

            self.print_statusline(complete, total, level, True)

            logging.info("Done")

        pool.close()
        pool.join()

        # Do the final one right here:
        for q in quadtrees:
            q.render_innertile(os.path.join(q.destdir, q.tiledir), "base")

    def _get_dirty_tiles(self, procs):
        """Returns two items:
        1) The total number of tiles needing rendering
        2) a list of (qtree, DirtyTiles) objects holding which tiles in the
           respective quadtrees need to be rendered

        """
        all_dirty = []
        total = 0
        numqtrees = len(self.quadtrees)
        procs = min(procs, numqtrees)

        # Create a private pool to do the chunk scanning. I purposfully don't
        # use the same pool as the rendering. The process of chunk scanning
        # seems to take a lot of memory. Even though the final tree only takes
        # a few megabytes at most, I suspect memory fragmentation causes the
        # process to take much more memory than that during the scanning
        # process. Since we use a private pool just for this purpose, the trees
        # are piped back to the master process and the fragmented
        # memory-hogging processes exit, returning that extra memory to the OS.
        if procs == 1:
            pool = FakePool()
        else:
            pool = multiprocessing.Pool(processes=procs)

        logging.info("Scanning chunks and determining tiles to update for each rendermode requested.")
        logging.info("Doing %s scan%s in %s worker process%s",
                numqtrees, "s" if numqtrees != 1 else "",
                procs, "es" if procs != 1 else "",
                )

        # Push all scan jobs to the workers
        results = []
        for q in self.quadtrees:
            r = pool.apply_async(scan_quadtree_chunks, (q,))
            results.append(r)
        pool.close()

        # Wait for workers to finish
        for q, r in zip(self.quadtrees, results):
            dirty, numtiles = r.get()
            total += numtiles
            all_dirty.append((q, dirty))
        pool.join() # ought to be redundant

        logging.info("%s finished. %s %s to be rendered at the highest level",
                "All scans" if numqtrees != 1 else "Scan",
                total,
                # Probably won't happen, but just in case:
                "total tiles need" if total != 1 else "tile needs",
                )
        return total, all_dirty

    def _apply_render_worldtiles(self, tileset, pool,batch_size):
        """This generator method dispatches batches of tiles to the given
        worker pool with the function render_worldtile_batch(). It yields
        multiprocessing.AsyncResult objects. Each result object returns the
        number of tiles rendered.

        tileset is a list of (QuadtreeGen object, DirtyTiles object)
        
        Returns an iterator over result objects. Each time a new result is
        requested, a new batch of tasks are added to the pool and a result
        object is returned.
        """
        # Make sure batch_size is a sane value
        if batch_size < len(self.quadtrees):
            batch_size = len(self.quadtrees)

        # tileset is a list of (quadtreegen object, dirtytiles tree object)
        # We want: a sequence of iterators that each iterate over
        # [qtree obj, tile obj] items
        def mktileiterable(qtree, dtiletree):
            return ([qtree, quadtree.Tile.from_path(tilepath)] for tilepath in dtiletree.iterate_dirty())
        iterables = []
        for qtree, dtiletree in tileset:
            tileiterable = mktileiterable(qtree, dtiletree)
            iterables.append(tileiterable)
        
        # batch is a list of (qtree index, Tile object). This list is slowly
        # added to and when it reaches size batch_size, it is sent off to the
        # pool.
        batch = []

        # roundrobin add tiles to a batch job (thus they should all roughly work on similar chunks)
        for job in util.roundrobin(iterables):
            # fixup so the worker knows which quadtree this is. It's a bit of a
            # hack but it helps not to keep re-sending the qtree objects to the
            # workers.
            job[0] = job[0]._render_index
            # Put this in the batch to be submited to the pool  
            batch.append(job)
            if len(batch) >= batch_size:
                yield pool.apply_async(func=render_worldtile_batch, args= [batch])
                batch = []
        if len(batch):
            yield pool.apply_async(func=render_worldtile_batch, args= [batch])

    def _apply_render_innertile(self, pool, zoom,batch_size):
        """Same as _apply_render_worltiles but for the innertile routine.
        Returns an iterator that yields result objects from tasks that have
        been applied to the pool.
        """
        
        if batch_size < len(self.quadtrees):
            batch_size = len(self.quadtrees)
        batch = []
        jobcount = 0
        # roundrobin add tiles to a batch job (thus they should all roughly work on similar chunks)
        iterables = [q.get_innertiles(zoom) for q in self.quadtrees if zoom <= q.p]
        for job in util.roundrobin(iterables):
            # fixup so the worker knows which quadtree this is  
            job[0] = job[0]._render_index
            # Put this in the batch to be submited to the pool  
            batch.append(job)
            jobcount += 1
            if jobcount >= batch_size:
                jobcount = 0
                yield pool.apply_async(func=render_innertile_batch, args= [batch])
                batch = []
                
        if jobcount > 0:
            yield pool.apply_async(func=render_innertile_batch, args= [batch])    
            

########################################################################################
# The following three functions are entry points for workers in the multiprocessing pool

@catch_keyboardinterrupt
def render_worldtile_batch(batch):
    """Main entry point for workers processing a render-tile (also called a
    world tile).  Returns the number of tiles rendered, which is the length of
    the batch list passed in

    batch should be a list of (qtree index, tile object)

    """
    # batch is a list of items to process. Each item is [quadtree_id, Tile object]
    global child_rendernode
    rendernode = child_rendernode
    count = 0
    #logging.debug("{0} working on batch of size {1}".format(os.getpid(),len(batch)))        
    for job in batch:
        count += 1    
        quadtree = rendernode.quadtrees[job[0]]
        tile = job[1]

        quadtree.render_worldtile(tile)
    return count

@catch_keyboardinterrupt
def render_innertile_batch(batch):    
    global child_rendernode
    rendernode = child_rendernode
    count = 0   
    #logging.debug("{0} working on batch of size {1}".format(os.getpid(),len(batch)))
    for job in batch:
        count += 1        
        quadtree = rendernode.quadtrees[job[0]]               
        dest = quadtree.full_tiledir+os.sep+job[1]
        quadtree.render_innertile(dest=dest,name=job[2])
    return count

@catch_keyboardinterrupt
def scan_quadtree_chunks(qtree):
    """The entry point for workers when scanning chunks for tiles needing
    updating. Builds and returns a dirtytiles tree.

    Returns two things: the dirtytree from qtree.scan_chunks(), and the total
    from the tree.count() method

    """
    logging.debug("Scanning chunks for rendermode '%s'", qtree.rendermode)
    tree = qtree.scan_chunks()
    return tree, tree.count()
    
class FakeResult(object):
    def __init__(self, res):
        self.res = res
    def get(self):
        return self.res
class FakePool(object):
    """A fake pool used to render things in sync. Implements a subset of
    multiprocessing.Pool"""
    def apply_async(self, func, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        result = func(*args, **kwargs)
        return FakeResult(result)
    def close(self):
        pass
    def join(self):
        pass
    
