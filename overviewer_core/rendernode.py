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
import Queue
import itertools
from itertools import cycle, islice
import os
import os.path
import functools
import re
import shutil
import collections
import json
import logging
import util
import textures
import c_overviewer
import cPickle
import stat
import errno 
import time
from time import gmtime, strftime, sleep


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
    for mode in rendernode.options.custom_rendermodes:
        c_overviewer.add_custom_render_mode(mode, rendernode.options.custom_rendermodes[mode])
    for mode in rendernode.options.rendermode_options:
        c_overviewer.set_render_mode_options(mode, rendernode.options.rendermode_options[mode])
    
    # load biome data in each process, if needed
    for quadtree in rendernode.quadtrees:
        if quadtree.world.useBiomeData:
            # make sure we've at least *tried* to load the color arrays in this process...
            textures.prepareBiomeData(quadtree.world.worlddir)
            if not textures.grasscolor or not textures.foliagecolor:
                raise Exception("Can't find grasscolor.png or foliagecolor.png")
            # only load biome data once
            break
                    
#http://docs.python.org/library/itertools.html    
def roundrobin(iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

            
class RenderNode(object):
    def __init__(self, quadtrees, options):
        """Distributes the rendering of a list of quadtrees."""

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

        manager = multiprocessing.Manager() 
        # queue for receiving interesting events from the renderer
        # (like the discovery of signs!
        #stash into the world object like we stash an index into the quadtree
        for world in self.worlds:
            world.poi_q = manager.Queue() 


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
                
    def go(self, procs):
        """Renders all tiles"""
        
        logging.debug("Parent process {0}".format(os.getpid()))
        # Create a pool
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
                
        # 1 quadtree object per rendermode requested
        quadtrees = self.quadtrees
        
        # Determine the total number of tiles by adding up the number of tiles
        # from each quadtree. Also find the max zoom level (max_p). Even though
        # each quadtree will always have the same zoom level, this bit of code
        # does not make that assumption.
        max_p = 0
        total = 0
        for q in quadtrees:
            total += 4**q.p
            if q.p > max_p:
                max_p = q.p
        self.max_p = max_p

        # The next sections of code render the highest zoom level of tiles. The
        # section after render the other levels.
        results = collections.deque()
        complete = 0
        logging.info("Rendering highest zoom level of tiles now.")
        logging.info("Rendering {0} layer{1}".format(len(quadtrees),'s' if len(quadtrees) > 1 else '' ))
        logging.info("There are {0} tiles to render".format(total))        
        logging.info("There are {0} total levels to render".format(self.max_p))
        logging.info("Don't worry, each level has only 25% as many tiles as the last.")
        logging.info("The others will go faster")
        count = 0
        batch_size = 4*len(quadtrees)
        while batch_size < 10:
            batch_size *= 2
        timestamp = time.time()
        for result in self._apply_render_worldtiles(pool,batch_size):
            results.append(result)               
            # every second drain some of the queue
            timestamp2 = time.time()
            if timestamp2 >= timestamp + 1:
                timestamp = timestamp2                
                count_to_remove = (1000//batch_size)
                if count_to_remove < len(results):
                    for world in self.worlds:
                        try:
                            while (1):
                                # an exception will break us out of this loop
                                item = world.poi_q.get(block=False)
                                if item[0] == "newpoi":
                                    if item[1] not in world.POI:
                                        #print "got an item from the queue!"
                                        world.POI.append(item[1])
                                elif item[0] == "animal":
                                    if item[1] not in world.POI:
                                        world.POI.append(item[1])
                                elif item[0] == "removePOI":
                                    world.persistentData['POI'] = filter(lambda x: x['chunk'] != item[1], world.persistentData['POI'])

                                elif item[0] == "rendered":
                                    self.rendered_tiles.append(item[1])

                        except Queue.Empty:
                            pass
                    while count_to_remove > 0:
                        count_to_remove -= 1
                        complete += results.popleft().get()
                        self.print_statusline(complete, total, 1)  
            if len(results) > (10000//batch_size):
                # Empty the queue before adding any more, so that memory
                # required has an upper bound
                while len(results) > (500//batch_size):
                    complete += results.popleft().get()
                    self.print_statusline(complete, total, 1)

        # Wait for the rest of the results
        while len(results) > 0:
            complete += results.popleft().get()
            self.print_statusline(complete, total, 1)
        for world in self.worlds:    
            try:
                while (1):
                    # an exception will break us out of this loop
                    item = world.poi_q.get(block=False)
                    if item[0] == "newpoi":
                        if item[1] not in world.POI:
                            #print "got an item from the queue!"
                            world.POI.append(item[1])
                    elif item[0] == "animal":
                        if item[1] not in world.POI:
                            world.POI.append(item[1])
                    elif item[0] == "removePOI":
                        world.persistentData['POI'] = filter(lambda x: x['chunk'] != item[1], world.persistentData['POI'])
                    elif item[0] == "rendered":
                        self.rendered_tiles.append(item[1])

            except Queue.Empty:
                pass

        self.print_statusline(complete, total, 1, True)

        # The highest zoom level has been rendered.
        # Now do the lower zoom levels
        for zoom in xrange(self.max_p-1, 0, -1):
            level = self.max_p - zoom + 1
            assert len(results) == 0
            complete = 0
            total = 0
            for q in quadtrees:
                if zoom <= q.p:
                    total += 4**zoom
            logging.info("Starting level {0}".format(level))
            timestamp = time.time()
            for result in self._apply_render_inntertile(pool, zoom,batch_size):
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
                if len(results) > (10000/batch_size):
                    while len(results) > (500/batch_size):
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

    def _apply_render_worldtiles(self, pool,batch_size):
        """Returns an iterator over result objects. Each time a new result is
        requested, a new task is added to the pool and a result returned.
        """
        if batch_size < len(self.quadtrees):
            batch_size = len(self.quadtrees)
        batch = []
        jobcount = 0
        # roundrobin add tiles to a batch job (thus they should all roughly work on similar chunks)
        iterables = [q.get_worldtiles() for q in self.quadtrees]
        for job in roundrobin(iterables):
            # fixup so the worker knows which quadtree this is                 
            job[0] = job[0]._render_index
            # Put this in the batch to be submited to the pool  
            batch.append(job)
            jobcount += 1
            if jobcount >= batch_size:
                jobcount = 0
                yield pool.apply_async(func=render_worldtile_batch, args= [batch])
                batch = []
        if jobcount > 0:
            yield pool.apply_async(func=render_worldtile_batch, args= [batch])

    def _apply_render_inntertile(self, pool, zoom,batch_size):
        """Same as _apply_render_worltiles but for the inntertile routine.
        Returns an iterator that yields result objects from tasks that have
        been applied to the pool.
        """
        
        if batch_size < len(self.quadtrees):
            batch_size = len(self.quadtrees)
        batch = []
        jobcount = 0
        # roundrobin add tiles to a batch job (thus they should all roughly work on similar chunks)
        iterables = [q.get_innertiles(zoom) for q in self.quadtrees if zoom <= q.p]
        for job in roundrobin(iterables):
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
            
@catch_keyboardinterrupt
def render_worldtile_batch(batch):
    # batch is a list. Each item is [quadtree_id, colstart, colend, rowstart, rowend, tilepath]
    global child_rendernode
    rendernode = child_rendernode
    count = 0
    #logging.debug("{0} working on batch of size {1}".format(os.getpid(),len(batch)))        
    for job in batch:
        count += 1    
        quadtree = rendernode.quadtrees[job[0]]
        colstart = job[1]
        colend = job[2]
        rowstart = job[3]
        rowend = job[4]
        path = job[5]
        poi_queue = quadtree.world.poi_q
        path = quadtree.full_tiledir+os.sep+path        
        # (even if tilechunks is empty, render_worldtile will delete
        # existing images if appropriate)    
        # And uses these chunks
        tilechunks = quadtree.get_chunks_in_range(colstart, colend, rowstart,rowend)
        #logging.debug(" tilechunks: %r", tilechunks)
        
        quadtree.render_worldtile(tilechunks,colstart, colend, rowstart, rowend, path, poi_queue)
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
    
