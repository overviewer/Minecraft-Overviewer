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

import util
import multiprocessing
import multiprocessing.managers
import Queue
import time
from signals import Signal

class Dispatcher(object):
    """This class coordinates the work of all the TileSet objects
    among one worker process. By subclassing this class and
    implementing setup_tilesets(), dispatch(), and close(), it is
    possible to create a Dispatcher that distributes this work to many
    worker processes.
    """
    def __init__(self):
        super(Dispatcher, self).__init__()

        # list of (tileset, workitem) tuples
        # keeps track of dispatched but unfinished jobs
        self._running_jobs = []
        # list of (tileset, workitem, dependencies) tuples
        # keeps track of jobs waiting to run after dependencies finish
        self._pending_jobs = []

    def render_all(self, tilesetlist, observer):
        """Render all of the tilesets in the given
        tilesetlist. status_callback is called periodically to update
        status. The callback should take the following arguments:
        (phase, items_completed, total_items), where total_items may
        be none if there is no useful estimate.
        """
        # TODO use status callback

        # setup tilesetlist
        self.setup_tilesets(tilesetlist)

        # iterate through all possible phases
        num_phases = [tileset.get_num_phases() for tileset in tilesetlist]
        for phase in xrange(max(num_phases)):
            # construct a list of iterators to use for this phase
            work_iterators = []
            for i, tileset in enumerate(tilesetlist):
                if phase < num_phases[i]:
                    def make_work_iterator(tset, p):
                        return ((tset, workitem) for workitem in tset.iterate_work_items(p))
                    work_iterators.append(make_work_iterator(tileset, phase))

            # keep track of total jobs, and how many jobs are done
            total_jobs = 0
            for tileset, phases in zip(tilesetlist, num_phases):
                if phase < phases:
                    jobs_for_tileset = tileset.get_phase_length(phase)
                    # if one is unknown, the total is unknown
                    if jobs_for_tileset is None:
                        total_jobs = None
                        break
                    else:
                        total_jobs += jobs_for_tileset

            observer.start(total_jobs)
            # go through these iterators round-robin style
            for tileset, (workitem, deps) in util.roundrobin(work_iterators):
                self._pending_jobs.append((tileset, workitem, deps))
                observer.add(self._dispatch_jobs())

            # after each phase, wait for the work to finish
            while len(self._pending_jobs) > 0 or len(self._running_jobs) > 0:
                observer.add(self._dispatch_jobs())

            observer.finish()

    def _dispatch_jobs(self):
        # helper function to dispatch pending jobs when their
        # dependencies are met, and to manage self._running_jobs
        dispatched_jobs = []
        finished_jobs = []

        pending_jobs_nodeps = [(j[0], j[1]) for j in self._pending_jobs]

        for pending_job in self._pending_jobs:
            tileset, workitem, deps = pending_job

            # see if any of the deps are in _running_jobs or _pending_jobs
            for dep in deps:
                if (tileset, dep) in self._running_jobs or (tileset, dep) in pending_jobs_nodeps:
                    # it is! don't dispatch this item yet
                    break
            else:
                # it isn't! all dependencies are finished
                finished_jobs += self.dispatch(tileset, workitem)
                self._running_jobs.append((tileset, workitem))
                dispatched_jobs.append(pending_job)

        # make sure to at least get finished jobs, even if we don't
        # submit any new ones...
        if len(dispatched_jobs) == 0:
            finished_jobs += self.dispatch(None, None)

        # clean out the appropriate lists
        for job in finished_jobs:
            self._running_jobs.remove(job)
        for job in dispatched_jobs:
            self._pending_jobs.remove(job)

        return len(finished_jobs)

    def close(self):
        """Close the Dispatcher. This should be called when you are
        done with the dispatcher, to ensure that it cleans up any
        processes or connections it may still have around.
        """
        pass

    def setup_tilesets(self, tilesetlist):
        """Called whenever a new list of tilesets are being used. This
        lets subclasses distribute the whole list at once, instead of
        for each work item."""
        pass

    def dispatch(self, tileset, workitem):
        """Dispatch the given work item. The end result of this call
        should be running tileset.do_work(workitem) somewhere. This
        function should return a list of (tileset, workitem) tuples
        that have completed since the last call. If tileset is None,
        then returning completed jobs is all this function should do.
        """
        if not tileset is None:
            tileset.do_work(workitem)
            return [(tileset, workitem),]
        return []

class MultiprocessingDispatcherManager(multiprocessing.managers.BaseManager):
    """This multiprocessing manager is responsible for giving worker
    processes access to the communication Queues, and also gives
    workers access to the current tileset list.
    """
    def _get_job_queue(self):
        return self.job_queue
    def _get_results_queue(self):
        return self.result_queue
    def _get_signal_queue(self):
        return self.signal_queue
    def _get_tileset_data(self):
        return self.tileset_data

    def __init__(self, address=None, authkey=None):
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.signal_queue = multiprocessing.Queue()

        self.tilesets = []
        self.tileset_version = 0
        self.tileset_data = [[], 0]

        self.register("get_job_queue", callable=self._get_job_queue)
        self.register("get_result_queue", callable=self._get_results_queue)
        self.register("get_signal_queue", callable=self._get_signal_queue)
        self.register("get_tileset_data", callable=self._get_tileset_data, proxytype=multiprocessing.managers.ListProxy)

        super(MultiprocessingDispatcherManager, self).__init__(address=address, authkey=authkey)

    @classmethod
    def from_address(cls, address, authkey, serializer):
        "Required to be implemented to make multiprocessing happy"
        c = cls(address=address, authkey=authkey)
        return c

    def set_tilesets(self, tilesets):
        """This is used in MultiprocessingDispatcher.setup_tilesets to
        update the tilesets each worker has access to. It also
        increments a `tileset_version` which is an easy way for
        workers to see if their tileset list is out-of-date without
        pickling and copying over the entire list.
        """
        self.tilesets = tilesets
        self.tileset_version += 1
        data = self.get_tileset_data()
        data[0] = self.tilesets
        data[1] = self.tileset_version


class MultiprocessingDispatcherProcess(multiprocessing.Process):
    """This class represents a single worker process. It is created
    automatically by MultiprocessingDispatcher, but it can even be
    used manually to spawn processes on different machines on the same
    network.
    """
    def __init__(self, manager):
        """Creates the process object. manager should be an instance
        of MultiprocessingDispatcherManager connected to the one
        created in MultiprocessingDispatcher.
        """
        super(MultiprocessingDispatcherProcess, self).__init__()
        self.job_queue = manager.get_job_queue()
        self.result_queue = manager.get_result_queue()
        self.signal_queue = manager.get_signal_queue()
        self.tileset_proxy = manager.get_tileset_data()

    def update_tilesets(self):
        """A convenience function to update our local tilesets to the
        current version in use by the MultiprocessingDispatcher.
        """
        self.tilesets, self.tileset_version = self.tileset_proxy._getvalue()

    def run(self):
        """The main work loop. Jobs are pulled from the job queue and
        executed, then the result is pushed onto the result
        queue. Updates to the tilesetlist are recognized and handled
        automatically. This is the method that actually runs in the
        new worker process.
        """
        # per-process job get() timeout
        timeout = 1.0

        # update our tilesets
        self.update_tilesets()

        # register for all available signals
        def register_signal(name, sig):
            def handler(*args, **kwargs):
                self.signal_queue.put((name, args, kwargs), False)
            sig.set_interceptor(handler)
        for name, sig in Signal.signals.iteritems():
            register_signal(name, sig)

        # notify that we're starting up
        self.result_queue.put(None, False)
        while True:
            try:
                job = self.job_queue.get(True, timeout)
                if job == None:
                    # this is a end-of-jobs sentinel
                    return

                # unpack job
                tv, ti, workitem = job

                if tv != self.tileset_version:
                    # our tilesets changed!
                    self.update_tilesets()
                    assert tv == self.tileset_version

                # do job
                ret = self.tilesets[ti].do_work(workitem)
                result = (ti, workitem, ret,)
                self.result_queue.put(result, False)
            except Queue.Empty:
                pass

class MultiprocessingDispatcher(Dispatcher):
    """A subclass of Dispatcher that spawns worker processes and
    distributes jobs to them to speed up processing.
    """
    def __init__(self, local_procs=-1, address=None, authkey=None):
        """Creates the dispatcher. local_procs should be the number of
        worker processes to spawn. If it's omitted (or negative)
        the number of available CPUs is used instead.
        """
        super(MultiprocessingDispatcher, self).__init__()

        # automatic local_procs handling
        if local_procs < 0:
            local_procs = multiprocessing.cpu_count()
        self.local_procs = local_procs

        self.outstanding_jobs = 0
        self.num_workers = 0
        self.manager = MultiprocessingDispatcherManager(address=address, authkey=authkey)
        self.manager.start()
        self.job_queue = self.manager.get_job_queue()
        self.result_queue = self.manager.get_result_queue()
        self.signal_queue = self.manager.get_signal_queue()

        # create and fill the pool
        self.pool = []
        for i in xrange(self.local_procs):
            proc = MultiprocessingDispatcherProcess(self.manager)
            proc.start()
            self.pool.append(proc)

    def close(self):
        # empty the queue
        self._handle_messages(timeout=0.0)
        while self.outstanding_jobs > 0:
            self._handle_messages()

        # send of the end-of-jobs sentinel
        for p in xrange(self.num_workers):
            self.job_queue.put(None, False)

        # TODO better way to be sure worker processes get the message
        time.sleep(1)

        # and close the manager
        self.manager.shutdown()
        self.manager = None
        self.pool = None

    def setup_tilesets(self, tilesets):
        self.manager.set_tilesets(tilesets)

    def dispatch(self, tileset, workitem):
        # handle the no-new-work case
        if tileset is None:
            return self._handle_messages()

        # create and submit the job
        tileset_index = self.manager.tilesets.index(tileset)
        self.job_queue.put((self.manager.tileset_version, tileset_index, workitem), False)
        self.outstanding_jobs += 1

        # make sure the queue doesn't fill up too much
        finished_jobs = self._handle_messages(timeout=0.0)
        while self.outstanding_jobs > self.num_workers * 10:
            finished_jobs += self._handle_messages()
        return finished_jobs

    def _handle_messages(self, timeout=0.01):
        # work function: takes results out of the result queue and
        # keeps track of how many outstanding jobs remain
        finished_jobs = []

        result_empty = False
        signal_empty = False
        while not (result_empty and signal_empty):
            if not result_empty:
                try:
                    result = self.result_queue.get(False)

                    if result != None:
                        # completed job
                        ti, workitem, ret = result
                        finished_jobs.append((self.manager.tilesets[ti], workitem))
                        self.outstanding_jobs -= 1
                    else:
                        # new worker
                        self.num_workers += 1
                except Queue.Empty:
                    result_empty = True
            if not signal_empty:
                try:
                    if timeout > 0.0:
                        name, args, kwargs = self.signal_queue.get(True, timeout)
                    else:
                        name, args, kwargs = self.signal_queue.get(False)
                    # timeout should only apply once
                    timeout = 0.0

                    sig = Signal.signals[name]
                    sig.emit_intercepted(*args, **kwargs)
                except Queue.Empty:
                    signal_empty = True

        return finished_jobs

    @classmethod
    def start_manual_process(cls, address, authkey):
        """A convenience method to start up a manual process, possibly
        on another machine. Address is a (hostname, port) tuple, and
        authkey must be the same as that provided to the
        MultiprocessingDispatcher constructor.
        """
        m = MultiprocessingDispatcherManager(address=address, authkey=authkey)
        m.connect()
        p = MultiprocessingDispatcherProcess(m)
        p.run()
