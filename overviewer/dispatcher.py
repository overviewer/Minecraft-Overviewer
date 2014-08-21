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

from . import util
import multiprocessing
import multiprocessing.managers
import pickle
import queue
import time

import overviewer.observer

class Worker(object):
    """This class provides jobs that can be dispatched by a
    Dispatcher. It provides default implementations for most methods,
    and you can derive from this to get started with making a
    Worker."""
    
    def get_num_phases(self):
        """This method returns an integer indicating how many phases
        of work this worker has to perform. Each phase of work is
        completed serially with the other phases; all work done by one
        phase is done before the next phase is started."""
        return 1
    
    def get_phase_length(self, phase):
        """This method returns an integer indicating how many work
        items there are in this phase. This number is used for purely
        informanional purposes. It can be exact, or an estimate. If
        there is no useful information on the size of a phase, return
        None."""
        return None
    
    def iterate_work_items(self, phase):
        """Takes a phase number (a non-negative integer). This method
        should return an iterator over work items and a list of
        dependencies i.e. (work_item, [d1, d2, ...]). The work items
        and dependencies can be any pickleable object; they are
        treated as opaque by the Dispatcher. The work item objects are
        passed back in to the do_work() method (perhaps in a
        different, identically configured instance).
        
        The dependency items are other work items that are compared
        for equality with work items that are already in the
        queue. The dispatcher guarantees that dependent items which
        are currently in the queue or in progress finish before the
        corresponding work item is started. Note that dependencies
        must have already been yielded as work items before they can
        be used as dependencies; the dispatcher requires this ordering
        or it cannot guarantee the dependencies are met."""
        raise NotImplementedError("Worker.iterate_work_items")
    
    def do_work(self, workobj):
        """Does the work for a given work object. This method is not
        expected to return anything, so the results of its work should
        be reflected by its side-effects."""
        raise NotImplementedError("Worker.do_work")

class Dispatcher(object):
    """This class coordinates the work of a list of Worker objects
    among one worker process. By subclassing this class and
    implementing setup_workers(), dispatch(), and close(), it is
    possible to create a Dispatcher that distributes this work to many
    worker processes.
    """
    def __init__(self):
        super(Dispatcher, self).__init__()

        # list of (worker, workitem) tuples
        # keeps track of dispatched but unfinished jobs
        self._running_jobs = []
        # list of (worker, workitem, dependencies) tuples
        # keeps track of jobs waiting to run after dependencies finish
        self._pending_jobs = []

    def dispatch_all(self, workerlist, observer=None):
        """Do all of the jobs provided by the given
        workerlist. observer should be an Observer object, or None, in
        which case it is ignored."""
        # setup workerlist
        self.setup_workers(workerlist)

        if observer is None:
            # Make a dummy observer
            observer = overviewer.observer.Observer()

        # iterate through all possible phases
        num_phases = [worker.get_num_phases() for worker in workerlist]
        for phase in range(max(num_phases)):
            # construct a list of iterators to use for this phase
            work_iterators = []
            for i, worker in enumerate(workerlist):
                if phase < num_phases[i]:
                    def make_work_iterator(wker, p):
                        return ((wker, workitem) for workitem in wker.iterate_work_items(p))
                    work_iterators.append(make_work_iterator(worker, phase))

            # keep track of total jobs, and how many jobs are done
            total_jobs = 0
            for worker, phases in zip(workerlist, num_phases):
                if phase < phases:
                    jobs_for_worker = worker.get_phase_length(phase)
                    # if one is unknown, the total is unknown
                    if jobs_for_worker is None:
                        total_jobs = None
                        break
                    else:
                        total_jobs += jobs_for_worker

            observer.start(total_jobs)
            # go through these iterators round-robin style
            for worker, (workitem, deps) in util.roundrobin(work_iterators):
                self._pending_jobs.append((worker, workitem, deps))
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
            worker, workitem, deps = pending_job

            # see if any of the deps are in _running_jobs or _pending_jobs
            for dep in deps:
                if (worker, dep) in self._running_jobs or (worker, dep) in pending_jobs_nodeps:
                    # it is! don't dispatch this item yet
                    break
            else:
                # it isn't! all dependencies are finished
                finished_jobs += self.dispatch(worker, workitem)
                self._running_jobs.append((worker, workitem))
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

    def setup_workers(self, workerlist):
        """Called whenever a new list of workers are being used. This
        lets subclasses distribute the whole list at once, instead of
        for each work item."""
        pass

    def dispatch(self, worker, workitem):
        """Dispatch the given work item. The end result of this call
        should be running worker.do_work(workitem) somewhere. This
        function should return a list of (worker, workitem) tuples
        that have completed since the last call. If worker is None,
        then returning completed jobs is all this function should do.
        """
        if not worker is None:
            worker.do_work(workitem)
            return [(worker, workitem),]
        return []

class MultiprocessingDispatcherManager(multiprocessing.managers.BaseManager):
    """This multiprocessing manager is responsible for giving worker
    processes access to the communication Queues, and also gives
    workers access to the current worker list.
    """
    def _get_job_queue(self):
        return self.job_queue
    def _get_results_queue(self):
        return self.result_queue
    def _get_worker_data(self):
        return self.worker_data

    def __init__(self, address=None, authkey=None):
        self.job_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

        self.workers = []
        self.worker_version = 0
        self.worker_data = [[], 0]

        self.register("get_job_queue", callable=self._get_job_queue)
        self.register("get_result_queue", callable=self._get_results_queue)
        self.register("get_worker_data", callable=self._get_worker_data, proxytype=multiprocessing.managers.ListProxy)

        super(MultiprocessingDispatcherManager, self).__init__(address=address, authkey=authkey)

    @classmethod
    def from_address(cls, address, authkey, serializer):
        "Required to be implemented to make multiprocessing happy"
        c = cls(address=address, authkey=authkey)
        return c

    def set_workers(self, workers):
        """This is used in MultiprocessingDispatcher.setup_workers to
        update the workers each worker has access to. It also
        increments a `worker_version` which is an easy way for
        workers to see if their worker list is out-of-date without
        pickling and copying over the entire list.
        """
        self.workers = workers
        self.worker_version += 1
        data = self.get_worker_data()
        data[0] = self.workers
        data[1] = self.worker_version


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
        self.worker_proxy = manager.get_worker_data()

    def update_workers(self):
        """A convenience function to update our local workers to the
        current version in use by the MultiprocessingDispatcher.
        """
        self.workers, self.worker_version = self.worker_proxy._getvalue()

    def run(self):
        """The main work loop. Jobs are pulled from the job queue and
        executed, then the result is pushed onto the result
        queue. Updates to the workerlist are recognized and handled
        automatically. This is the method that actually runs in the
        new worker process.
        """
        # per-process job get() timeout
        timeout = 1.0

        # update our workers
        self.update_workers()

        # notify that we're starting up
        self.result_queue.put(None, False)
        while True:
            try:
                job = self.job_queue.get(True, timeout)
                if job == None:
                    # this is a end-of-jobs sentinel
                    return

                # unpack job
                wv, wi, workitem = job

                if wv != self.worker_version:
                    # our workers changed!
                    self.update_workers()
                    assert wv == self.worker_version

                # do job
                ret = self.workers[wi].do_work(workitem)
                result = (wi, workitem, ret,)
                self.result_queue.put(result, False)
            except queue.Empty:
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

        # create and fill the pool
        self.pool = []
        for i in range(self.local_procs):
            proc = MultiprocessingDispatcherProcess(self.manager)
            proc.start()
            self.pool.append(proc)

    def close(self):
        # empty the queue
        self._handle_messages(timeout=0.0)
        while self.outstanding_jobs > 0:
            self._handle_messages()

        # send of the end-of-jobs sentinel
        for p in range(self.num_workers):
            self.job_queue.put(None, False)

        # TODO better way to be sure worker processes get the message
        time.sleep(1)

        # and close the manager
        self.manager.shutdown()
        self.manager = None
        self.pool = None

    def setup_workers(self, workers):
        self.manager.set_workers(workers)

    def dispatch(self, worker, workitem):
        # handle the no-new-work case
        if worker is None:
            return self._handle_messages()

        # create and submit the job
        worker_index = self.manager.workers.index(worker)
        self.job_queue.put((self.manager.worker_version, worker_index, workitem), False)
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
        while True:
            try:
                result = self.result_queue.get(False)
                if result != None:
                    # completed job
                    wi, workitem, ret = result
                    finished_jobs.append((self.manager.workers[wi], workitem))
                    self.outstanding_jobs -= 1
                else:
                    # new worker
                    self.num_workers += 1
            except queue.Empty:
                break

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
