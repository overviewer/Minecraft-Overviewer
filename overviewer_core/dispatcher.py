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

class Dispatcher(object):
    def render_all(self, tilesetlist, status_callback):
        # TODO use status callback
        
        # preprocessing
        for tileset in tilesetlist:
            tileset.do_preprocessing()
        
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
            
            # go through these iterators round-robin style
            for tileset, workitem in util.roundrobin(work_iterators):
                self.dispatch(tileset, workitem)
            
            # after each phase, wait for the jobs to finish
            self.finish_jobs()
    
    def dispatch(self, tileset, workitem):
        tileset.do_work(workitem)
    def finish_jobs(self):
        pass
