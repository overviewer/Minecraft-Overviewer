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

"""This module has supporting functions for the caching logic used in world.py.

"""
import functools

def lru_cache(max_size=100):
    """A quick-and-dirty LRU implementation.
    Uses a dict to store mappings, and a list to store orderings.

    Only supports positional arguments

    """
    def lru_decorator(fun):

        cache = {}
        lru_ordering = []
        
        @functools.wraps(fun)
        def new_fun(*args):
            try:
                result = cache[args]
            except KeyError:
                # cache miss =(
                new_fun.miss += 1
                result = fun(*args)

                # Insert into cache
                cache[args] = result
                lru_ordering.append(args)

                if len(cache) > max_size:
                    # Evict an item
                    del cache[ lru_ordering.pop(0) ]

            else:
                # Move the result item to the end of the list
                new_fun.hits += 1
                position = lru_ordering.index(args)
                lru_ordering.append(lru_ordering.pop(position))

            return result

        new_fun.hits = 0
        new_fun.miss = 0
        return new_fun

    return lru_decorator
