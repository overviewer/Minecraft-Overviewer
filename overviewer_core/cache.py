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

Each cache class should implement the standard container type interface
(__getitem__ and __setitem__), as well as provide a "hits" and "misses"
attribute.

"""

class LRUCache(object):
    """A simple, generic, in-memory LRU cache that implements the standard
    python container interface.

    An ordered dict type would simplify this implementation a bit, but we want
    Python 2.6 compatibility and the standard library ordereddict was added in
    2.7. It's probably okay because this implementation can be tuned for
    exactly what we need and nothing more.

    This implementation keeps a linked-list of cache keys and values, ordered
    in least-recently-used order. A dictionary maps keys to linked-list nodes.

    On cache hit, the link is moved to the end of the list. On cache miss, the
    first item of the list is evicted. All operations have constant time
    complexity (dict lookups are worst case O(n) time)

    """
    class _LinkNode(object):
        __slots__ = ['left', 'right', 'key', 'value']
        def __init__(self,l=None,r=None,k=None,v=None):
            self.left = l
            self.right = r
            self.key = k
            self.value = v

    def __init__(self, size=100, destructor=None):
        """Initialize a new LRU cache with the given size.

        destructor, if given, is a callable that is called upon an item being
        evicted from the cache. It takes one argument, the value stored in the
        cache.

        """
        self.cache = {}

        # Two sentinel nodes at the ends of the linked list simplify boundary
        # conditions in the code below.
        self.listhead = LRUCache._LinkNode()
        self.listtail = LRUCache._LinkNode()
        self.listhead.right = self.listtail
        self.listtail.left = self.listhead

        self.hits = 0
        self.misses = 0

        self.size = size

        self.destructor = destructor

    # Initialize an empty cache of the same size for worker processes
    def __getstate__(self):
        return self.size
    def __setstate__(self, size):
        self.__init__(size)

    def __getitem__(self, key):
        try:
            link = self.cache[key]
        except KeyError:
            self.misses += 1
            raise

        # Disconnect the link from where it is
        link.left.right = link.right
        link.right.left = link.left

        # Insert the link at the end of the list
        tail = self.listtail
        link.left = tail.left
        link.right = tail
        tail.left.right = link
        tail.left = link

        self.hits += 1
        return link.value

    def __setitem__(self, key, value):
        cache = self.cache
        if key in cache:
            # Shortcut this case
            cache[key].value = value
            return
        if len(cache) >= self.size:
            # Evict a node
            link = self.listhead.right
            del cache[link.key]
            link.left.right = link.right
            link.right.left = link.left
            d = self.destructor
            if d:
                d(link.value)
            del link

        # The node doesn't exist already, and we have room for it. Let's do this.
        tail = self.listtail
        link = LRUCache._LinkNode(tail.left, tail,key,value)
        tail.left.right = link
        tail.left = link

        cache[key] = link

    def __delitem__(self, key):
        # Used to flush the cache of this key
        cache = self.cache
        link = cache[key]
        del cache[key]
        link.left.right = link.right
        link.right.left = link.left
        
        # Call the destructor
        d = self.destructor
        if d:
            d(link.value)
