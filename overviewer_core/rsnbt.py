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

import redstone as rs
from .nbt import CorruptRegionError, CorruptChunkError, CorruptNBTError

def load(path):
    try:
        n = rs.NBT.parse_from_file(path)
        return _load_from_nbt(n)
    except RuntimeError, e:
        raise CorruptNBTError(str(e))

def load_region(path):
    return MCRFileReader(path)

def _load_from_tag(t):
    if t.type == rs.TAG_LIST:
        return map(_load_from_tag, t.value)
    elif t.type == rs.TAG_COMPOUND:
        ret = {}
        for key, val in t.iteritems():
            ret[unicode(key)] = _load_from_tag(val)
        return ret
    elif t.type == rs.TAG_STRING:
        return unicode(t.value)
    else:
        return t.value

def _load_from_nbt(n):
    return (n.name, _load_from_tag(n.root))

class MCRFileReader(object):
    def __init__(self, path):
        try:
            self.r = rs.Region.open(path)
        except RuntimeError, e:
            raise CorruptRegionError(str(e))
    
    def close(self):
        del self.r
        self.r = None

    def get_chunks(self):
        for x in xrange(32): 
            for z in xrange(32): 
                if self.r.contains_chunk(x, z):
                    yield (x, z)
        
    def get_chunk_timestamp(self, x, z):
        x = x % 32
        z = z % 32
        return self.r.get_chunk_timestamp(x, z)
    
    def chunk_exists(self, x, z):
        x = x % 32
        z = z % 32
        return self.r.contains_chunk(x, z)

    def load_chunk(self, x, z):
        x = x % 32
        z = z % 32
        
        if not self.chunk_exists(x, z):
            return None
        
        try:
            n = rs.NBT.parse_from_region(self.r, x, z)
            return _load_from_nbt(n)
        except RuntimeError, e:
            raise CorruptChunkError(str(e))
