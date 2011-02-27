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

import gzip, zlib
import struct
import StringIO
import os

# decorator to handle filename or object as first parameter
def _file_loader(func):
    def wrapper(fileobj, *args):
        if isinstance(fileobj, basestring):
            if not os.path.isfile(fileobj):
               return None

            # Is actually a filename
            fileobj = open(fileobj, 'rb')
        return func(fileobj, *args)
    return wrapper

@_file_loader
def load(fileobj):
    return NBTFileReader(fileobj).read_all()

@_file_loader
def load_from_region(fileobj, x, y):
    nbt = MCRFileReader(fileobj).load_chunk(x, y)
    if not nbt:
        return None ## return none.  I think this is who we should indicate missing chunks
        #raise IOError("No such chunk in region: (%i, %i)" % (x, y))
    return nbt.read_all()

class NBTFileReader(object):
    def __init__(self, fileobj, is_gzip=True):
        if is_gzip:
            self._file = gzip.GzipFile(fileobj=fileobj, mode='rb')
        else:
            # pure zlib stream -- maybe later replace this with
            # a custom zlib file object?
            data = zlib.decompress(fileobj.read())
            self._file = StringIO.StringIO(data)

    # These private methods read the payload only of the following types
    def _read_tag_end(self):
        # Nothing to read
        return 0

    def _read_tag_byte(self):
        byte = self._file.read(1)
        return struct.unpack("b", byte)[0]
    
    def _read_tag_short(self):
        bytes = self._file.read(2)
        return struct.unpack(">h", bytes)[0]

    def _read_tag_int(self):
        bytes = self._file.read(4)
        return struct.unpack(">i", bytes)[0]

    def _read_tag_long(self):
        bytes = self._file.read(8)
        return struct.unpack(">q", bytes)[0]

    def _read_tag_float(self):
        bytes = self._file.read(4)
        return struct.unpack(">f", bytes)[0]

    def _read_tag_double(self):
        bytes = self._file.read(8)
        return struct.unpack(">d", bytes)[0]

    def _read_tag_byte_array(self):
        length = self._read_tag_int()
        bytes = self._file.read(length)
        return bytes

    def _read_tag_string(self):
        length = self._read_tag_short()

        # Read the string
        string = self._file.read(length)

        # decode it and return
        return string.decode("UTF-8")

    def _read_tag_list(self):
        tagid = self._read_tag_byte()
        length = self._read_tag_int()

        read_tagmap = {
                0: self._read_tag_end,
                1: self._read_tag_byte,
                2: self._read_tag_short,
                3: self._read_tag_int,
                4: self._read_tag_long,
                5: self._read_tag_float,
                6: self._read_tag_double,
                7: self._read_tag_byte_array,
                8: self._read_tag_string,
                9: self._read_tag_list,
                10:self._read_tag_compound,
                }

        read_method = read_tagmap[tagid]
        l = []
        for _ in xrange(length):
            l.append(read_method())
        return l

    def _read_tag_compound(self):
        # Build a dictionary of all the tag names mapping to their payloads
        tags = {}
        while True:
            # Read a tag
            tagtype = ord(self._file.read(1))

            if tagtype == 0:
                break

            name = self._read_tag_string()
            read_tagmap = {
                    0: self._read_tag_end,
                    1: self._read_tag_byte,
                    2: self._read_tag_short,
                    3: self._read_tag_int,
                    4: self._read_tag_long,
                    5: self._read_tag_float,
                    6: self._read_tag_double,
                    7: self._read_tag_byte_array,
                    8: self._read_tag_string,
                    9: self._read_tag_list,
                    10:self._read_tag_compound,
                    }
            payload = read_tagmap[tagtype]()
            
            tags[name] = payload

        return tags



    def read_all(self):
        """Reads the entire file and returns (name, payload)
        name is the name of the root tag, and payload is a dictionary mapping
        names to their payloads

        """
        # Read tag type
        tagtype = ord(self._file.read(1))
        if tagtype != 10:
            raise Exception("Expected a tag compound")

        # Read the tag name
        name = self._read_tag_string()

        payload = self._read_tag_compound()

        return name, payload


# For reference, the MCR format is outlined at
# <http://www.minecraftwiki.net/wiki/Beta_Level_Format>
class MCRFileReader(object):
    """A class for reading chunk region files, as introduced in the
    Beta 1.3 update. It provides functions for opening individual
    chunks (as instances of NBTFileReader), getting chunk timestamps,
    and for listing chunks contained in the file."""
    
    def __init__(self, fileobj):
        self._file = fileobj
        
        # cache used when the entire header tables are read in get_chunks()
        self._locations = None
        self._timestamps = None
        self._chunks = None
    
    def _read_24bit_int(self):
        """Read in a 24-bit, big-endian int, used in the chunk
        location table."""
        
        ret = 0
        bytes = self._file.read(3)
        for i in xrange(3):
            ret = ret << 8
            ret += struct.unpack("B", bytes[i])[0]
        
        return ret
    
    def _read_chunk_location(self, x=None, y=None):
        """Read and return the (offset, length) of the given chunk
        coordinate, or None if the requested chunk doesn't exist. x
        and y must be between 0 and 31, or None. If they are None,
        then there will be no file seek before doing the read."""
        
        if x != None and y != None:
            if (not x >= 0) or (not x < 32) or (not y >= 0) or (not y < 32):
                raise ValueError("Chunk location out of range.")
            
            # check for a cached value
            if self._locations:
                return self._locations[x + y * 32]
            
            # go to the correct entry in the chunk location table
            self._file.seek(4 * (x + y * 32))
        
        # 3-byte offset in 4KiB sectors
        offset_sectors = self._read_24bit_int()
        
        # 1-byte length in 4KiB sectors, rounded up
        byte = self._file.read(1)
        length_sectors = struct.unpack("B", byte)[0]
        
        # check for empty chunks
        if offset_sectors == 0 or length_sectors == 0:
            return None
        
        return (offset_sectors * 4096, length_sectors * 4096)
    
    def _read_chunk_timestamp(self, x=None, y=None):
        """Read and return the last modification time of the given
        chunk coordinate. x and y must be between 0 and 31, or
        None. If they are, None, then there will be no file seek
        before doing the read."""
        
        if x != None and y != None:
            if (not x >= 0) or (not x < 32) or (not y >= 0) or (not y < 32):
                raise ValueError("Chunk location out of range.")
            
            # check for a cached value
            if self._timestamps:
                return self._timestamps[x + y * 32]
            
            # go to the correct entry in the chunk timestamp table
            self._file.seek(4 * (x + y * 32) + 4096)
        
        bytes = self._file.read(4)
        timestamp = struct.unpack(">I", bytes)[0]
        
        return timestamp
    
    def get_chunks(self):
        """Return a list of all chunks contained in this region file,
        as a list of (x, y) coordinate tuples. To load these chunks,
        provide these coordinates to load_chunk()."""
        
        if self._chunks:
            return self._chunks
        
        self._chunks = []
        self._locations = []
        self._timestamps = []
        
        # go to the beginning of the file
        self._file.seek(0)
        
        # read chunk location table
        for y in xrange(32):
            for x in xrange(32):
                location = self._read_chunk_location()
                self._locations.append(location)
                if location:
                    self._chunks.append((x, y))
        
        # read chunk timestamp table
        for y in xrange(32):
            for x in xrange(32):
                timestamp = self._read_chunk_timestamp()
                self._timestamps.append(timestamp)
        
        return self._chunks
    
    def get_chunk_timestamp(self, x, y):
        """Return the given chunk's modification time. If the given
        chunk doesn't exist, this number may be nonsense. Like
        load_chunk(), this will wrap x and y into the range [0, 31].
        """
        
        return self._read_chunk_timestamp(x % 32, y % 32)
    
    def load_chunk(self, x, y):
        """Return a NBTFileReader instance for the given chunk, or
        None if the given chunk doesn't exist in this region file. If
        you provide an x or y not between 0 and 31, it will be
        modulo'd into this range (x % 32, etc.) This is so you can
        provide chunk coordinates in global coordinates, and still
        have the chunks load out of regions properly."""
        
        location = self._read_chunk_location(x % 32, y % 32)
        if not location:
            return None
        
        # seek to the data
        self._file.seek(location[0])
        
        # read in the chunk data header
        bytes = self._file.read(4)
        data_length = struct.unpack(">I", bytes)[0]
        bytes = self._file.read(1)
        compression = struct.unpack("B", bytes)[0]
        
        # figure out the compression
        is_gzip = True
        if compression == 1:
            # gzip -- not used by the official client, but trivial to support here so...
            is_gzip = True
        elif compression == 2:
            # deflate -- pure zlib stream
            is_gzip = False
        else:
            # unsupported!
            raise Exception("Unsupported chunk compression type: %i" % (compression,))
        
        # turn the rest of the data into a StringIO object
        # (using data_length - 1, as we already read 1 byte for compression)
        data = self._file.read(data_length - 1)
        data = StringIO.StringIO(data)
        
        return NBTFileReader(data, is_gzip=is_gzip)
