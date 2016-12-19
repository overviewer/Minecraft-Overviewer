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
import functools

# decorator that turns the first argument from a string into an open file
# handle
def _file_loader(func):
    @functools.wraps(func)
    def wrapper(fileobj, *args):
        if isinstance(fileobj, basestring):
            # Is actually a filename
            fileobj = open(fileobj, 'rb', 4096)
        return func(fileobj, *args)
    return wrapper

@_file_loader
def load(fileobj):
    """Reads in the given file as NBT format, parses it, and returns the
    result as a (name, data) tuple.    
    """
    return NBTFileReader(fileobj).read_all()

@_file_loader
def load_region(fileobj):
    """Reads in the given file as a MCR region, and returns an object
    for accessing the chunks inside."""
    return MCRFileReader(fileobj)


class CorruptionError(Exception):
    pass
class CorruptRegionError(CorruptionError):
    """An exception raised when the MCRFileReader class encounters an
    error during region file parsing.
    """
    pass
class CorruptChunkError(CorruptionError):
    pass
class CorruptNBTError(CorruptionError):
    """An exception raised when the NBTFileReader class encounters
    something unexpected in an NBT file."""
    pass

class NBTFileReader(object):
    """Low level class that reads the Named Binary Tag format used by Minecraft

    """
    
    # compile the unpacker's into a classes
    _byte   = struct.Struct("b")
    _short  = struct.Struct(">h")
    _ushort = struct.Struct(">H")
    _int    = struct.Struct(">i")
    _uint   = struct.Struct(">I")
    _long   = struct.Struct(">q")
    _float  = struct.Struct(">f")
    _double = struct.Struct(">d") 
 
    def __init__(self, fileobj, is_gzip=True):
        """Create a NBT parsing object with the given file-like
        object. Setting is_gzip to False parses the file as a zlib
        stream instead."""
        if is_gzip:
            self._file = gzip.GzipFile(fileobj=fileobj, mode='rb')
        else:
            # pure zlib stream -- maybe later replace this with
            # a custom zlib file object?
            data = zlib.decompress(fileobj.read())
            self._file = StringIO.StringIO(data)

        # mapping of NBT type ids to functions to read them out
        self._read_tagmap = {
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
            11:self._read_tag_int_array,
        }

    # These private methods read the payload only of the following types
    def _read_tag_end(self):
        # Nothing to read
        return 0

    def _read_tag_byte(self):
        byte = self._file.read(1)
        return self._byte.unpack(byte)[0]
    
    def _read_tag_short(self):
        bytes = self._file.read(2)
        return self._short.unpack(bytes)[0]

    def _read_tag_int(self):
        bytes = self._file.read(4)
        return self._int.unpack(bytes)[0]

    def _read_tag_long(self):
        bytes = self._file.read(8)
        return self._long.unpack(bytes)[0]

    def _read_tag_float(self):
        bytes = self._file.read(4)
        return self._float.unpack(bytes)[0]

    def _read_tag_double(self):
        bytes = self._file.read(8)
        return self._double.unpack(bytes)[0]

    def _read_tag_byte_array(self):
        length = self._uint.unpack(self._file.read(4))[0]
        bytes = self._file.read(length)
        return bytes

    def _read_tag_int_array(self):
        length = self._uint.unpack(self._file.read(4))[0]
        int_bytes = self._file.read(length*4)
        return struct.unpack(">%ii" % length, int_bytes)

    def _read_tag_string(self):
        length = self._ushort.unpack(self._file.read(2))[0]
        # Read the string
        string = self._file.read(length)
        # decode it and return
        return string.decode("UTF-8")

    def _read_tag_list(self):
        tagid = self._read_tag_byte()
        length = self._uint.unpack(self._file.read(4))[0]

        read_method = self._read_tagmap[tagid]
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
            payload = self._read_tagmap[tagtype]()
            tags[name] = payload

        return tags
    
    def read_all(self):
        """Reads the entire file and returns (name, payload)
        name is the name of the root tag, and payload is a dictionary mapping
        names to their payloads

        """
        # Read tag type
        try:
            tagtype = ord(self._file.read(1))
            if tagtype != 10:
                raise Exception("Expected a tag compound")
            
            # Read the tag name
            name = self._read_tag_string()
            payload = self._read_tag_compound()
            
            return (name, payload)
        except (struct.error, ValueError, TypeError), e:
            raise CorruptNBTError("could not parse nbt: %s" % (str(e),))

# For reference, the MCR format is outlined at
# <http://www.minecraftwiki.net/wiki/Beta_Level_Format>
class MCRFileReader(object):
    """A class for reading chunk region files, as introduced in the
    Beta 1.3 update. It provides functions for opening individual
    chunks (as (name, data) tuples), getting chunk timestamps, and for
    listing chunks contained in the file.
    """
    
    _location_table_format = struct.Struct(">1024I")
    _timestamp_table_format = struct.Struct(">1024i")
    _chunk_header_format = struct.Struct(">I B")
    
    def __init__(self, fileobj):
        """This creates a region object from the given file-like
        object. Chances are you want to use load_region instead."""
        self._file = fileobj
        
        # read in the location table
        location_data = self._file.read(4096)
        if not len(location_data) == 4096:
            raise CorruptRegionError("invalid location table")
        # read in the timestamp table
        timestamp_data = self._file.read(4096)
        if not len(timestamp_data) == 4096:
            raise CorruptRegionError("invalid timestamp table")

        # turn this data into a useful list
        self._locations = self._location_table_format.unpack(location_data)
        self._timestamps = self._timestamp_table_format.unpack(timestamp_data)

    def close(self):
        """Close the region file and free any resources associated
        with keeping it open. Using this object after closing it
        results in undefined behaviour.
        """
        
        self._file.close()
        self._file = None

    def get_chunks(self):    
        """Return an iterator of all chunks contained in this region
        file, as (x, z) coordinate tuples. To load these chunks,
        provide these coordinates to load_chunk()."""
        
        for x in xrange(32): 
            for z in xrange(32): 
                if self._locations[x + z * 32] >> 8 != 0:
                    yield (x,z)
        
    def get_chunk_timestamp(self, x, z):
        """Return the given chunk's modification time. If the given
        chunk doesn't exist, this number may be nonsense. Like
        load_chunk(), this will wrap x and z into the range [0, 31].
        """
        x = x % 32
        z = z % 32        
        return self._timestamps[x + z * 32]   
    
    def chunk_exists(self, x, z):
        """Determines if a chunk exists."""
        x = x % 32
        z = z % 32
        return self._locations[x + z * 32] >> 8 != 0

    def load_chunk(self, x, z):
        """Return a (name, data) tuple for the given chunk, or
        None if the given chunk doesn't exist in this region file. If
        you provide an x or z not between 0 and 31, it will be
        modulo'd into this range (x % 32, etc.) This is so you can
        provide chunk coordinates in global coordinates, and still
        have the chunks load out of regions properly."""
        x = x % 32
        z = z % 32
        location = self._locations[x + z * 32]
        offset = (location >> 8) * 4096;
        sectors = location & 0xff;
        
        if offset == 0:
            return None
        
        # seek to the data
        self._file.seek(offset)
        
        # read in the chunk data header
        header = self._file.read(5)
        if len(header) != 5:
            raise CorruptChunkError("chunk header is invalid")
        data_length, compression =  self._chunk_header_format.unpack(header)
        
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
            raise CorruptRegionError("unsupported chunk compression type: %i (should be 1 or 2)" % (compression,))
        
        # turn the rest of the data into a StringIO object
        # (using data_length - 1, as we already read 1 byte for compression)
        data = self._file.read(data_length - 1)
        if len(data) != data_length - 1:
            raise CorruptRegionError("chunk length is invalid")
        data = StringIO.StringIO(data)
        
        try:
            return NBTFileReader(data, is_gzip=is_gzip).read_all()
        except CorruptionError:
            raise
        except Exception, e:
            raise CorruptChunkError("Misc error parsing chunk: " + str(e))
