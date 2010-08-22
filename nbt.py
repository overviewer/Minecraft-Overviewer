import gzip
import struct

def load(fileobj):
    if isinstance(fileobj, basestring):
        # Is actually a filename
        fileobj = open(fileobj, 'r')
    return NBTFileReader(fileobj).read_all()

class NBTFileReader(object):
    def __init__(self, fileobj):
        self._file = gzip.GzipFile(fileobj=fileobj, mode='r')

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

