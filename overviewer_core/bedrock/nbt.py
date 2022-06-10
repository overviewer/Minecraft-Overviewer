# A simple and flexible NBT parser.

import struct

# Allows for easy sequential reading of binary data
class DataReader:
  def __init__(self, data):
    self.data = data
    self.idx = 0

  def pop(self, key):
    key = "<{}".format(key)
    size = struct.calcsize(key)
    popped = struct.unpack(key, self.data[self.idx:self.idx + size])[0]
    self.idx += size
    return popped

  # Specific to the NBT string format, two bytes for size followed by that many bytes of string.
  def popString(self):
    size = self.pop("h")
    popped = struct.unpack("<{}s".format(size), self.data[self.idx:self.idx + size])[0]
    self.idx += size
    try:
      popped = popped.decode("utf-8")
    except UnicodeDecodeError:
      pass
    return popped

  # Useful when dealing with an unknown number of compound tags back to back.
  def finished(self):
    return self.idx >= len(self.data)

  def __repr__(self):
    return str(self.data[self.idx:])

# Allows for easy sequential writing of binary data.
class DataWriter:
  def __init__(self):
    self.data = []

  def put(self, key, *data):
    key = "<{}".format(key)
    self.data.append(struct.pack(key, *data))

  def putString(self, string):
    if not isinstance(string, bytes):
      string = string.encode("utf-8")
    self.put("h", len(string))
    self.data.append(struct.pack("<{}s".format(len(string)), string))

  def get(self):
    return b"".join(self.data)

  def __repr__(self):
    return str(self.data)

# Generic base tag, calls self.decode with binary data to fill in payload.
class TAG:
  ID = None
  def __init__(self, name, data):
    self.name = name
    if isinstance(data, DataReader):
      self.payload = self.decode(data)
    else:
      self.payload = data

  def decode(self, dataReader):
    raise NotImplementedError("Decode method not overridden by subclass.")

  def encode(self, dataWriter):
    raise NotImplementedError("Encode method not overridden by subclass.")

  def __getitem__(self, name):
    for item in self.payload:
      if item.name == name:
        return item
    raise KeyError("{} not found in {}".format(name, self.payload))

  def __getattr__(self, name):
    return self.__getitem__(name)

  def __eq__(self, other):
    return self.name == other.name and self.payload == other.payload and self.ID == other.ID

  def __repr__(self):
    return "{}-{}:{}".format(self.__class__.__name__, self.name, self.payload)

def TAG_Generator(ID, fmt, name):
  def _decode(self, dataReader):
    return dataReader.pop(fmt)
  def _encode(self, dataWriter):
    return dataWriter.put(fmt, self.payload)
  return type("TAG_{}".format(name), (TAG,), {"ID": ID, "decode": _decode, "encode": _encode})

tags = [] # Need to pre define tags for the later classes.

TAG_Byte = TAG_Generator(1, "B", "Byte")
TAG_Short = TAG_Generator(2, "h", "Short")
TAG_Int = TAG_Generator(3, "i", "Int")
TAG_Long = TAG_Generator(4, "q", "Long")
TAG_Float = TAG_Generator(5, "f", "Float")
TAG_Double = TAG_Generator(6, "d", "Double")

# Similar to TAG_List, except the type of tag is not specified, as we know it is a byte.
class TAG_Byte_Array(TAG):
  ID = 7
  def decode(self, dataReader):
    size = dataReader.pop("i")
    payload = []
    for i in range(size):
      payload.append(TAG_Byte(i, dataReader))
    return payload

  def encode(self, dataWriter):
    dataWriter.put("i", len(self.payload)) # Size
    for item in self.payload:
      item.encode(dataWriter)

class TAG_String(TAG):
  ID = 8
  def decode(self, dataReader):
    return dataReader.popString()
  def encode(self, dataWriter):
    dataWriter.putString(self.payload)

# Basically a TAG_Compound, but the items don't have names, and instead are named integer indexes.
#  This allows for a generic __getitem__ function in the TAG class.
class TAG_List(TAG):
  ID = 9
  def decode(self, dataReader):
    self.itemID = dataReader.pop("b")
    size = dataReader.pop("i")
    payload = []
    for i in range(size):
      payload.append(tags[self.itemID](i, dataReader))
    return payload

  def encode(self, dataWriter):
    if self.payload == []: # We don't know the dataWriter type.
      dataWriter.put("b", 0) # Default to TAG_End
    else:
      dataWriter.put("b", tags.index(type(self.payload[0])))
    dataWriter.put("i", len(self.payload))
    for item in self.payload:
      item.encode(dataWriter)

  def add(self, tag):
    self.payload.append(tag)

# Stores some number of complete tags, followed by a TAG_End
class TAG_Compound(TAG):
  ID = 10
  def decode(self, dataReader):
    payload = []
    tagID = dataReader.pop("b")
    while tagID != 0:
      if tags[tagID] is not None:
        name = dataReader.popString()
        payload.append(tags[tagID](name, dataReader))
      else:
        raise NotImplementedError("Tag {} not implemented.".format(tagID))
      tagID = dataReader.pop("b")
    return payload

  def encode(self, dataWriter):
    for item in self.payload:
      dataWriter.put("b", item.ID)
      dataWriter.putString(item.name)
      item.encode(dataWriter)
    dataWriter.put("b", 0)

  def add(self, tag):
    self.payload.append(tag)

  def pop(self, name):
    for i in range(len(self.payload)):
      if self.payload[i].name == name:
        return self.payload.pop(i)
    return None

  def __contains__(self, name):
    for item in self.payload:
      if item.name == name:
        return True
    return False

# Similar to TAG_List, except the type of tag is not specified, as we know it is an int.
class TAG_Int_Array(TAG):
  ID = 7
  def decode(self, dataReader):
    size = dataReader.pop("i")
    payload = []
    for i in range(size):
      payload.append(TAG_Int(i, dataReader))
    return payload

  def encode(self, dataWriter):
    dataWriter.put("i", len(self.payload)) # Size
    for item in self.payload:
      item.encode(dataWriter)

# Similar to TAG_List, except the type of tag is not specified, as we know it is a long.
class TAG_Long_Array(TAG):
  ID = 7
  def decode(self, dataReader):
    size = dataReader.pop("i")
    payload = []
    for i in range(size):
      payload.append(TAG_Long(i, dataReader))
    return payload

  def encode(self, dataWriter):
    dataWriter.put("i", len(self.payload)) # Size
    for item in self.payload:
      item.encode(dataWriter)

tags = [None,
        TAG_Byte,
        TAG_Short,
        TAG_Int,
        TAG_Long,
        TAG_Float,
        TAG_Double,
        TAG_Byte_Array,
        TAG_String,
        TAG_List,
        TAG_Compound,
        TAG_Int_Array,
        TAG_Long_Array]

def decode(dataReader):
  tagID = dataReader.pop("b")
  if tags[tagID] is not None:
    name = dataReader.popString()
    return tags[tagID](name, dataReader)
  raise NotImplementedError("Tag {} not implemented.".format(tagID))

def encode(toEncode, dataWriter=None):
  new = not dataWriter
  dataWriter = dataWriter or DataWriter()
  dataWriter.put("b", toEncode.ID)
  dataWriter.putString(toEncode.name)
  toEncode.encode(dataWriter)
  if new:
    return dataWriter.get()
