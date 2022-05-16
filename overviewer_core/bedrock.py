# Interprets the minecraft bedrock world format.

import struct
import os.path
import numpy as np
from . import leveldb as ldb
from . import nbt2

# Handles chunk loading and mapping blocks to chunks.
class World:
  def __init__(self, path):
    self.path = os.path.join(path, "db")
    self.db = ldb.open(self.path)
    self.chunks = {}

  # Enable use in a with statement.
  def __enter__(self):
    self.db = ldb.open(self.path)
    return self

  def __exit__(self, exceptionType, exception, tb):
    if exceptionType is None:
      self.save()
    ldb.close(self.db)
    return False

  def getChunk(self, x, z):
    chunk = self.chunks.get((x, z), None)
    if chunk is None:
      chunk = Chunk(self.db, x, z)
      self.chunks[(x, z)] = chunk
    return chunk

  def getBlock(self, x, y, z, layer=0):
    cx = x // 16
    x %= 16
    cz = z // 16
    z %= 16
    chunk = self.getChunk(cx, cz)
    return chunk.getBlock(x, y, z, layer)

  def setBlock(self, x, y, z, block, layer=0):
    cx = x // 16
    x %= 16
    cz = z // 16
    z %= 16
    chunk = self.getChunk(cx, cz)
    return chunk.setBlock(x, y, z, block, layer)

  def save(self):
    for chunk in self.chunks.values():
      chunk.save(self.db)

  def iterKeys(self, start=None, end=None):
    yield from ldb.iterate(self.db, start, end)

  def iterChunks(self, start=None, end=None):
    for k, _ in ldb.iterate(self.db):
      if len(k) == 9 and k.endswith((b"v", b",")):
        x, z = struct.unpack("<ii", k[:8])
        if start and (x < start[0] or x >= end[0]):
          continue
        if end and (z < start[1] or z >= end[1]):
          continue
        try:
          yield self.getChunk(x, z)
        except Exception as e:
          print("Error: Couldn't load chunk at {} {}: {}".format(x, z, e))

# Handles biomes and tile entities. Maps blocks to subchunks.
class Chunk:
  def __init__(self, db, x, z):
    self.x = x
    self.z = z
    # Leveldb chunks are stored in a number of keys with the same prefix.
    self.keyBase = struct.pack("<ii", self.x, self.z)

    self.version = self._loadVersion(db)
    self.cavesAndCliffs = self.version >= 25
    if not self.cavesAndCliffs:
      self.hMap, self.biomes = self._load2D(db)
    else:
      self.hMap, self.biomes = None, None

    self.subchunks = []
    for i in range(24 if self.cavesAndCliffs else 16):
      try:
        self.subchunks.append(SubChunk(db, self.x, self.z, i)) #Pass off processing to the subchunk class
      #Supposedly if a subchunk exists then all the subchunks below it exist. This is not the case.
      except NotFoundError:
        self.subchunks.append(None)

    self._loadTileEntities(db)
    self.entities = self._loadEntities(db)

  # Version is simply a stored value.
  def _loadVersion(self, db):
    try:
      try:
        version = ldb.get(db, self.keyBase + b",")
      except KeyError:
        version = ldb.get(db, self.keyBase + b"v")
      version = struct.unpack("<B", version)[0]
      if version not in [10, 13, 14, 15, 18, 19, 21, 22, 25]:
        raise NotImplementedError("Unexpected chunk version {} at chunk {} {}.".format(version, self.x, self.z))
    except KeyError:
      raise KeyError("Chunk at {}, {} does not exist.".format(self.x, self.z))
    return version

  # Load heightmap (seemingly useless) and biome info
  def _load2D(self, db):
    data = ldb.get(db, self.keyBase + b'-')
    heightMap = struct.unpack("<" + "H" * 16 * 16, data[:2 * 16 * 16])
    biomes = struct.unpack("B" * 16 * 16, data[2 * 16 * 16:])
    return heightMap, biomes

  # Tile entities are stored as a bunch of NBT compound tags end to end.
  def _loadTileEntities(self, db):
    try:
      data = ldb.get(db, self.keyBase + b"1")
    except KeyError:
      return
    data = nbt2.DataReader(data)
    while not data.finished():
      nbtData = nbt2.decode(data)
      x = nbtData.pop("x").payload # We add back theses with the correct value on save, they are important.
      y = nbtData.pop("y").payload
      z = nbtData.pop("z").payload
      block = self.getBlock(x % 16, y, z % 16)
      if not block:
        print("Warning: Cannot apply nbt to block at {} {} {} since it does not exist.".format(x, y, z))
        continue
      block.nbt = nbtData

  def _loadEntities(self, db):
    try:
      data = ldb.get(db, self.keyBase + b"2")
    except KeyError:
      return []
    data = nbt2.DataReader(data)
    entities = []
    while not data.finished():
      entities.append(nbt2.decode(data))
    return entities

  def getBlock(self, x, y, z, layer=0):
    if self.cavesAndCliffs:
      y += 64
    if y // 16 + 1 > len(self.subchunks) or self.subchunks[y // 16] is None:
      return None
    return self.subchunks[y // 16].getBlock(x, y % 16, z, layer)

  def setBlock(self, x, y, z, block, layer=0):
    if self.cavesAndCliffs:
      y += 64
    while y // 16 + 1 > len(self.subchunks):
      self.subchunks.append(SubChunk.empty(self.x, self.z, len(self.subchunks)))
    if self.subchunks[y // 16] is None:
      self.subchunks[y // 16] = SubChunk.empty(self.x, self.z, y // 16)
    self.subchunks[y // 16].setBlock(x, y % 16, z, block, layer)

  def save(self, db):
    version = struct.pack("<B", self.version)
    ldb.put(db, self.keyBase + b",", version)
    if not self.cavesAndCliffs:
      self._save2D(db)
    for subchunk in self.subchunks:
      if subchunk is None:
        continue
      subchunk.save(db)
    self._saveTileEntities(db)
    self._saveEntities(db)

  def _save2D(self, db):
    data = struct.pack("<" + "H" * 16 * 16, *self.hMap)
    data += struct.pack("B" * 16 * 16, *self.biomes)
    ldb.put(db, self.keyBase + b'-', data)

  def _saveTileEntities(self, db):
    data = nbt2.DataWriter()
    for subchunk in self.subchunks:
      if subchunk is None:
        continue
      for x in range(16):
        for y in range(16):
          for z in range(16):
            block = subchunk.getBlock(x, y, z)
            if block.nbt is not None: # Add back the correct position.
              block.nbt.add(nbt2.TAG_Int("x", subchunk.x * 16 + x))
              block.nbt.add(nbt2.TAG_Int("y", subchunk.y * 16 + y))
              block.nbt.add(nbt2.TAG_Int("z", subchunk.z * 16 + z))
              nbt2.encode(block.nbt, data)
    ldb.put(db, self.keyBase + b"1", data.get())

  def _saveEntities(self, db):
    data = nbt2.DataWriter()
    for entity in self.entities:
      nbt2.encode(entity, data)
    ldb.put(db, self.keyBase + b"2", data.get())

  def __repr__(self):
    return "Chunk {} {}: {} subchunks".format(self.x, self.z, len(self.subchunks))

# Handles the blocks and block palette format.
class SubChunk:
  def __init__(self, db, x, z, y):
    self.dirty = False
    self.x = x
    self.z = z
    self.y = y
    if db is not None: # For creating subchunks, there will be no DB.
      # Subchunks are stored as base key + subchunk key `/` + subchunk id (y level // 16)
      key = struct.pack("<iicB", x, z, b'/', y)
      try:
        data = ldb.get(db, key)
      except KeyError:
        raise NotFoundError("Subchunk at {} {}/{} not found.".format(x, z, y))
      self.version, data = data[0], data[1:]
      if self.version not in [8, 9]:
        raise NotImplementedError("Unsupported subchunk version {} at {} {}/{}".format(self.version, x, z, y))
      numStorages, data = data[0], data[1:]

      if self.version == 9:
        self.y_db, data = data[0], data[1:]
      else:
        self.y_db = None

      self.blocks = []
      for i in range(numStorages):
        blocks, data = self._loadBlocks(data)
        palette, data = self._loadPalette(data)

        self.blocks.append(np.empty(4096, dtype=Block)) # Prepare with correct dtype
        for j, block in enumerate(blocks):
          block = palette[block]
          try: # 1.13 format
            #if block["version"].payload != 17629200:
            #  raise NotImplementedError("Unexpected block version {}".format(block["version"].payload))
            self.blocks[i][j] = Block(block["name"].payload, block["states"].payload) # .payload to get actual val
          except KeyError: # 1.12 format
            self.blocks[i][j] = Block(block["name"].payload, block["val"].payload) # .payload to get actual val
        self.blocks[i] = self.blocks[i].reshape(16, 16, 16).swapaxes(1, 2) # Y and Z saved in an inverted order

  # These arent actual blocks, just ids pointing to the palette.
  def _loadBlocks(self, data):
    #Ignore LSB of data (its a flag) and get compacting level
    bitsPerBlock, data = data[0] >> 1, data[1:]
    blocksPerWord = 32 // bitsPerBlock # Word = 4 bytes, basis of compacting.
    numWords = - (-4096 // blocksPerWord) # Ceiling divide is inverted floor divide

    blockWords, data = struct.unpack("<" + "I" * numWords, data[:4 * numWords]), data[4 * numWords:]
    blocks = np.empty(4096, dtype=np.uint32)
    for i, word in enumerate(blockWords):
      for j in range(blocksPerWord):
        block = word & ((1 << bitsPerBlock) - 1) # Mask out number of bits for one block
        word >>= bitsPerBlock # For next iteration
        if i * blocksPerWord + j < 4096: # Safety net for padding at end.
          blocks[i * blocksPerWord + j] = block
    return blocks, data

  # NBT encoded block names (with minecraft:) and data values.
  def _loadPalette(self, data):
    palletLen, data = struct.unpack("<I", data[:4])[0], data[4:]
    dr = nbt2.DataReader(data)
    palette = []
    for _ in range(palletLen):
      block = nbt2.decode(dr)
      palette.append(block)
    return palette, data[dr.idx:]

  def getBlock(self, x, y, z, layer=0):
    if layer >= len(self.blocks):
      raise KeyError("Subchunk {} {}/{} does not have a layer {}".format(self.x, self.z, self.y, layer))
    return self.blocks[layer][x, y, z]

  def setBlock(self, x, y, z, block, layer=0):
    if layer >= len(self.blocks):
      raise KeyError("Subchunk {} {}/{} does not have a layer {}".format(self.x, self.z, self.y, layer))
    self.blocks[layer][x, y, z] = block
    self.dirty = True

  def save(self, db, force=False):
    if self.dirty or force:
      data = struct.pack("<BB", self.version, len(self.blocks))
      for i in range(len(self.blocks)):
        palette, blockIDs = self._savePalette(i)
        data += self._saveBlocks(len(palette), blockIDs)
        data += struct.pack("<I", len(palette))
        for block in palette:
          data += nbt2.encode(block)

      if self.version == 9:
        data = struct.pack("B", self.y_db) + data

      key = struct.pack("<iicB", self.x, self.z, b'/', self.y)
      ldb.put(db, key, data)

  # Compact blockIDs bitwise. See _loadBlocks for details.
  def _saveBlocks(self, paletteSize, blockIDs):
    bitsPerBlock = max(int(np.ceil(np.log2(paletteSize))), 1)
    for bits in [1, 2, 3, 4, 5, 6, 8, 16]:
      if bits >= bitsPerBlock:
        bitsPerBlock = bits
        break
    else:
      raise NotImplementedError("Too many bits per block needed {} at {} {}/{}".format(bitsPerBlock, self.x, self.z, self.y))
    blocksPerWord = 32 // bitsPerBlock
    numWords = - (-4096 // blocksPerWord)
    data = struct.pack("<B", bitsPerBlock << 1)

    for i in range(numWords):
      word = 0
      for j in range(blocksPerWord - 1, -1, -1):
        if i * blocksPerWord + j < 4096:
          word <<= bitsPerBlock
          word |= blockIDs[i * blocksPerWord + j]
      data += struct.pack("<I", word)
    return data

  # Make a palette, and get the block ids at the same time
  def _savePalette(self, layer):
    blocks = self.blocks[layer].swapaxes(1, 2).reshape(4096) # Y and Z saved in a inverted order
    blockIDs = np.empty(4096, dtype=np.uint32)
    palette = []
    mapping = {}
    for i, block in enumerate(blocks):
      # Generate the palette nbt for the given block
      short = (block.name, str(block.properties))
      if short not in mapping:
        if isinstance(block.properties, int): # 1.12
          palette.append(nbt2.TAG_Compound("", [nbt2.TAG_String("name", block.name), nbt2.TAG_Short("val", block.properties)]))
        else: # 1.13
          palette.append(nbt2.TAG_Compound("", [
            nbt2.TAG_String("name", block.name),
            nbt2.TAG_Compound("states", block.properties),
            nbt2.TAG_Int("version", 17629200)
          ]))
        mapping[short] = len(palette) - 1
      blockIDs[i] = mapping[short]
    return palette, blockIDs

  @classmethod
  def empty(cls, x, z, y):
    subchunk = cls(None, x, z, y)
    subchunk.version = 8
    subchunk.blocks = [np.full((16, 16, 16), Block("minecraft:air"), dtype=Block)]
    return subchunk

# Generic block storage.
class Block:
  __slots__ = ["name", "properties", "nbt"]
  def __init__(self, name, properties=None, nbtData=None):
    self.name = name
    self.properties = properties or []
    self.nbt = nbtData

  def __eq__(self, other):
    if not isinstance(other, Block):
      return False
    return self.name == other.name and self.properties == other.properties and self.nbt == other.nbt

  def __repr__(self):
    return "{} {}".format(self.name, self.properties)

  def __hash__(self):
    return self.__repr__().__hash__()

# Handles NBT generation for command blocks.
class CommandBlock(Block):
  nameMap = {"I": "command_block", "C": "chain_command_block", "R": "repeating_command_block"}
  dMap = {"d": 0, "u": 1, "-z": 2, "+z": 3, "-x": 4, "+x": 5}
  def __init__(self, cmd="", hover="", block="I", d="u", cond=False, redstone=False, time=0, first=False):
    name = "minecraft:" + self.nameMap[block]
    dv = self.dMap[d]
    if cond:
      dv += 8
    nbtData = nbt2.TAG_Compound("", [])
    nbtData.add(nbt2.TAG_Byte("auto", int(not redstone)))
    nbtData.add(nbt2.TAG_String("Command", cmd))
    nbtData.add(nbt2.TAG_String("CustomName", hover))
    nbtData.add(nbt2.TAG_Byte("powered", int(block == "R" and not redstone)))
    if time == 0 and not first:
      nbtData.add(nbt2.TAG_Int("Version", 8))
    else:
      nbtData.add(nbt2.TAG_Int("Version", 9))
      nbtData.add(nbt2.TAG_Byte("ExecuteOnFirstTick", int(first)))
      nbtData.add(nbt2.TAG_Int("TickDelay", time))

    nbtData.add(nbt2.TAG_Byte("conditionMet", 0))
    nbtData.add(nbt2.TAG_String("id", "CommandBlock"))
    nbtData.add(nbt2.TAG_Byte("isMovable", 1))
    nbtData.add(nbt2.TAG_Int("LPCommandMode", 0)) # Not sure what these LPModes do. This works.
    nbtData.add(nbt2.TAG_Byte("LPConditionalMode", 0))
    nbtData.add(nbt2.TAG_Byte("LPRedstoneMode", 0))
    nbtData.add(nbt2.TAG_Long("LastExecution", 0))
    nbtData.add(nbt2.TAG_String("LastOutput", ""))
    nbtData.add(nbt2.TAG_List("LastOutputParams", []))
    nbtData.add(nbt2.TAG_Int("SuccessCount", 0))
    nbtData.add(nbt2.TAG_Byte("TrackOutput", 1))
    super().__init__(name, dv, nbtData)

class NotFoundError(Exception):
  pass
