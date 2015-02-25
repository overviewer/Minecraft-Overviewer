from coil cimport *
from libc.stdlib cimport malloc, calloc, free

cdef extern from "buffer.h":
    ctypedef struct Buffer:
        pass
    
    void buffer_init(Buffer *buffer, unsigned int element_size, unsigned int initial_length=0)
    void buffer_free(Buffer *buffer)
    void buffer_reserve(Buffer *buffer, unsigned int length)
    void buffer_append(Buffer *buffer, const void *newdata, unsigned int newdata_length)
    void buffer_clear(Buffer *buffer)

cdef extern from "indexing.h":
    unsigned char bytes1b(bytes b, unsigned int i)
    unsigned char bytes1n(bytes b, unsigned int i)
    unsigned char bytes2b(bytes b, unsigned int x, unsigned int z)
    unsigned char bytes2n(bytes b, unsigned int x, unsigned int z)
    unsigned char bytes3b(bytes b, unsigned int x, unsigned int y, unsigned int z)
    unsigned char bytes3n(bytes b, unsigned int x, unsigned int y, unsigned int z)

DEF SECTIONS_PER_CHUNK = 16
DEF BLOCK_BUFFER_SIZE = 32

ctypedef enum FaceType:
    FACE_TYPE_PX = (1<<0)
    FACE_TYPE_NX = (1<<1)
    FACE_TYPE_PY = (1<<2)
    FACE_TYPE_NY = (1<<3)
    FACE_TYPE_PZ = (1<<4)
    FACE_TYPE_NZ = (1<<5)
    FACE_TYPE_DIRECTION_MASK = (1<<6)-1,
    # Lower 6 bits are the face directions. upper bits are other flags.
    # BIOME_COLOED tells the renderer to apply biome coloring to this face
    FACE_TYPE_BIOME_COLORED = (1<<6)

PX = FACE_TYPE_PX
NX = FACE_TYPE_NX
PY = FACE_TYPE_PY
NY = FACE_TYPE_NY
PZ = FACE_TYPE_PZ
NZ = FACE_TYPE_NZ
DIRECTION_MASK = FACE_TYPE_DIRECTION_MASK
BIOME_COLORED = FACE_BIOME_COLORED

