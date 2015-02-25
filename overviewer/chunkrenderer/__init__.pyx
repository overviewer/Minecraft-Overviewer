from overviewer.oil.cdefs cimport *
from overviewer.oil cimport *
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
BIOME_COLORED = FACE_TYPE_BIOME_COLORED

ctypedef struct FaceDef:
    # these are the points of the triangle
    unsigned int p[3]
    # see the FaceType enum
    FaceType face_type
    # texture to use
    OILImage *tex

cdef class BlockModel:
    """This represents a mesh, or a description of how to draw a model. It
    includes the vertices, textures, and face definitions.
    The constructor expects the following arguments:
    
     * vertices: a list of (coordinate, texcoords, color) tuples,
       where coordinate is a 3-tuple of numbers, texcoords is a
       2-tuple, and color is a 4-tuple of integers between 0 and 255.
     * faces: a list of ((pt, ...), type, texture) where pt is at least 3
       integers corresponding to indexes into the vertices array. type is a
       bitmask made from chunkrenderer.FACE_TYPE_* constants. texture is an
       oil image.
     
    """
    
    cdef bint known
    cdef unsigned int vertices_length
    cdef OILVertex *vertices
    cdef unsigned int faces_length
    cdef FaceDef *faces
    cdef set textures
    
    def __cinit__(self):
        self.known = False
        self.vertices = NULL
        self.faces = NULL
    
    def __dealloc__(self):
        if self.vertices:
            free(<void*>self.vertices)
            self.vertices = NULL
        if self.faces:
            free(<void*>self.faces)
            self.faces = NULL
    
    def __init__(self, vertices, faces):
        self.textures = set()
        self.vertices_length = len(vertices)
        self.faces_length = 0
        cdef Image tex
        for pts, typ, tex in faces:
            if len(pts) < 3:
                raise ValueError("face must have at least 3 points")
            self.faces_length += len(pts) - 2
        
        self.vertices = <OILVertex*>malloc(self.vertices_length * sizeof(OILVertex))
        self.faces = <FaceDef*>malloc(self.faces_length * sizeof(FaceDef))
        if not (self.vertices and self.faces):
            raise MemoryError
        
        for i, ((x, y, z), (s, t), (r, g, b, a)) in enumerate(vertices):
            self.vertices[i].x = x
            self.vertices[i].y = y
            self.vertices[i].z = z
            self.vertices[i].s = s
            self.vertices[i].t = t
            self.vertices[i].color.r = r
            self.vertices[i].color.g = g
            self.vertices[i].color.b = b
            self.vertices[i].color.a = a
        
        cdef unsigned int j = 0
        for pts, typ, pytex in faces:
            tex = pytex
            self.textures.add(tex)
            first = pts[0]
            a = pts[1:]
            b = pts[2:]
            for ix, iy in zip(a, b):
                self.faces[j].p[0] = first
                self.faces[j].p[1] = ix
                self.faces[j].p[2] = iy
                self.faces[j].face_type = typ
                self.faces[j].tex = tex._im
                j += 1
        
        self.known = True

cdef inline void c_render_block(BlockModel block, OILMatrix *mat, OILImage *im):
    for i in range(block.faces_length):
        oil_image_draw_triangles(im, mat, block.faces[i].tex, block.vertices, block.vertices_length, block.faces[i].p, 3, OIL_DEPTH_TEST)

def render_block(BlockModel block, Matrix matrix, Image im):
    return c_render_block(block, &matrix._m, im._im)
