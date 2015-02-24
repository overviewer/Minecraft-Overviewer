from libc.string cimport memcpy
from libc.stdlib cimport malloc, free
from cpython cimport bytes

cdef extern from "oil.h":
    unsigned int OIL_BACKEND_CPU
    unsigned int OIL_BACKEND_MAX
    
    bint oil_backend_set(int backend)
    
    ctypedef struct OILMatrix:
        float data[4][4]
    void oil_matrix_set_identity(OILMatrix *matrix)
    void oil_matrix_set_data(OILMatrix *matrix, const float *data)
    void oil_matrix_copy(OILMatrix *dest, const OILMatrix *src)
    bint oil_matrix_is_identity(const OILMatrix *matrix)
    bint oil_matrix_is_zero(const OILMatrix *matrix)
    void oil_matrix_transform(const OILMatrix *matrix, float *x, float *y, float *z)
    void oil_matrix_add(OILMatrix *result, const OILMatrix *a, const OILMatrix *b)
    void oil_matrix_subtract(OILMatrix *result, const OILMatrix *a, const OILMatrix *b)
    void oil_matrix_multiply(OILMatrix *result, const OILMatrix *a, const OILMatrix *b)
    void oil_matrix_negate(OILMatrix *result, const OILMatrix *matrix)
    bint oil_matrix_invert(OILMatrix *result, const OILMatrix* matrix)
    void oil_matrix_translate(OILMatrix *matrix, float x, float y, float z)
    void oil_matrix_scale(OILMatrix *matrix, float x, float y, float z)
    void oil_matrix_rotate(OILMatrix *matrix, float x, float y, float z)
    void oil_matrix_orthographic(OILMatrix *matrix, float x1, float x2, float y1, float y2, float z1, float z2)
    
    ctypedef struct OILPixel:
        unsigned char r, g, b, a
    
    ctypedef struct OILImage:
        pass
    
    ctypedef struct OILFile:
        void *file;
        size_t (*read)(void *file, void *data, size_t length)
        size_t (*write)(void *file, void *data, size_t length)
        void (*flush)(void *file)
    
    unsigned int OIL_FORMAT_PNG
    unsigned int OIL_FORMAT_MAX
    
    ctypedef struct OILFormatOptions:
        int indexed
        unsigned int palette_size
    
    ctypedef struct OILVertex:
        float x, y, z
        float s, t
        OILPixel color
    
    ctypedef enum OILTriangleFlags:
        OIL_DEPTH_TEST
    
    OILImage *oil_image_new(unsigned int width, unsigned int height)
    void oil_image_free(OILImage *im)
    OILImage *oil_image_load(const char *path)
    OILImage *oil_image_load_ex(OILFile *file)
    int oil_image_save(OILImage *im, const char *path, int format, OILFormatOptions *opts)
    int oil_image_save_ex(OILImage *im, OILFile *file, int format, OILFormatOptions *opts)
    void oil_image_get_size(OILImage *im, unsigned int *width, unsigned int *height)
    const OILPixel *oil_image_get_data(OILImage *im)
    const OILPixel *oil_image_get_pixel(OILImage *im, unsigned int x, unsigned int y)
    OILPixel *oil_image_lock(OILImage *im)
    void oil_image_unlock(OILImage *im)

    int oil_image_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize)
    void oil_image_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags)
    int oil_image_resize_half(OILImage *im, OILImage *src)
    void oil_image_clear(OILImage *im)

BACKEND_CPU = OIL_BACKEND_CPU
FORMAT_PNG = OIL_FORMAT_PNG
DEPTH_TEST = OIL_DEPTH_TEST

def backend_set(unsigned int backend):
    if not backend < OIL_BACKEND_MAX:
        raise ValueError("invalid backend")
    if not oil_backend_set(backend):
        raise RuntimeError("could not set backend")

cdef class Matrix:
    """Encapsulates matrix data and operations."""
    cdef OILMatrix _m
    
    def __init__(self, data=None):
        if data:
            self.data = data
        else:
            oil_matrix_set_identity(&self._m)
    
    property data:
        """A nested tuple of matrix data."""
        def __get__(self):
            tuples = []
            for i in range(4):
                tuples.append((self._m.data[i][0], self._m.data[i][1], self._m.data[i][2], self._m.data[i][3]))
            return tuple(tuples)
            
        def __set__(self, other):
            if isinstance(other, Matrix):
                memcpy(self._m.data, (<Matrix>other)._m.data, 16 * sizeof(float))
                return
                
            cdef float data[4][4];
            if len(other) != 4:
                raise ValueError("matrix data is not a 4x4 sequence of sequences")
            for i, seq in enumerate(other):
                if len(seq) != 4:
                    raise ValueError("matrix data is not a 4x4 sequence of sequences")
                for j, v in enumerate(seq):
                    data[i][j] = v
                
            oil_matrix_set_data(&self._m, <float*>data)
        
        def __del__(self):
            oil_matrix_set_identity(&self._m)

    def get_data(self):
        """Returns a nested tuple of matrix data."""
        return self.data
    
    def transform(self, float x, float y, float z):
        """Transform 3 coordinates."""
        oil_matrix_transform(&self._m, &x, &y, &z)
        return (x, y, z)
    
    def __str__(self):
        return repr(self.data)
    
    def __repr__(self):
        return "Matrix({0})".format(self.__str__())
    
    @staticmethod
    cdef _binop(pa, pb, void (*op)(OILMatrix*, const OILMatrix*, const OILMatrix*), bint inplace):
        cdef Matrix result, a, b
        if not (isinstance(pa, Matrix) and isinstance(pb, Matrix)):
            return NotImplemented
        a = pa
        b = pb
        
        if not inplace:
            result = Matrix.__new__(Matrix)
            op(&result._m, &a._m, &b._m)
            return result
        
        op(&a._m, &a._m, &b._m)
        return a
    
    def __add__(a, b):
        return Matrix._binop(a, b, &oil_matrix_add, False)
    def __iadd__(self, x):
        return Matrix._binop(self, x, &oil_matrix_add, True)
    
    def __sub__(a, b):
        return Matrix._binop(a, b, &oil_matrix_subtract, False)
    def __isub__(self, x):
        return Matrix._binop(self, x, &oil_matrix_subtract, True)
    
    def __mul__(a, b):
        return Matrix._binop(a, b, &oil_matrix_multiply, False)
    def __imul__(self, x):
        return Matrix._binop(self, x, &oil_matrix_multiply, True)
    
    def __neg__(self):
        cdef Matrix result = Matrix.__new__(Matrix)
        oil_matrix_negate(&result._m, &self._m)
        return result
    def __pos__(self):
        return self
    
    def __nonzero__(self):
        return not oil_matrix_is_zero(&self._m)
    
    property inverse:
        """Return the inverse of this matrix."""
        def __get__(self):
            cdef Matrix result = Matrix.__new__(Matrix)
            if not oil_matrix_invert(&result._m, &self._m):
                raise ValueError("cannot invert matrix")
            return result
    
    def get_inverse(self):
        """Returns the inverse of the matrix."""
        return self.inverse
    
    def invert(self):
        """Invert the matrix in-place."""
        if not oil_matrix_invert(&self._m, &self._m):
            raise ValueError("cannot invert matrix")
    
    def translate(self, float x, float y, float z):
        """Multiply on a translation matrix."""
        oil_matrix_translate(&self._m, x, y, z)
        return self
    
    def scale(self, float x, float y, float z):
        """Multiply on a scaling matrix."""
        oil_matrix_scale(&self._m, x, y, z)
        return self
    
    def rotate(self, float x, float y, float z):
        """Multiply on a rotation matrix."""
        oil_matrix_rotate(&self._m, x, y, z)
        return self
    
    def orthographic(self, float x1, float x2, float y1, float y2, float z1, float z2):
        """Multiply on an orthographic matrix."""
        oil_matrix_orthographic(&self._m, x1, x2, y1, y2, z1, z2)
        return self

# support for turning a python file-like object into OILFile
cdef size_t oil_python_read(void *f, void *data, size_t length):
    cdef object file = <object>f
    cdef bytes buf
    cdef size_t buflength
    buf = file.read(length)
    buflength = len(buf)
    memcpy(data, <char*>buf, buflength)
    return buflength

cdef size_t oil_python_write(void *f, void *data, size_t length):
    cdef object file = <object>f
    file.write(bytes.PyBytes_FromStringAndSize(<char*>data, length))
    return length

cdef void oil_python_flush(void* f):
    cdef object file = <object>f
    file.flush()

cdef class Image:
    """Encapsulates image data and image operations."""    
    cdef OILImage *_im
    
    def __cinit__(self):
        pass
    
    def __init__(self, unsigned int width, unsigned int height):
        if width == 0 or height == 0:
            raise ValueError("cannot create image with 0 size")
        self._im = oil_image_new(width, height)
        if not self._im:
            raise MemoryError
    
    def __dealloc__(self):
        if self._im:
            oil_image_free(self._im)
            self._im = NULL
    
    @classmethod
    def load(cls, path_or_file):
        """Load the given path name into an Image object."""
        cdef Image self = cls.__new__(cls)
        cdef OILFile file
        
        if isinstance(path_or_file, str):
            self._im = oil_image_load(path_or_file)
        else:
            file.file = <void*>path_or_file
            file.read = &oil_python_read
            file.write = &oil_python_write
            file.flush = &oil_python_flush
            self._im = oil_image_load_ex(&file)
        
        if not self._im:
            raise IOError("cannot load image") # FIXME propogate file err
        
        return self
    
    def save(self, dest, int format, bint indexed=0, unsigned int palette_size=0):
        """Save the Image object to a file."""
        cdef OILFile file
        cdef OILFormatOptions opts
        cdef bint save_success
        
        if not format < OIL_FORMAT_MAX:
            raise ValueError("invalid format")

        opts.indexed = indexed
        opts.palette_size = palette_size
        
        if isinstance(dest, str):
            save_success = oil_image_save(self._im, dest, format, &opts)
        else:
            file.file = <void*>dest
            file.read = &oil_python_read
            file.write = &oil_python_write
            file.flush = &oil_python_flush
            save_success = oil_image_save_ex(self._im, &file, format, &opts)
        
        if not save_success:
            raise IOError("cannot save image") # FIXME propogate file err
    
    property size:
        """Return a (width, height) tuple."""
        def __get__(self):
            cdef unsigned int width, height;
            oil_image_get_size(self._im, &width, &height)
            return (width, height)
    
    def get_size(self):
        """Return a (width, height) tuple."""
        return self.size
    
    def composite(self, Image src, unsigned char alpha=255, int dx=0, int dy=0, unsigned int sx=0, unsigned int sy=0, unsigned int xsize=0, unsigned int ysize=0):
        """Composite another image on top of this one."""
        if not oil_image_composite(self._im, src._im, alpha, dx, dy, sx, sy, xsize, ysize):
            raise RuntimeError("cannot composite image")
    
    def draw_triangles(self, Matrix matrix, Image tex, vertices, indices, OILTriangleFlags flags):
        """Draw 3D triangles on top of the image."""
        cdef unsigned int vertices_length = len(vertices)
        cdef unsigned int indices_length = len(indices)
        cdef OILVertex *c_vertices
        cdef unsigned int *c_indices
        
        try:
            c_vertices = <OILVertex*>malloc(sizeof(OILVertex) * vertices_length)
            c_indices = <unsigned int*>malloc(sizeof(unsigned int) * indices_length)
            if not (c_vertices and c_indices):
                raise MemoryError
            
            for i, ((x, y, z), (s, t), (r, g, b, a)) in enumerate(vertices):
                c_vertices[i].x = x
                c_vertices[i].y = y
                c_vertices[i].z = z
                c_vertices[i].s = s
                c_vertices[i].t = t
                c_vertices[i].color.r = r
                c_vertices[i].color.g = g
                c_vertices[i].color.b = b
                c_vertices[i].color.a = a
            
            for i, ind in enumerate(indices):
                c_indices[i] = ind
            
            oil_image_draw_triangles(self._im, &matrix._m, tex._im, c_vertices, vertices_length, c_indices, indices_length, flags)
            
        finally:
            if c_vertices:
                free(c_vertices)
            if c_indices:
                free(c_indices)
    
    def resize_half(self, Image src):
        """Shrink the given image by half and copy onto self."""
        if not oil_image_resize_half(self._im, src._im):
            raise RuntimeError("cannot resize image in half (likely a size mismatch)")
    
    def clear(self):
        """Clear the image."""
        oil_image_clear(self._im)

