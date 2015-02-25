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
