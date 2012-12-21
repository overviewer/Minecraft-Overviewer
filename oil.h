#ifndef __OIL_H_INCLUDED__
#define __OIL_H_INCLUDED__

#include <stdlib.h>

#define OIL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define OIL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define OIL_CLAMP(a, min, max) OIL_MIN((max), OIL_MAX((min), (a)))

#ifdef _MSC_VER
#  ifndef inline
#    define inline __inline
#  endif /* inline */
#endif /* _MSC_VER */

typedef enum {
    OIL_BACKEND_CPU,
    OIL_BACKEND_DEBUG,
    
    /* used to detect invalid backends */
    OIL_BACKEND_MAX
} OILBackendName;

void oil_backend_set(OILBackendName backend);

typedef struct {
    float data[4][4];
} OILMatrix;

void oil_matrix_set_identity(OILMatrix *matrix);
/* row-major */
void oil_matrix_set_data(OILMatrix *matrix, const float *data);
void oil_matrix_copy(OILMatrix *dest, const OILMatrix *src);
int oil_matrix_is_identity(const OILMatrix *matrix);
int oil_matrix_is_zero(const OILMatrix *matrix);
void oil_matrix_transform(const OILMatrix *matrix, float *x, float *y, float *z);
/* result == a is allowed, result == b is not */
void oil_matrix_add(OILMatrix *result, const OILMatrix *a, const OILMatrix *b);
void oil_matrix_subtract(OILMatrix *result, const OILMatrix *a, const OILMatrix *b);
void oil_matrix_multiply(OILMatrix *result, const OILMatrix *a, const OILMatrix *b);
/* matrix == result allowed */
void oil_matrix_negate(OILMatrix *result, const OILMatrix *matrix);
/* returns 0 on failure */
int oil_matrix_invert(OILMatrix *result, const OILMatrix* matrix);
/* all the following construct a type of matrix and multiply it on the right */
void oil_matrix_translate(OILMatrix *matrix, float x, float y, float z);
void oil_matrix_scale(OILMatrix *matrix, float x, float y, float z);
/* rotate around axis given, by magnitude of axis (as radians) */
void oil_matrix_rotate(OILMatrix *matrix, float x, float y, float z);
void oil_matrix_orthographic(OILMatrix *matrix, float x1, float x2, float y1, float y2, float z1, float z2);

typedef struct {
    unsigned char r, g, b, a;
} OILPixel;

typedef struct _OILImage OILImage;

typedef struct {
    void *file;
    size_t (*read)(void *file, void *data, size_t length);
    size_t (*write)(void *file, void *data, size_t length);
    void (*flush)(void *file);
} OILFile;

typedef struct {
    int indexed;
    unsigned int palette_size;
} OILFormatOptions;

typedef struct {
    float x, y, z;
    float s, t;
    OILPixel color;
} OILVertex;

typedef enum {
    OIL_DEPTH_TEST = 0x1,
} OILTriangleFlags;

OILImage *oil_image_new(unsigned int width, unsigned int height);
void oil_image_free(OILImage *im);
OILImage *oil_image_load(const char *path);
OILImage *oil_image_load_ex(OILFile *file);
int oil_image_save(OILImage *im, const char *path, OILFormatOptions *opts);
int oil_image_save_ex(OILImage *im, OILFile *file, OILFormatOptions *opts);
void oil_image_get_size(OILImage *im, unsigned int *width, unsigned int *height);
const OILPixel *oil_image_get_data(OILImage *im);
OILPixel *oil_image_lock(OILImage *im);
void oil_image_unlock(OILImage *im);

int oil_image_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize);
void oil_image_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags);

#endif /* __OIL_H_INCLUDED__ */
