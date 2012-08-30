#ifndef __OIL_H_INCLUDED__
#define __OIL_H_INCLUDED__

#include <stdlib.h>

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
    int (*save)(OILImage *im, OILFile *file);
    OILImage *(*load)(OILFile *file);
} OILFormat;

OILImage *oil_image_new(unsigned int width, unsigned int height);
void oil_image_free(OILImage *im);
OILImage *oil_image_load(const char *path);
OILImage *oil_image_load_ex(OILFile *file);
int oil_image_save(OILImage *im, const char *path);
int oil_image_save_ex(OILImage *im, OILFile *file);
void oil_image_get_size(OILImage *im, unsigned int *width, unsigned int *height);
const OILPixel *oil_image_get_data(OILImage *im);
OILPixel *oil_image_lock(OILImage *im);
void oil_image_unlock(OILImage *im);


#endif /* __OIL_H_INCLUDED__ */
