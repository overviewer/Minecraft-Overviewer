#include "oil-private.h"

#include <stdlib.h>
#include <stdio.h>

/*
 * file IO implementation for plain old FILEs
 */

static size_t oil_image_read(void *file, void *data, size_t length) {
    return fread(data, 1, length, file);
}

static size_t oil_image_write(void *file, void *data, size_t length) {
    return fwrite(data, 1, length, file);
}

static void oil_image_flush(void *file) {
    fflush((FILE *)file);
}

OILImage *oil_image_new(unsigned int width, unsigned int height) {
    OILImage *im;
    if (width == 0 || height == 0)
        return NULL;
    
    im = malloc(sizeof(OILImage));
    if (im) {
        im->width = width;
        im->height = height;
        im->locked = 0;
        im->data = malloc(sizeof(OILPixel) * width * height);
        if (im->data == NULL) {
            free(im);
            return NULL;
        }
    }
    return im;
}

void oil_image_free(OILImage *im) {
    if (im) {
        if (im->data)
            free(im->data);
        free(im);
    }
}

OILImage *oil_image_load(const char *path) {
    FILE *fp;
    OILImage *ret;
    OILFile file;
    if (!path)
        return NULL;
    
    fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    
    file.file = fp;
    file.read = oil_image_read;
    file.write = oil_image_write;
    file.flush = oil_image_flush;
    
    ret = oil_image_load_ex(&file);
    fclose(fp);
    return ret;
}

OILImage *oil_image_load_ex(OILFile *file) {
    return oil_formats[0]->load(file);
}

int oil_image_save(OILImage *im, const char *path) {
    FILE *fp;
    int ret;
    OILFile file;
    if (!path || !im)
        return 0;
    
    fp = fopen(path, "wb");
    if (!fp)
        return 0;
    
    file.file = fp;
    file.read = oil_image_read;
    file.write = oil_image_write;
    file.flush = oil_image_flush;
    
    ret = oil_image_save_ex(im, &file);
    fclose(fp);
    return ret;
}

int oil_image_save_ex(OILImage *im, OILFile *file) {
    return oil_formats[0]->save(im, file);
}

void oil_image_get_size(OILImage *im, unsigned int *width, unsigned int *height) {
    if (im) {
        if (width)
            *width = im->width;
        if (height)
            *height = im->height;
    } else {
        if (width)
            *width = 0;
        if (height)
            *height = 0;
    }
}

const OILPixel *oil_image_get_data(OILImage *im) {
    if (im) {
        return im->data;
    }
    return NULL;
}

OILPixel *oil_image_lock(OILImage *im) {
    if (im) {
        if (im->locked)
            return NULL;
        im->locked = 1;
        return im->data;
    }
    return NULL;
}

void oil_image_unlock(OILImage *im) {
    if (im) {
        im->locked = 0;
    }
}
