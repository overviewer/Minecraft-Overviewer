#include "oil.h"
#include "oil-image-private.h"
#include "oil-format-private.h"

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
        im->data = calloc(width * height, sizeof(OILPixel));
        if (im->data == NULL) {
            free(im);
            return NULL;
        }
        
        im->backend = oil_backend;
    }
    
    im->backend->new(im);
    return im;
}

void oil_image_free(OILImage *im) {
    if (im) {
        im->backend->free(im);
        
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

int oil_image_save(OILImage *im, const char *path, OILFormatOptions *opts) {
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
    
    ret = oil_image_save_ex(im, &file, opts);
    fclose(fp);
    return ret;
}

int oil_image_save_ex(OILImage *im, OILFile *file, OILFormatOptions *opts) {
    return oil_formats[0]->save(im, file, opts);
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
        im->backend->load(im);
        return im->data;
    }
    return NULL;
}

OILPixel *oil_image_lock(OILImage *im) {
    if (im) {
        if (im->locked)
            return NULL;
        
        im->locked = 1;
        im->backend->load(im);
        return im->data;
    }
    return NULL;
}

void oil_image_unlock(OILImage *im) {
    if (im) {
        im->locked = 0;
        im->backend->save(im);
    }
}

int oil_image_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    /* duh, this is bad */
    if (im == NULL || src == NULL)
        return 0;
    
    /* firstly, are these images from the same backend? */
    if (im->backend != src->backend)
        return 0;
    
    /* if alpha == 0, there's nothing to do */
    if (alpha == 0)
        return 1;
    
    /* if dx, dy are out of bounds, we're already done */
    if (dx >= im->width || dy >= im->height)
        return 1;
    
    /* same for sx, sy */
    if (sx >= src->width || sy >= src->height)
        return 1;
    
    /* okay, now set up the 0 size == use as much as possible */
    if (xsize == 0)
        xsize = src->width - sx;
    if (ysize == 0)
        ysize = src->height - sy;
    
    /* handle negative dx, dy */
    if (dx < 0) {
        sx -= dx;
        xsize += dx;
        dx = 0;
    }
    if (dy < 0) {
        sy -= dy;
        ysize += dy;
        dy = 0;
    }
    
    /* clip the source rect to fit inside dest, if needed */
    if (dx + xsize > im->width)
        xsize = im->width - dx;
    if (dy + ysize > im->height)
        ysize = im->height - dy;
    
    /* clip the source rect to fit inside src, if needed */
    if (sx + xsize > src->width)
        xsize = src->width - sx;
    if (sy + ysize > src->height)
        ysize = src->height - sy;
    
    /* now, sx, sy, dx, dy, and xsize, ysize are all inside their respective
       bounds, and data can be copied freely */
    
    /* return now if there is nothing to copy */
    if (xsize == 0 || ysize == 0)
        return 1;
    
    /* and finally, call back into the backend if there is work to do */
    return im->backend->composite(im, src, alpha, dx, dy, sx, sy, xsize, ysize);
}
