#include "oil.h"
#include "oil-image-private.h"
#include "oil-format-private.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

/*
 * file IO implementation that caches a small buffer in the beginning
 * used for resetting IO after a format fails to load
 *
 * (only implements read)
 */

#define RESET_FILE_SIZE 4096

typedef struct {
    OILFile *file;
    size_t readpos;
    unsigned char buffer[RESET_FILE_SIZE];
    size_t buffer_size;
} OILResetFile;

static size_t oil_reset_file_read(void *file, void *data, size_t length) {
    size_t read = 0;
    OILResetFile *rfile = (OILResetFile *)file;
    
    if (rfile->buffer_size == 0) {
        rfile->buffer_size = rfile->file->read(rfile->file->file, rfile->buffer, RESET_FILE_SIZE);
        if (rfile->buffer_size == 0)
            return 0;
    }
    
    if (rfile->readpos < rfile->buffer_size) {
        size_t from_buffer = rfile->buffer_size - rfile->readpos;
        if (from_buffer > length)
            from_buffer = length;
        
        memcpy(data, &(rfile->buffer[rfile->readpos]), from_buffer);
        
        rfile->readpos += from_buffer;
        read += from_buffer;
        
        /* modify these for the next block */
        data += from_buffer;
        length -= from_buffer;
    }
    
    if (length > 0) {
        read += rfile->file->read(rfile->file->file, data, length);
        rfile->readpos += read;
    }
    
    return read;
}

static void oil_reset_file_init(OILFile *out, OILResetFile *rfile, OILFile *file) {
    rfile->file = file;
    rfile->readpos = 0;
    rfile->buffer_size = 0;
    
    out->file = rfile;
    out->read = oil_reset_file_read;
    out->write = NULL;
    out->flush = NULL;
}

static inline int oil_reset_file_reset(OILResetFile *rfile) {
    if (rfile->readpos < RESET_FILE_SIZE) {
        rfile->readpos = 0;
        return 1; /* was reset */
    }
    return 0; /* cannot be reset */
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
    OILFile rfile;
    OILResetFile rfiledata;
	unsigned int i;
	if (!file)
		return NULL;
    
    oil_reset_file_init(&rfile, &rfiledata, file);
	
	for (i = 0; i < OIL_FORMAT_MAX; i++) {
		OILImage *im = oil_formats[i]->load(&rfile);
		if (im != NULL)
			return im;
        
        if (!oil_reset_file_reset(&rfiledata))
            return NULL;
	}
	return NULL;
}

int oil_image_save(OILImage *im, const char *path, OILFormatName format, OILFormatOptions *opts) {
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
    
    ret = oil_image_save_ex(im, &file, format, opts);
    fclose(fp);
    return ret;
}

int oil_image_save_ex(OILImage *im, OILFile *file, OILFormatName format, OILFormatOptions *opts) {
	if (!im || !file)
		return NULL;
	if (!(format < OIL_FORMAT_MAX))
		return NULL;

    return oil_formats[format]->save(im, file, opts);
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

void oil_image_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    /* all of these are unhandleable */
    if (!im || !vertices || !matrix || !indices || indices_length % 3 != 0 || vertices_length == 0)
        return;
    if (tex && (im->backend != tex->backend))
        return;
    
    /* this is short-circuit-able */
    if (indices_length == 0)
        return;
    
    /* ok now that that's out of the way, throw it to the backend */
    im->backend->draw_triangles(im, matrix, tex, vertices, vertices_length, indices, indices_length, flags);
}

int oil_image_resize_half(OILImage *im, OILImage *src) {
    if (!im || !src)
        return 0;
    if (im->backend != src->backend)
        return 0;
    if (im->height * 2 != src->height || im->width * 2 != src->width)
        return 0;
    
    return im->backend->resize_half(im, src);
}

void oil_image_clear(OILImage *im) {
    if (!im)
        return;
    
    im->backend->clear(im);
}
