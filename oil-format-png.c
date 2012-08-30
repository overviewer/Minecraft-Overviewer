#include "oil.h"

#include <stdlib.h>
#include <png.h>

static void oil_format_png_read(png_structp png, png_bytep data, png_size_t length) {
    OILFile *file = png_get_io_ptr(png);
    file->read(file->file, data, length);
}

static void oil_format_png_write(png_structp png, png_bytep data, png_size_t length) {
    OILFile *file = png_get_io_ptr(png);
    file->write(file->file, data, length);
}

static void oil_format_png_flush(png_structp png) {
    OILFile *file = png_get_io_ptr(png);
    file->flush(file->file);
}

int oil_format_png_save(OILImage *im, OILFile *file) {
    png_structp png;
    png_infop info;
    unsigned int width, height;
    const OILPixel *data;
    unsigned int y;
    
    if (!im || !file)
        return 0;
    
    oil_image_get_size(im, &width, &height);
    data = oil_image_get_data(im);
    
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        return 0;
    }
    
    info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, NULL);
        return 0;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        return 0;
    }
    
    png_set_write_fn(png, file, oil_format_png_write, oil_format_png_flush);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB_ALPHA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
    png_write_info(png, info);
    
    for (y = 0; y < height; y++) {
        png_write_row(png, (png_bytep)(&(data[width * y])));
    }
    
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    
    return 1;
}

OILImage *oil_format_png_load(OILFile *file) {
    OILImage *im = NULL;
    OILPixel *data;
    unsigned int width, height;
    png_byte color_type, bit_depth;
    int passes;
    png_structp png;
    png_infop info;
    char header[8];
    unsigned int y;
    
    if (!file)
        return NULL;
    
    if (file->read(file->file, header, 8) != 8) {
        return NULL;
    }
    
    if (png_sig_cmp(header, 0, 8)) {
        return NULL;
    }
    
    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        return NULL;
    }
    
    info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        return NULL;
    }
    
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        if (im)
            oil_image_free(im);
        return NULL;
    }
    
    png_set_read_fn(png, file, oil_format_png_read);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);
    
    width = png_get_image_width(png, info);
    height = png_get_image_height(png, info);
    color_type = png_get_color_type(png, info);
    bit_depth = png_get_bit_depth(png, info);
    passes = png_set_interlace_handling(png);
    
    /* convert palettes to RGB, expand low-bit grayscale to 8-bit
       and use tRNS chunks for alpha channels
       after this, it's all 8/16 bit RGB/grayscale, with or without alpha */
    png_set_expand(png);
    
    /* if it's grayscale, convert to RGB
       after this, it's all 8/16 bit RGB or RGBA */
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    
    
    /* if it's 16-bit, strip that down to 8
       after this it's all 8-bit RGB or RGBA */
    if (bit_depth == 16)
        png_set_strip_16(png);
    
    /* if it has no alpha, add it
       after this, it's all 8-bit RGBA */
    if (!(color_type & PNG_COLOR_MASK_ALPHA))
        png_set_filler(png, 0xff, PNG_FILLER_AFTER);
    
    png_read_update_info(png, info);
    
    im = oil_image_new(width, height);
    if (!im) {
        png_destroy_read_struct(&png, &info, NULL);
        return NULL;
    }
    
    data = oil_image_lock(im);
    if (!data) {
        oil_image_free(im);
        png_destroy_read_struct(&png, &info, NULL);
        return NULL;
    }
    
    for (; passes > 0; passes--) {
        for (y = 0; y < height; y++) {
            png_read_row(png, (png_bytep)(&(data[y * width])), NULL);
        }
    }
    oil_image_unlock(im);
    
    png_destroy_read_struct(&png, &info, NULL);
    return im;
}

OILFormat oil_format_png = {
    oil_format_png_save,
    oil_format_png_load,
};
