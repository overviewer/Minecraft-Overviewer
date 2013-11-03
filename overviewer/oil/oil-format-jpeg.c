#include "oil.h"
#include "oil-format-private.h"

#include <stdlib.h>
#include <stdio.h>
#include <setjmp.h>
#include <jpeglib.h>
#include <jerror.h>

#define BUF_SIZE 4096

typedef struct {
    struct jpeg_error_mgr parent;
    jmp_buf setjmp_buffer;
} oil_error_mgr;

static void oil_error_exit(j_common_ptr cinfo) {
    oil_error_mgr *err = (oil_error_mgr *)(cinfo->err);
    cinfo->err->output_message(cinfo);
    longjmp(err->setjmp_buffer, 1);
}

typedef struct {
    struct jpeg_destination_mgr parent;
    OILFile *file;
    JOCTET *buffer;
} oil_destination_mgr;

static void oil_init_destination(j_compress_ptr cinfo) {
    oil_destination_mgr *dest = (oil_destination_mgr *)(cinfo->dest);
    
    dest->buffer = (JOCTET *)(cinfo->mem->alloc_small((j_common_ptr)cinfo, JPOOL_IMAGE, BUF_SIZE * sizeof(JOCTET)));
    dest->parent.next_output_byte = dest->buffer;
    dest->parent.free_in_buffer = BUF_SIZE;
}

static boolean oil_empty_output_buffer(j_compress_ptr cinfo) {
    oil_destination_mgr *dest = (oil_destination_mgr *)(cinfo->dest);
    
    if (dest->file->write(dest->file->file, dest->buffer, BUF_SIZE) != BUF_SIZE) {
        ERREXIT(cinfo, JERR_FILE_WRITE);
    }
    
    dest->parent.next_output_byte = dest->buffer;
    dest->parent.free_in_buffer = BUF_SIZE;
    
    return TRUE;
}

static void oil_term_destination(j_compress_ptr cinfo) {
    oil_destination_mgr *dest = (oil_destination_mgr *)(cinfo->dest);
    size_t datacount = BUF_SIZE - dest->parent.free_in_buffer;
    
    if (datacount > 0) {
        if (dest->file->write(dest->file->file, dest->buffer, datacount) != datacount) {
            ERREXIT(cinfo, JERR_FILE_WRITE);
        }
    }
    
    dest->file->flush(dest->file->file);
}

static int oil_format_jpeg_save(OILImage *im, OILFile *file, OILFormatOptions *opts) {
    unsigned int width, height;
    struct jpeg_compress_struct cinfo;
    oil_error_mgr jerr;
    oil_destination_mgr dest;
    const OILPixel *data;
    JOCTET *row;
    
    oil_image_get_size(im, &width, &height);
    
    row = malloc(sizeof(JOCTET) * width * 3);
    if (!row)
        return 0;
    
    cinfo.err = jpeg_std_error((struct jpeg_error_mgr*)&jerr);
    jerr.parent.error_exit = oil_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_compress(&cinfo);
        free(row);
        
        return 0;
    }
    
    jpeg_create_compress(&cinfo);
    
    /* create our jpeg destination manager */
    dest.parent.init_destination = oil_init_destination;
    dest.parent.empty_output_buffer = oil_empty_output_buffer;
    dest.parent.term_destination = oil_term_destination;
    dest.file = file;
    cinfo.dest = (struct jpeg_destination_mgr *)&dest;
    
    /* set image info */
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    
    /* compression defaults */
    jpeg_set_defaults(&cinfo);
    
    /* get the data */
    data = oil_image_get_data(im);
    
    /* compress our data */
    jpeg_start_compress(&cinfo, TRUE);
    while (cinfo.next_scanline < cinfo.image_height) {
        unsigned int x;
        for (x = 0; x < width; x++) {
            const OILPixel p = data[cinfo.next_scanline * width + x];
            row[3 * x + 0] = p.r;
            row[3 * x + 1] = p.g;
            row[3 * x + 2] = p.b;
        }
        jpeg_write_scanlines(&cinfo, &row, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    free(row);
    return 1;
}

typedef struct {
    struct jpeg_source_mgr parent;
    OILFile *file;
    JOCTET *buffer;
    boolean start_of_file;
} oil_source_mgr;

static void oil_init_source(j_decompress_ptr cinfo) {
    oil_source_mgr *src = (oil_source_mgr *)(cinfo->src);
    
    src->buffer = (JOCTET *)(cinfo->mem->alloc_small((j_common_ptr)cinfo, JPOOL_IMAGE, BUF_SIZE * sizeof(JOCTET)));
    src->start_of_file = TRUE;
    src->parent.bytes_in_buffer = 0;
    src->parent.next_input_byte = NULL;
}

static boolean oil_fill_input_buffer(j_decompress_ptr cinfo) {
    oil_source_mgr *src = (oil_source_mgr *)(cinfo->src);
    size_t nbytes;
    
    nbytes = src->file->read(src->file->file, src->buffer, BUF_SIZE);
    if (nbytes == 0) {
        if (src->start_of_file)
            ERREXIT(cinfo, JERR_INPUT_EMPTY);
        
        /* insert a fake EOI marker */
        WARNMS(cinfo, JWRN_JPEG_EOF);
        src->buffer[0] = 0xFF;
        src->buffer[1] = JPEG_EOI;
        nbytes = 2;
    }
    
    src->parent.next_input_byte = src->buffer;
    src->parent.bytes_in_buffer = nbytes;
    src->start_of_file = FALSE;
    
    return TRUE;
}

static void oil_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    oil_source_mgr *src = (oil_source_mgr *)(cinfo->src);
    
    if (num_bytes > 0) {
        while (num_bytes > src->parent.bytes_in_buffer) {
            num_bytes -= src->parent.bytes_in_buffer;
            oil_fill_input_buffer(cinfo);
        }
        
        src->parent.next_input_byte += num_bytes;
        src->parent.bytes_in_buffer -= num_bytes;
    }
}

static void oil_term_source(j_decompress_ptr cinfo) {
    /* nothing doing */
}

static OILImage *oil_format_jpeg_load(OILFile *file) {
    OILImage *im = NULL;
    OILPixel *data = NULL;
    JOCTET *row = NULL;
    struct jpeg_decompress_struct cinfo;
    oil_error_mgr jerr;
    oil_source_mgr src;
    
    cinfo.err = jpeg_std_error((struct jpeg_error_mgr*)&jerr);
    jerr.parent.error_exit = oil_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        if (row)
            free(row);
        if (im)
            oil_image_free(im);
        
        return NULL; // FIXME
    }
    
    jpeg_create_decompress(&cinfo);
    
    /* create our jpeg source manager */
    src.parent.init_source = oil_init_source;
    src.parent.fill_input_buffer = oil_fill_input_buffer;
    src.parent.skip_input_data = oil_skip_input_data;
    src.parent.resync_to_restart = jpeg_resync_to_restart;
    src.parent.term_source = oil_term_source;
    src.file = file;
    cinfo.src = (struct jpeg_source_mgr *)&src;
    
    jpeg_read_header(&cinfo, TRUE);
    jpeg_start_decompress(&cinfo);
    
    row = malloc(sizeof(JOCTET) * cinfo.output_components * cinfo.output_width);
    if (!row) {
        jpeg_destroy_decompress(&cinfo);
        return NULL;
    }
    
    im = oil_image_new(cinfo.output_width, cinfo.output_height);
    if (!im) {
        jpeg_destroy_decompress(&cinfo);
        free(row);
        return NULL;
    }
    
    data = oil_image_lock(im);
    if (!data) {
        jpeg_destroy_decompress(&cinfo);
        free(row);
        oil_image_free(im);
        return NULL;
    }
    
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned int x;
        jpeg_read_scanlines(&cinfo, &row, 1);
        for (x = 0; x < cinfo.output_width; x++) {
            OILPixel p;
            if (cinfo.output_components == 1) {
                p.r = row[x];
                p.g = row[x];
                p.b = row[x];
            } else {
                p.r = row[3 * x + 0];
                p.g = row[3 * x + 1];
                p.b = row[3 * x + 2];
            }
            p.a = 255;
            /* -1 because we just read one in */
            data[(cinfo.output_scanline - 1) * cinfo.output_width + x] = p;
        }
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    free(row);
    
    oil_image_unlock(im);
    
    return im;
}

OILFormat oil_format_jpeg = {
    oil_format_jpeg_save,
    oil_format_jpeg_load,
};
