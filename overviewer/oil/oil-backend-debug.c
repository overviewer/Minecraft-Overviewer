#include "oil.h"
#include "oil-backend-private.h"

#include <stdio.h>

extern OILBackend oil_backend_cpu;

static int oil_backend_debug_initialize(void) {
    printf("initialize\n");
    return oil_backend_cpu.initialize();
}

static void oil_backend_debug_new(OILImage *im) {
    printf("new(%p)\n", im);
    oil_backend_cpu.new(im);
}

static void oil_backend_debug_free(OILImage *im) {
    printf("free(%p)\n", im);
    oil_backend_cpu.free(im);
}

static void oil_backend_debug_load(OILImage *im) {
    printf("load(%p)\n", im);
    oil_backend_cpu.load(im);
}

static void oil_backend_debug_save(OILImage *im) {
    printf("save(%p)\n", im);
    oil_backend_cpu.save(im);
}

static int oil_backend_debug_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    printf("composite(%p, %p, %i, %i, %i, %i, %i, %i, %i)\n", im, src, alpha, dx, dy, sx, sy, xsize, ysize);
    return oil_backend_cpu.composite(im, src, alpha, dx, dy, sx, sy, xsize, ysize);
}

static void oil_backend_debug_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    printf("draw_triangles(%p, %p, %p, %p, %i, %p, %i, %i)\n", im, matrix, tex, vertices, vertices_length, indices, indices_length, flags);
    oil_backend_cpu.draw_triangles(im, matrix, tex, vertices, vertices_length, indices, indices_length, flags);
}

static int oil_backend_debug_resize_half(OILImage *im, OILImage *src) {
    printf("resize_half(%p, %p)\n", im, src);
    return oil_backend_cpu.resize_half(im, src);
}

static void oil_backend_debug_clear(OILImage *im) {
    printf("clear(%p)\n", im);
    oil_backend_cpu.clear(im);
}

OILBackend oil_backend_debug = {
    oil_backend_debug_initialize,
    oil_backend_debug_new,
    oil_backend_debug_free,
    oil_backend_debug_load,
    oil_backend_debug_save,
    oil_backend_debug_composite,
    oil_backend_debug_draw_triangles,
    oil_backend_debug_resize_half,
    oil_backend_debug_clear,
};
