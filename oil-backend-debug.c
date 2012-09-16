#include "oil.h"
#include "oil-backend-private.h"

#include <stdio.h>

extern OILBackend oil_backend_cpu;

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

OILBackend oil_backend_debug = {
    oil_backend_debug_new,
    oil_backend_debug_free,
    oil_backend_debug_load,
    oil_backend_debug_save,
    oil_backend_debug_composite,
};
