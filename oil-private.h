#ifndef __OIL_PRIVATE_H_INCLUDED__
#define __OIL_PRIVATE_H_INCLUDED__

#include "oil.h"

#define OIL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define OIL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define OIL_CLAMP(a, min, max) OIL_MIN((max), OIL_MAX((min), (a)))

#ifdef _MSC_VER
#  define inline __inline
#endif

struct _OILImage {
    unsigned int width;
    unsigned int height;
    OILPixel *data;
    int locked;
};

extern OILFormat *oil_formats[];

typedef struct {
    unsigned int size;
    OILPixel *table;
} OILPalette;

OILPalette *oil_palette_median_cut(OILImage *im, unsigned int size);
void oil_palette_free(OILPalette *p);

unsigned char *oil_dither_nearest(OILImage *im, OILPalette *pal);
unsigned char *oil_dither_floyd_steinberg(OILImage *im, OILPalette *pal);

#endif /* __OIL_PRIVATE_H_INCLUDED__ */
