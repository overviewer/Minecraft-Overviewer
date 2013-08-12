#ifndef __OIL_PALETTE_PRIVATE_H_INCLUDED__
#define __OIL_PALETTE_PRIVATE_H_INCLUDED__

#include "oil.h"

typedef struct {
    unsigned int size;
    OILPixel *table;
} OILPalette;

OILPalette *oil_palette_median_cut(OILImage *im, unsigned int size);
void oil_palette_free(OILPalette *p);

#endif /* __OIL_PALETTE_PRIVATE_H_INCLUDED__ */
