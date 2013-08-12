#ifndef __OIL_DITHER_PRIVATE_H_INCLUDED__
#define __OIL_DITHER_PRIVATE_H_INCLUDED__

#include "oil.h"
#include "oil-palette-private.h"

unsigned char *oil_dither_nearest(OILImage *im, OILPalette *pal);
unsigned char *oil_dither_floyd_steinberg(OILImage *im, OILPalette *pal);

#endif /* __OIL_DITHER_PRIVATE_H_INCLUDED__ */
