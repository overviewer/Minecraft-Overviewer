#ifndef __OIL_PRIVATE_H_INCLUDED__
#define __OIL_PRIVATE_H_INCLUDED__

#include "oil.h"

struct _OILImage {
    unsigned int width;
    unsigned int height;
    OILPixel *data;
    int locked;
};

extern OILFormat *oil_formats[];

#endif /* __OIL_PRIVATE_H_INCLUDED__ */
