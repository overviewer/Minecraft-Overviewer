#ifndef __OIL_IMAGE_PRIVATE_H_INCLUDED__
#define __OIL_IMAGE_PRIVATE_H_INCLUDED__

#include "oil.h"
#include "oil-backend-private.h"

struct _OILImage {
    unsigned int width;
    unsigned int height;
    OILPixel *data;
    int locked;
    
    OILBackend *backend;
};

#endif /* __OIL_IMAGE_PRIVATE_H_INCLUDED__ */
