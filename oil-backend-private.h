#ifndef __OIL_BACKEND_PRIVATE_H_INCLUDED__
#define __OIL_BACKEND_PRIVATE_H_INCLUDED__

#include "oil.h"

typedef struct {
    /* called when creating an image */
    void (*new)(OILImage *im);
    /* called when destroying an image */
    void (*free)(OILImage *im);
    /* load data out of backend and into im
       called during (for example) get_data() */
    void (*load)(OILImage *im);
    /* save data from im into backend
       called during (for example) unlock() */
    void (*save)(OILImage *im);
} OILBackend;

extern OILBackend *oil_backend;

#endif /* __OIL_BACKEND_PRIVATE_H_INCLUDED__ */
