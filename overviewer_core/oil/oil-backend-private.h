#ifndef __OIL_BACKEND_PRIVATE_H_INCLUDED__
#define __OIL_BACKEND_PRIVATE_H_INCLUDED__

#include "oil.h"

typedef struct {
    /* called when starting up this backend, return 0 on failure */
    int (*initialize)(void);
    
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
    
    /* do a composite. */
    int (*composite)(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize);
    /* draw triangles */
    void (*draw_triangles)(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags);
    
    /* cut src in half, write to im */
    int (*resize_half)(OILImage *im, OILImage *src);
    
    /* clear the image */
    void (*clear)(OILImage *im);
} OILBackend;

extern OILBackend *oil_backend;

#endif /* __OIL_BACKEND_PRIVATE_H_INCLUDED__ */
