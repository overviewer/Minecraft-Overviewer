#include "oil.h"
#include "oil-image-private.h"
#include "oil-backend-private.h"

/* like (a * b + 127) / 255), but fast */
#define MULDIV255(a, b, tmp)                                \
	(tmp = (a) * (b) + 128, ((((tmp) >> 8) + (tmp)) >> 8))

static void oil_backend_cpu_new(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_free(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_load(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_save(OILImage *im) {
    /* nothing to do */
}

static int oil_backend_cpu_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    /* used by MULDIV255 */
    int tmp1, tmp2, tmp3;
    unsigned int x, y;
    
    for (y = 0; y < ysize; y++) {
        OILPixel *out = &(im->data[dx + (dy + y) * im->width]);
        OILPixel *in = &(src->data[sx + (sy + y) * src->width]);
        
        for (x = 0; x < xsize; x++) {
            unsigned char in_alpha;
            
            /* apply overall alpha */
            if (alpha != 255 && in->a != 0) {
                in_alpha = MULDIV255(in->a, alpha, tmp1);
            } else {
                in_alpha = in->a;
            }
            
            /* special cases */
            if (in_alpha == 255 || (out->a == 0 && in_alpha > 0)) {
                /* straight-up copy */
                *out = *in;
                out->a = in_alpha;
            } else if (in_alpha == 0) {
                /* source is fully transparent, do nothing */
            } else {
                /* general case */
                int comp_alpha = in_alpha + MULDIV255(out->a, 255 - in_alpha, tmp1);
                
                out->r = MULDIV255(in->r, in_alpha, tmp1) + MULDIV255(MULDIV255(out->r, out->a, tmp2), 255 - in_alpha, tmp3);
                out->r = (out->r * 255) / comp_alpha;
                
                out->g = MULDIV255(in->g, in_alpha, tmp1) + MULDIV255(MULDIV255(out->g, out->a, tmp2), 255 - in_alpha, tmp3);
                out->g = (out->g * 255) / comp_alpha;
                
                out->b = MULDIV255(in->b, in_alpha, tmp1) + MULDIV255(MULDIV255(out->b, out->a, tmp2), 255 - in_alpha, tmp3);
                out->b = (out->b * 255) / comp_alpha;
                
                out->a = comp_alpha;
            }
            
            /* move forward */
            out++;
            in++;
        }
    }
    
    return 1;
}

OILBackend oil_backend_cpu = {
    oil_backend_cpu_new,
    oil_backend_cpu_free,
    oil_backend_cpu_load,
    oil_backend_cpu_save,
    oil_backend_cpu_composite,
};
