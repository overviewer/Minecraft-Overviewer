#include "oil.h"
#include "oil-image-private.h"
#include "oil-backend-private.h"

typedef struct {
    unsigned short *depth_buffer;
} CPUPriv;

/* like (a * b + 127) / 255), but fast */
#define MULDIV255(a, b, tmp)                                \
	(tmp = (a) * (b) + 128, ((((tmp) >> 8) + (tmp)) >> 8))

static void oil_backend_cpu_new(OILImage *im) {
    /* add our data struct. FIXME fail if any of these are NULL */
    CPUPriv *priv;
    priv = im->backend_data = malloc(sizeof(CPUPriv));
    priv->depth_buffer = NULL;
}

static void oil_backend_cpu_free(OILImage *im) {
    /* free our data struct */
    CPUPriv *priv = im->backend_data;
    if (priv->depth_buffer)
        free(priv->depth_buffer);
    free(priv);
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

/* draws a triangle on the destination image, multiplicatively!
 * used for smooth lighting
 * (excuse the ridiculous number of parameters!)
 *
 * Algorithm adapted from _Fundamentals_of_Computer_Graphics_
 * by Peter Shirley, Michael Ashikhmin
 * (or at least, the version poorly reproduced here:
 *  http://www.gidforums.com/t-20838.html )
 */

static inline void draw_triangle(OILImage *im, int inclusive, OILVertex v0, OILVertex v1, OILVertex v2, OILTriangleFlags flags) {
    CPUPriv *priv = im->backend_data;
    /* ranges of pixels that are affected */
    int xmin, xmax, ymin, ymax;
    /* constant coefficients for alpha, beta, gamma */
    int a12, a20, a01;
    int b12, b20, b01;
    int c12, c20, c01;
    /* constant normalizers for alpha, beta, gamma */
    float alpha_norm, beta_norm, gamma_norm;
    /* temporary variables */
    /*int tmp;*/
    /* iteration variables */
    int x, y;
    
    /* if we need to, initialize the depth buffer */
    if (flags & OIL_DEPTH_TEST && priv->depth_buffer == NULL) {
        priv->depth_buffer = calloc(im->width * im->height, sizeof(unsigned short));
    }
    
    /* set up draw ranges */
    xmin = OIL_MIN((int)(v0.x), OIL_MIN((int)(v1.x), (int)(v2.x)));
    ymin = OIL_MIN((int)(v0.y), OIL_MIN((int)(v1.y), (int)(v2.y)));
    xmax = OIL_MAX((int)(v0.x), OIL_MAX((int)(v1.x), (int)(v2.x))) + 1;
    ymax = OIL_MAX((int)(v0.y), OIL_MAX((int)(v1.y), (int)(v2.y))) + 1;
    
    xmin = OIL_MAX(xmin, 0);
    ymin = OIL_MAX(ymin, 0);
    xmax = OIL_MIN(xmax, im->width);
    ymax = OIL_MIN(ymax, im->height);
    
    /* setup coefficients */
    a12 = (int)(v1.y) - (int)(v2.y); b12 = (int)(v2.x) - (int)(v1.x); c12 = ((int)(v1.x) * (int)(v2.y)) - ((int)(v2.x) * (int)(v1.y));
    a20 = (int)(v2.y) - (int)(v0.y); b20 = (int)(v0.x) - (int)(v2.x); c20 = ((int)(v2.x) * (int)(v0.y)) - ((int)(v0.x) * (int)(v2.y));
    a01 = (int)(v0.y) - (int)(v1.y); b01 = (int)(v1.x) - (int)(v0.x); c01 = ((int)(v0.x) * (int)(v1.y)) - ((int)(v1.x) * (int)(v0.y));
    
    /* setup normalizers */
    alpha_norm = 1.0f / ((a12 * (int)(v0.x)) + (b12 * (int)(v0.y)) + c12);
    beta_norm  = 1.0f / ((a20 * (int)(v1.x)) + (b20 * (int)(v1.y)) + c20);
    gamma_norm = 1.0f / ((a01 * (int)(v2.x)) + (b01 * (int)(v2.y)) + c01);
    
    /* iterate over the destination rect */
    for (y = ymin; y < ymax; y++) {
        OILPixel *out = &(im->data[y * im->width + xmin]);
        
        for (x = xmin; x < xmax; x++) {
            float alpha, beta, gamma;
            alpha = alpha_norm * ((a12 * x) + (b12 * y) + c12);
            beta  = beta_norm  * ((a20 * x) + (b20 * y) + c20);
            gamma = gamma_norm * ((a01 * x) + (b01 * y) + c01);
            
            if (alpha >= 0 && beta >= 0 && gamma >= 0 &&
                (inclusive || (alpha * beta * gamma > 0))) {
                if (flags & OIL_DEPTH_TEST) {
                    int depth = alpha * v0.z + beta * v1.z + gamma * v2.z;
                    unsigned short *dbuffer = &(priv->depth_buffer[y * im->width + x]);
                    if (depth >= *dbuffer) {
                        /* write to our buffer */
                        *dbuffer = depth;
                    } else {
                        /* skip this, it's behind something */
                        out++;
                        continue;
                    }
                }
                out->r = alpha * v0.color.r + beta * v1.color.r + gamma * v2.color.r;
                out->g = alpha * v0.color.g + beta * v1.color.g + gamma * v2.color.g;
                out->b = alpha * v0.color.b + beta * v1.color.b + gamma * v2.color.b;
                out->a = 255; /* FIXME */
            }
            
            out++;
        }
    }
}

static void oil_backend_cpu_draw_triangles(OILImage *im, OILMatrix *matrix, OILVertex *vertices, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    OILMatrix realmat;
    unsigned int i;
    
    /* first we need to take the given matrix which yields [-1, 1] coordinates
       to something that gives pixel x/y coordinates
       also, invert Y because we want +Y to be up in 3D.
       finally, we need [-1, 1] Z to map to [0, 2^16 - 1] for depth buffer */
    oil_matrix_set_identity(&realmat);
    oil_matrix_scale(&realmat, im->width/2.0f, -(im->height/2.0f), (0xffff/2.0));
    oil_matrix_translate(&realmat, 1.0f, -1.0f, 1.0f);
    oil_matrix_multiply(&realmat, &realmat, matrix);
    
    
    for (i = 0; i < indices_length; i += 3) {
        OILVertex v0, v1, v2;
        v0 = vertices[indices[i]];
        v1 = vertices[indices[i + 1]];
        v2 = vertices[indices[i + 2]];
        
        oil_matrix_transform(&realmat, &(v0.x), &(v0.y), &(v0.z));
        oil_matrix_transform(&realmat, &(v1.x), &(v1.y), &(v1.z));
        oil_matrix_transform(&realmat, &(v2.x), &(v2.y), &(v2.z));
                
        draw_triangle(im, 1, v0, v1, v2, flags);
    }
}

OILBackend oil_backend_cpu = {
    oil_backend_cpu_new,
    oil_backend_cpu_free,
    oil_backend_cpu_load,
    oil_backend_cpu_save,
    oil_backend_cpu_composite,
    oil_backend_cpu_draw_triangles,
};
