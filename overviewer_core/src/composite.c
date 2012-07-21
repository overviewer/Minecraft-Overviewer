/* 
 * This file is part of the Minecraft Overviewer.
 *
 * Minecraft Overviewer is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or (at
 * your option) any later version.
 *
 * Minecraft Overviewer is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.
 */

/* 
 * This file implements a custom alpha_over function for (some) PIL
 * images. It's designed to be used through composite.py, which
 * includes a proxy alpha_over function that falls back to the default
 * PIL paste if this extension is not found.
 */

#include "overviewer.h"

typedef struct {
    PyObject_HEAD
    Imaging image;
} ImagingObject;

inline Imaging
imaging_python_to_c(PyObject *obj)
{
    PyObject *im;
    Imaging image;

    /* first, get the 'im' attribute */
    im = PyObject_GetAttrString(obj, "im");
    if (!im)
        return NULL;

    /* make sure 'im' is the right type */
    if (strcmp(im->ob_type->tp_name, "ImagingCore") != 0) {
        /* it's not -- raise an error and exit */
        PyErr_SetString(PyExc_TypeError,
                        "image attribute 'im' is not a core Imaging type");
        return NULL;
    }

    image = ((ImagingObject *)im)->image;
    Py_DECREF(im);
    return image;
}

/* helper function to setup s{x,y}, d{x,y}, and {x,y}size variables
   in these composite functions -- even handles auto-sizing to src! */
static inline void
setup_source_destination(Imaging src, Imaging dest,
                         int *sx, int *sy, int *dx, int *dy, int *xsize, int *ysize)
{
   /* handle negative/zero sizes appropriately */
    if (*xsize <= 0 || *ysize <= 0) {
        *xsize = src->xsize;
        *ysize = src->ysize;
    }
    
    /* set up the source position, size and destination position */
    /* handle negative dest pos */
    if (*dx < 0) {
        *sx = -(*dx);
        *dx = 0;
    } else {
        *sx = 0;
    }

    if (*dy < 0) {
        *sy = -(*dy);
        *dy = 0;
    } else {
        *sy = 0;
    }

    /* set up source dimensions */
    *xsize -= *sx;
    *ysize -= *sy;

    /* clip dimensions, if needed */
    if (*dx + *xsize > dest->xsize)
        *xsize = dest->xsize - *dx;
    if (*dy + *ysize > dest->ysize)
        *ysize = dest->ysize - *dy;
}

/* convenience alpha_over with 1.0 as overall_alpha */
inline PyObject* alpha_over(PyObject *dest, PyObject *src, PyObject *mask,
                            int dx, int dy, int xsize, int ysize) {
    return alpha_over_full(dest, src, mask, 1.0f, dx, dy, xsize, ysize);
}

/* the full alpha_over function, in a form that can be called from C
 * overall_alpha is multiplied with the whole mask, useful for lighting...
 * if xsize, ysize are negative, they are instead set to the size of the image in src
 * returns NULL on error, dest on success. You do NOT need to decref the return!
 */
inline PyObject *
alpha_over_full(PyObject *dest, PyObject *src, PyObject *mask, float overall_alpha,
                int dx, int dy, int xsize, int ysize) {
    /* libImaging handles */
    Imaging imDest, imSrc, imMask;
    /* cached blend properties */
    int src_has_alpha, mask_offset, mask_stride;
    /* source position */
    int sx, sy;
    /* iteration variables */
    unsigned int x, y, i;
    /* temporary calculation variables */
    int tmp1, tmp2, tmp3;
    /* integer [0, 255] version of overall_alpha */
    UINT8 overall_alpha_int = 255 * overall_alpha;
    
    /* short-circuit this whole thing if overall_alpha is zero */
    if (overall_alpha_int == 0)
        return dest;

    imDest = imaging_python_to_c(dest);
    imSrc = imaging_python_to_c(src);
    imMask = imaging_python_to_c(mask);

    if (!imDest || !imSrc || !imMask)
        return NULL;

    /* check the various image modes, make sure they make sense */
    if (strcmp(imDest->mode, "RGBA") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given destination image does not have mode \"RGBA\"");
        return NULL;
    }

    if (strcmp(imSrc->mode, "RGBA") != 0 && strcmp(imSrc->mode, "RGB") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given source image does not have mode \"RGBA\" or \"RGB\"");
        return NULL;
    }

    if (strcmp(imMask->mode, "RGBA") != 0 && strcmp(imMask->mode, "L") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given mask image does not have mode \"RGBA\" or \"L\"");
        return NULL;
    }

    /* make sure mask size matches src size */
    if (imSrc->xsize != imMask->xsize || imSrc->ysize != imMask->ysize) {
        PyErr_SetString(PyExc_ValueError,
                        "mask and source image sizes do not match");
        return NULL;
    }

    /* set up flags for the src/mask type */
    src_has_alpha = (imSrc->pixelsize == 4 ? 1 : 0);
    /* how far into image the first alpha byte resides */
    mask_offset = (imMask->pixelsize == 4 ? 3 : 0);
    /* how many bytes to skip to get to the next alpha byte */
    mask_stride = imMask->pixelsize;

    /* setup source & destination vars */
    setup_source_destination(imSrc, imDest, &sx, &sy, &dx, &dy, &xsize, &ysize);

    /* check that there remains any blending to be done */
    if (xsize <= 0 || ysize <= 0) {
        /* nothing to do, return */
        return dest;
    }

    for (y = 0; y < ysize; y++) {
        UINT8 *out = (UINT8 *)imDest->image[dy + y] + dx * 4;
        UINT8 *outmask = (UINT8 *)imDest->image[dy + y] + dx * 4 + 3;
        UINT8 *in = (UINT8 *)imSrc->image[sy + y] + sx * (imSrc->pixelsize);
        UINT8 *inmask = (UINT8 *)imMask->image[sy + y] + sx * mask_stride + mask_offset;

        for (x = 0; x < xsize; x++) {
            UINT8 in_alpha;
            
            /* apply overall_alpha */
            if (overall_alpha_int != 255 && *inmask != 0) {
                in_alpha = MULDIV255(*inmask, overall_alpha_int, tmp1);
            } else {
                in_alpha = *inmask;
            }
            
            /* special cases */
            if (in_alpha == 255 || (*outmask == 0 && in_alpha > 0)) {
                *outmask = in_alpha;

                *out = *in;
                out++, in++;
                *out = *in;
                out++, in++;
                *out = *in;
                out++, in++;
            } else if (in_alpha == 0) {
                /* do nothing -- source is fully transparent */
                out += 3;
                in += 3;
            } else {
                /* general case */
                int alpha = in_alpha + MULDIV255(*outmask, 255 - in_alpha, tmp1);
                for (i = 0; i < 3; i++) {
                    /* general case */
                    *out = MULDIV255(*in, in_alpha, tmp1) +
                        MULDIV255(MULDIV255(*out, *outmask, tmp2), 255 - in_alpha, tmp3);
                    
                    *out = (*out * 255) / alpha;
                    out++, in++;
                }

                *outmask = alpha;
            }

            out++;
            if (src_has_alpha)
                in++;
            outmask += 4;
            inmask += mask_stride;
        }
    }

    return dest;
}

/* wraps alpha_over so it can be called directly from python */
/* properly refs the return value when needed: you DO need to decref the return */
PyObject *
alpha_over_wrap(PyObject *self, PyObject *args)
{
    /* raw input python variables */
    PyObject *dest, *src, *pos = NULL, *mask = NULL;
    /* destination position and size */
    int dx, dy, xsize, ysize;
    /* return value: dest image on success */
    PyObject *ret;

    if (!PyArg_ParseTuple(args, "OO|OO", &dest, &src, &pos, &mask))
        return NULL;
    
    if (mask == NULL)
        mask = src;
    
    /* destination position read */
    if (pos == NULL) {
        xsize = 0;
        ysize = 0;
        dx = 0;
        dy = 0;
    } else {
        if (!PyArg_ParseTuple(pos, "iiii", &dx, &dy, &xsize, &ysize)) {
            /* try again, but this time try to read a point */
            PyErr_Clear();
            xsize = 0;
            ysize = 0;
            if (!PyArg_ParseTuple(pos, "ii", &dx, &dy)) {
                PyErr_SetString(PyExc_TypeError,
                                "given blend destination rect is not valid");
                return NULL;
            }
        }
    }

    ret = alpha_over(dest, src, mask, dx, dy, xsize, ysize);
    if (ret == dest) {
        /* Python needs us to own our return value */
        Py_INCREF(dest);
    }
    return ret;
}

/* like alpha_over, but instead of src image it takes a source color
 * also, it multiplies instead of doing an over operation
 */
PyObject *
tint_with_mask(PyObject *dest, unsigned char sr, unsigned char sg,
               unsigned char sb, unsigned char sa,
               PyObject *mask, int dx, int dy, int xsize, int ysize) {
    /* libImaging handles */
    Imaging imDest, imMask;
    /* cached blend properties */
    int mask_offset, mask_stride;
    /* source position */
    int sx, sy;
    /* iteration variables */
    unsigned int x, y;
    /* temporary calculation variables */
    int tmp1, tmp2;

    imDest = imaging_python_to_c(dest);
    imMask = imaging_python_to_c(mask);

    if (!imDest || !imMask)
        return NULL;

    /* check the various image modes, make sure they make sense */
    if (strcmp(imDest->mode, "RGBA") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given destination image does not have mode \"RGBA\"");
        return NULL;
    }

    if (strcmp(imMask->mode, "RGBA") != 0 && strcmp(imMask->mode, "L") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given mask image does not have mode \"RGBA\" or \"L\"");
        return NULL;
    }

    /* how far into image the first alpha byte resides */
    mask_offset = (imMask->pixelsize == 4 ? 3 : 0);
    /* how many bytes to skip to get to the next alpha byte */
    mask_stride = imMask->pixelsize;

    /* setup source & destination vars */
    setup_source_destination(imMask, imDest, &sx, &sy, &dx, &dy, &xsize, &ysize);

    /* check that there remains any blending to be done */
    if (xsize <= 0 || ysize <= 0) {
        /* nothing to do, return */
        return dest;
    }

    for (y = 0; y < ysize; y++) {
        UINT8 *out = (UINT8 *)imDest->image[dy + y] + dx * 4;
        UINT8 *inmask = (UINT8 *)imMask->image[sy + y] + sx * mask_stride + mask_offset;

        for (x = 0; x < xsize; x++) {
            /* special cases */
            if (*inmask == 255) {
                *out = MULDIV255(*out, sr, tmp1);
                out++;
                *out = MULDIV255(*out, sg, tmp1);
                out++;
                *out = MULDIV255(*out, sb, tmp1);
                out++;
                *out = MULDIV255(*out, sa, tmp1);
                out++;
            } else if (*inmask == 0) {
                /* do nothing -- source is fully transparent */
                out += 4;
            } else {
                /* general case */
                
                /* TODO work out general case */
                *out = MULDIV255(*out, (255 - *inmask) + MULDIV255(sr, *inmask, tmp1), tmp2);
                out++;
                *out = MULDIV255(*out, (255 - *inmask) + MULDIV255(sg, *inmask, tmp1), tmp2);
                out++;
                *out = MULDIV255(*out, (255 - *inmask) + MULDIV255(sb, *inmask, tmp1), tmp2);
                out++;
                *out = MULDIV255(*out, (255 - *inmask) + MULDIV255(sa, *inmask, tmp1), tmp2);
                out++;
            }

            inmask += mask_stride;
        }
    }

    return dest;
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
PyObject *
draw_triangle(PyObject *dest, int inclusive,
              int x0, int y0,
              unsigned char r0, unsigned char g0, unsigned char b0,
              int x1, int y1,
              unsigned char r1, unsigned char g1, unsigned char b1,
              int x2, int y2,
              unsigned char r2, unsigned char g2, unsigned char b2,
              int tux, int tuy, int *touchups, unsigned int num_touchups) {
    
    /* destination image */
    Imaging imDest;
    /* ranges of pixels that are affected */
    int xmin, xmax, ymin, ymax;
    /* constant coefficients for alpha, beta, gamma */
    int a12, a20, a01;
    int b12, b20, b01;
    int c12, c20, c01;
    /* constant normalizers for alpha, beta, gamma */
    float alpha_norm, beta_norm, gamma_norm;
    /* temporary variables */
    int tmp;
    /* iteration variables */
    int x, y;
    
    imDest = imaging_python_to_c(dest);
    if (!imDest)
        return NULL;

    /* check the various image modes, make sure they make sense */
    if (strcmp(imDest->mode, "RGBA") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given destination image does not have mode \"RGBA\"");
        return NULL;
    }
    
    /* set up draw ranges */
    xmin = MIN(x0, MIN(x1, x2));
    ymin = MIN(y0, MIN(y1, y2));
    xmax = MAX(x0, MAX(x1, x2)) + 1;
    ymax = MAX(y0, MAX(y1, y2)) + 1;
    
    xmin = MAX(xmin, 0);
    ymin = MAX(ymin, 0);
    xmax = MIN(xmax, imDest->xsize);
    ymax = MIN(ymax, imDest->ysize);
    
    /* setup coefficients */
    a12 = y1 - y2; b12 = x2 - x1; c12 = (x1 * y2) - (x2 * y1);
    a20 = y2 - y0; b20 = x0 - x2; c20 = (x2 * y0) - (x0 * y2);
    a01 = y0 - y1; b01 = x1 - x0; c01 = (x0 * y1) - (x1 * y0);
    
    /* setup normalizers */
    alpha_norm = 1.0f / ((a12 * x0) + (b12 * y0) + c12);
    beta_norm  = 1.0f / ((a20 * x1) + (b20 * y1) + c20);
    gamma_norm = 1.0f / ((a01 * x2) + (b01 * y2) + c01);
    
    /* iterate over the destination rect */
    for (y = ymin; y < ymax; y++) {
        UINT8 *out = (UINT8 *)imDest->image[y] + xmin * 4;
        
        for (x = xmin; x < xmax; x++) {
            float alpha, beta, gamma;
            alpha = alpha_norm * ((a12 * x) + (b12 * y) + c12);
            beta  = beta_norm  * ((a20 * x) + (b20 * y) + c20);
            gamma = gamma_norm * ((a01 * x) + (b01 * y) + c01);
            
            if (alpha >= 0 && beta >= 0 && gamma >= 0 &&
                (inclusive || (alpha * beta * gamma > 0))) {
                unsigned int r = alpha * r0 + beta * r1 + gamma * r2;
                unsigned int g = alpha * g0 + beta * g1 + gamma * g2;
                unsigned int b = alpha * b0 + beta * b1 + gamma * b2;
                
                *out = MULDIV255(*out, r, tmp); out++;
                *out = MULDIV255(*out, g, tmp); out++;
                *out = MULDIV255(*out, b, tmp); out++;
                
                /* keep alpha the same */
                out++;
            } else {
                /* skip */
                out += 4;
            }
        }
    }
    
    while (num_touchups > 0) {
        float alpha, beta, gamma;
        unsigned int r, g, b;
        UINT8 *out;
        
        x = touchups[0] + tux;
        y = touchups[1] + tuy;
        touchups += 2;
        num_touchups--;
        
        if (x < 0 || x >= imDest->xsize || y < 0 || y >= imDest->ysize)
            continue;

        out = (UINT8 *)imDest->image[y] + x * 4;

        alpha = alpha_norm * ((a12 * x) + (b12 * y) + c12);
        beta  = beta_norm  * ((a20 * x) + (b20 * y) + c20);
        gamma = gamma_norm * ((a01 * x) + (b01 * y) + c01);
        
        r = alpha * r0 + beta * r1 + gamma * r2;
        g = alpha * g0 + beta * g1 + gamma * g2;
        b = alpha * b0 + beta * b1 + gamma * b2;
                
        *out = MULDIV255(*out, r, tmp); out++;
        *out = MULDIV255(*out, g, tmp); out++;
        *out = MULDIV255(*out, b, tmp); out++;
    }
    
    return dest;
}

/* scales the image to half size
 */
inline PyObject *
resize_half(PyObject *dest, PyObject *src) {
    /* libImaging handles */
    Imaging imDest, imSrc;
    /* alpha properties */
    int src_has_alpha, dest_has_alpha;
    /* iteration variables */
    unsigned int x, y;
    /* temp color variables */
    unsigned int r, g, b, a;    
    /* size values for source and destination */
    int src_width, src_height, dest_width, dest_height;
    
    imDest = imaging_python_to_c(dest);
    imSrc = imaging_python_to_c(src);
    
    if (!imDest || !imSrc)
        return NULL;
    
    /* check the various image modes, make sure they make sense */
    if (strcmp(imDest->mode, "RGBA") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given destination image does not have mode \"RGBA\"");
        return NULL;
    }
    
    if (strcmp(imSrc->mode, "RGBA") != 0 && strcmp(imSrc->mode, "RGB") != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "given source image does not have mode \"RGBA\" or \"RGB\"");
        return NULL;
    }
    
    src_width = imSrc->xsize;
    src_height = imSrc->ysize;
    dest_width = imDest->xsize;
    dest_height = imDest->ysize;
    
    /* make sure destination size is 1/2 src size */
    if (src_width / 2 != dest_width || src_height / 2 != dest_height) {
        PyErr_SetString(PyExc_ValueError,
                        "destination image size is not one-half source image size");
        return NULL;
    }
    
    /* set up flags for the src/mask type */
    src_has_alpha = (imSrc->pixelsize == 4 ? 1 : 0);
    dest_has_alpha = (imDest->pixelsize == 4 ? 1 : 0);
    
    /* check that there remains anything to resize */
    if (dest_width <= 0 || dest_height <= 0) {
        /* nothing to do, return */
        return dest;
    }
    
    /* set to fully opaque if source has no alpha channel */
    if(!src_has_alpha)
        a = 0xFF << 2;
    
    for (y = 0; y < dest_height; y++) {
        
        UINT8 *out = (UINT8 *)imDest->image[y];
        UINT8 *in_row1 = (UINT8 *)imSrc->image[y * 2];
        UINT8 *in_row2 = (UINT8 *)imSrc->image[y * 2 + 1];
        
        for (x = 0; x < dest_width; x++) {
            
            // read first column
            r = *in_row1;    
            r += *in_row2;
            in_row1++;
            in_row2++;
            g = *in_row1;        
            g += *in_row2;
            in_row1++;
            in_row2++;
            b = *in_row1;        
            b += *in_row2;
            in_row1++;
            in_row2++;            
            
            if (src_has_alpha)
            {
                a = *in_row1;        
                a += *in_row2;
                in_row1++;
                in_row2++;
            }
            
            // read second column 
            r += *in_row1;        
            r += *in_row2;
            in_row1++;
            in_row2++;
            g += *in_row1;        
            g += *in_row2;
            in_row1++;
            in_row2++;
            b += *in_row1;        
            b += *in_row2;
            in_row1++;
            in_row2++;
            
            if (src_has_alpha)
            {
                a += *in_row1;        
                a += *in_row2;
                in_row1++;
                in_row2++;
            }
            
            // write blended color            
            *out = (UINT8)(r >> 2);
            out++;
            *out = (UINT8)(g >> 2);
            out++;
            *out = (UINT8)(b >> 2);
            out++;
            
            if (dest_has_alpha)
            {
                *out = (UINT8)(a >> 2);
                out++;
            }
        }
    }
    
    return dest;
}

/* wraps resize_half so it can be called directly from python */
PyObject *
resize_half_wrap(PyObject *self, PyObject *args)
{
    /* raw input python variables */
    PyObject *dest, *src;
    /* return value: dest image on success */
    PyObject *ret;
    
    if (!PyArg_ParseTuple(args, "OO", &dest, &src))
        return NULL;
    
    ret = resize_half(dest, src);
    if (ret == dest) {
        /* Python needs us to own our return value */
        Py_INCREF(dest);
    }
    return ret;
}
