#include "oil-private.h"

/* container for pixel errors */
typedef struct {
    int r, g, b, a;
} OILPixelError;

/* helper to find the nearest match for a pixel in a palette */
static inline unsigned char oil_dither_find(const OILPixel p, OILPalette *pal) {
    unsigned int i;
    unsigned int ret = 0;
    unsigned long ret_dist = 256 * 256 * 4;
    
    for (i = 0; i < pal->size; i++) {
        OILPixel c = pal->table[i];
        unsigned long dist = 0;
        
        dist += (c.r - p.r) * (c.r - p.r);
        dist += (c.g - p.g) * (c.g - p.g);
        dist += (c.b - p.b) * (c.b - p.b);
        dist += (c.a - p.a) * (c.a - p.a);
        
        if (dist == 0) {
            return i;
        } else if (dist < ret_dist) {
            ret = i;
            ret_dist = dist;
        }
    }
    
    return ret;
}

unsigned char *oil_dither_nearest(OILImage *im, OILPalette *pal) {
    unsigned int width, height;
    unsigned int x, y;
    const OILPixel *data;
    unsigned char *dithered;
    
    if (!im || !pal || pal->size == 0)
        return NULL;
    
    oil_image_get_size(im, &width, &height);
    data = oil_image_get_data(im);
    if (width == 0 || height == 0 || data == NULL)
        return NULL;
    
    dithered = malloc(sizeof(unsigned char) * width * height);
    if (!dithered)
        return NULL;
    
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            dithered[y * width + x] = oil_dither_find(data[y * width + x], pal);
        }
    }
    
    return dithered;
}

unsigned char *oil_dither_floyd_steinberg(OILImage *im, OILPalette *pal) {
    unsigned int width, height;
    unsigned int x, y;
    const OILPixel *data;
    OILPixelError *error;
    unsigned char *dithered;
    
    if (!im || !pal || pal->size == 0)
        return NULL;
    
    oil_image_get_size(im, &width, &height);
    data = oil_image_get_data(im);
    if (width == 0 || height == 0 || data == NULL)
        return NULL;
    
    error = calloc(width * height, sizeof(OILPixelError));
    if (!error)
        return NULL;
    
    dithered = malloc(sizeof(unsigned char) * width * height);
    if (!dithered) {
        free(error);
        return NULL;
    }
    
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            /* current image pixel */
            OILPixel p = data[y * width + x];
            /* current error */
            OILPixelError pe = error[y * width + x];
            
            /* palette index and color */
            unsigned char i;
            OILPixel pi;
            
            /* add on the previous error */
            p.r += OIL_CLAMP(pe.r, -p.r, 255 - p.r);
            p.g += OIL_CLAMP(pe.g, -p.g, 255 - p.g);
            p.b += OIL_CLAMP(pe.b, -p.b, 255 - p.b);
            p.a += OIL_CLAMP(pe.a, -p.a, 255 - p.a);
            
            /* find the nearest palette color and use it */
            i = oil_dither_find(p, pal);
            dithered[y * width + x] = i;
            pi = pal->table[i];
            
            /* calculate the error incurred */
            pe.r = p.r - pi.r;
            pe.g = p.g - pi.g;
            pe.b = p.b - pi.b;
            pe.a = p.a - pi.a;
            
            /* now distribute the error forward */
            
            if (x + 1 < width) {
                /* x+1, y weight 7/16 */
                error[y * width + x + 1].r = pe.r * 7 / 16;
                error[y * width + x + 1].g = pe.g * 7 / 16;
                error[y * width + x + 1].b = pe.b * 7 / 16;
                error[y * width + x + 1].a = pe.a * 7 / 16;
                
                if (y + 1 < height) {
                    /* x+1, y+1 weight 1/16 */
                    error[(y + 1) * width + x + 1].r = pe.r / 16;
                    error[(y + 1) * width + x + 1].g = pe.g / 16;
                    error[(y + 1) * width + x + 1].b = pe.b / 16;
                    error[(y + 1) * width + x + 1].a = pe.a / 16;
                }
            }
            
            if (y + 1 < height) {
                if (x >= 1) {
                    /* x-1, y+1 weight 3/16 */
                    error[(y + 1) * width + x - 1].r = pe.r * 3 / 16;
                    error[(y + 1) * width + x - 1].g = pe.g * 3 / 16;
                    error[(y + 1) * width + x - 1].b = pe.b * 3 / 16;
                    error[(y + 1) * width + x - 1].a = pe.a * 3 / 16;
                }
                
                /* x, y+1 weight 5/16 */
                error[(y + 1) * width + x].r = pe.r * 5 / 16;
                error[(y + 1) * width + x].g = pe.g * 5 / 16;
                error[(y + 1) * width + x].b = pe.b * 5 / 16;
                error[(y + 1) * width + x].a = pe.a * 5 / 16;
            }
        }
    }
    
    free(error);
    return dithered;
}
