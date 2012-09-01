#include "oil-private.h"

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
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            dithered[y * width + x] = oil_dither_find(data[y * width + x], pal);
        }
    }
    
    return dithered;
}
