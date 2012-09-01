#include "oil.h"
#include "oil-private.h"
#include <stdio.h>

int main(int argc, char **argv) {
    OILImage *im = oil_image_load("input.png");
    if (!im) {
        printf("failed to load\n");
        return 1;
    }
    
    OILPalette *pal = oil_palette_median_cut(im, 256);
    if (!pal) {
        printf("failed to palette\n");
        oil_image_free(im);
        return 1;
    }
    
    unsigned int i;
    OILImage *palim = oil_image_new(pal->size, 1);
    OILPixel *data = oil_image_lock(palim);
    for (i = 0; i < pal->size; i++) {
        printf("color %i: (%i, %i, %i, %i)\n", i, pal->table[i].r, pal->table[i].g, pal->table[i].b, pal->table[i].a);
        data[i] = pal->table[i];
    }
    oil_image_unlock(palim);
    
    oil_image_save(palim, "palette.png");
    oil_image_free(palim);
    oil_palette_free(pal);
    oil_image_free(im);
    
    printf("success!\n");
    return 0;
}
