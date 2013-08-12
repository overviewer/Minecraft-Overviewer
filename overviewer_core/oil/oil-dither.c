#include "oil.h"
#include "oil-dither-private.h"

#include <math.h>

/* controls the number of bins for the color search table
   careful! much more than 3 splits results in mad memory usage */
#define COLOR_SEARCH_SPLITS 3
#define COLOR_SEARCH_BINS (1 << COLOR_SEARCH_SPLITS)
#define BIN_FOR_COLOR(val) ((val) >> (8 - COLOR_SEARCH_SPLITS))

/* container for pixel errors */
typedef struct {
    int r, g, b, a;
} OILPixelError;

/* a linked list of pixels with indexes */
typedef struct _PixelList PixelList;
struct _PixelList {
    unsigned int index;
    OILPixel pixel;
    PixelList *next;
};

/* used in ColorSearchTable, below */
typedef struct {
    int initialized;
    PixelList *colors;
} ColorSearchSubpalette;

/* a structure to optimize nearest-neighbor color searches */
typedef struct {
    OILPalette *palette;
    ColorSearchSubpalette subpalettes[COLOR_SEARCH_BINS][COLOR_SEARCH_BINS][COLOR_SEARCH_BINS][COLOR_SEARCH_BINS];
} ColorSearchTable;

/* helper to initialize a search table */
static inline void oil_color_search_init(ColorSearchTable *table, OILPalette *pal) {
    unsigned int r, g, b, a;
    table->palette = pal;
    for (r = 0; r < COLOR_SEARCH_BINS; r++) {
        for (g = 0; g < COLOR_SEARCH_BINS; g++) {
            for (b = 0; b < COLOR_SEARCH_BINS; b++) {
                for (a = 0; a < COLOR_SEARCH_BINS; a++) {
                    table->subpalettes[r][g][b][a].initialized = 0;
                    table->subpalettes[r][g][b][a].colors = NULL;
                }
            }
        }
    }
}

/* helper to free a search table */
static inline void oil_color_search_free(ColorSearchTable *table) {
    unsigned int r, g, b, a;
    for (r = 0; r < COLOR_SEARCH_BINS; r++) {
        for (g = 0; g < COLOR_SEARCH_BINS; g++) {
            for (b = 0; b < COLOR_SEARCH_BINS; b++) {
                for (a = 0; a < COLOR_SEARCH_BINS; a++) {
                    if (table->subpalettes[r][g][b][a].colors)
                        free(table->subpalettes[r][g][b][a].colors);
                }
            }
        }
    }
}

/* helper to find the nearest match for a pixel in a palette
   using the color search table */
static inline unsigned char oil_color_search_find(ColorSearchTable *table, const OILPixel p) {
    OILPalette *pal = table->palette;
    ColorSearchSubpalette *subpal;
    PixelList *cell;
    unsigned int ret = 0;
    unsigned long ret_dist = 256 * 256 * 4;
    
    /* first, find our subpalette */
    subpal = &(table->subpalettes[BIN_FOR_COLOR(p.r)][BIN_FOR_COLOR(p.g)][BIN_FOR_COLOR(p.b)][BIN_FOR_COLOR(p.a)]);
    
    if (!(subpal->initialized)) {
        /* we have to initialize this before we use it
           first, find the color closest to the center of this bin */
        unsigned int i;
        unsigned long shortest_dist = 256 * 256 * 4;
        double disttmp;
        OILPixel center;
        
        /* find the center first */
        center.r = (BIN_FOR_COLOR(p.r) * 256  + 128) / COLOR_SEARCH_BINS;
        center.g = (BIN_FOR_COLOR(p.g) * 256  + 128) / COLOR_SEARCH_BINS;
        center.b = (BIN_FOR_COLOR(p.b) * 256  + 128) / COLOR_SEARCH_BINS;
        center.a = (BIN_FOR_COLOR(p.a) * 256  + 128) / COLOR_SEARCH_BINS;
        
        /* find the nearest to center */
        for (i = 0; i < pal->size; i++) {
            unsigned long dist = 0;
            OILPixel c = pal->table[i];
            
            dist += (center.r - c.r) * (center.r - c.r);
            dist += (center.g - c.g) * (center.g - c.g);
            dist += (center.b - c.b) * (center.b - c.b);
            dist += (center.a - c.a) * (center.a - c.a);
            
            if (dist < shortest_dist) {
                shortest_dist = dist;
                if (dist == 0)
                    break;
            }
        }
        
        /* okay, now we need to add on 2 * sqrt(2) * box_width to this
           
           so, all colors closer than this to the center apply to this
           box (do the math yourself! triangle inequality! go!
        */
        disttmp = sqrt(shortest_dist);
        disttmp += 2 * sqrt(2) * (128 / COLOR_SEARCH_BINS);
        shortest_dist = (disttmp * disttmp) + 1;
        
        /* now, finally, we re-iterate and throw together our subpalette */
        for (i = 0; i < pal->size; i++) {
            unsigned long dist = 0;
            OILPixel c = pal->table[i];
            
            dist += (center.r - c.r) * (center.r - c.r);
            dist += (center.g - c.g) * (center.g - c.g);
            dist += (center.b - c.b) * (center.b - c.b);
            dist += (center.a - c.a) * (center.a - c.a);
            
            if (dist <= shortest_dist) {
                PixelList *newcell = malloc(sizeof(PixelList));
                newcell->index = i;
                newcell->pixel = c;
                newcell->next = subpal->colors;
                subpal->colors = newcell;
            }
        }
        
        subpal->initialized = 1;
    }
    
    /* continue on with our search */
    for (cell = subpal->colors; cell != NULL; cell = cell->next) {
        OILPixel c = cell->pixel;
        unsigned long dist = 0;
        
        dist += (c.r - p.r) * (c.r - p.r);
        dist += (c.g - p.g) * (c.g - p.g);
        dist += (c.b - p.b) * (c.b - p.b);
        dist += (c.a - p.a) * (c.a - p.a);
        
        if (dist == 0) {
            return cell->index;
        } else if (dist < ret_dist) {
            ret = cell->index;
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
    ColorSearchTable table;
    
    if (!im || !pal || pal->size == 0)
        return NULL;
    
    oil_image_get_size(im, &width, &height);
    data = oil_image_get_data(im);
    if (width == 0 || height == 0 || data == NULL)
        return NULL;
    
    dithered = malloc(sizeof(unsigned char) * width * height);
    if (!dithered)
        return NULL;
    
    oil_color_search_init(&table, pal);
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            dithered[y * width + x] = oil_color_search_find(&table, data[y * width + x]);
        }
    }
    oil_color_search_free(&table);
    
    return dithered;
}

unsigned char *oil_dither_floyd_steinberg(OILImage *im, OILPalette *pal) {
    unsigned int width, height;
    unsigned int x, y;
    const OILPixel *data;
    OILPixelError *error;
    unsigned char *dithered;
    ColorSearchTable table;
    
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
    
    oil_color_search_init(&table, pal);
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
            i = oil_color_search_find(&table, p);
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
    oil_color_search_free(&table);
    
    free(error);
    return dithered;
}
