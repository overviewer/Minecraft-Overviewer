#include "oil-private.h"

#include <string.h>

/* represent a 4d bounding box.
   for instance, all r such that lr <= r < ur
   and associated metadata for median-cut */
typedef struct {
    unsigned int lr, lg, lb, la;
    unsigned int ur, ug, ub, ua;
    
    /* dimension of the longest side (0=r, 1=g, 2=b, 3=a) */
    unsigned char longest_side;
    /* length of that side */
    unsigned int longest_side_length;
    /* volume of this box */
    unsigned int volume;
    /* number of pixels in this box */
    unsigned int pixels;
    /* histogram of pixels along longest axis */
    unsigned int *histogram;
    /* average color inside the box */
    OILPixel average;
} MedianCutBox;

/* linked list of boxes, used for a queue */
typedef struct _MedianCutBoxList MedianCutBoxList;
struct _MedianCutBoxList {
    MedianCutBox *box;
    MedianCutBoxList *next;
};

/* the queue made of that linked list */
typedef struct {
    MedianCutBoxList *first;
    MedianCutBoxList *last;
    unsigned int length;
} BoxQueue;

/* pop a box from the front of the list */
static inline MedianCutBox *oil_box_queue_pop(BoxQueue *q) {
    if (q->first) {
        MedianCutBox *box = q->first->box;
        MedianCutBoxList *tmp = q->first;
        q->first = q->first->next;
        free(tmp);
        
        if (q->first == NULL)
            q->last = NULL;
        q->length--;
        return box;
    }
    return NULL;
}

/* push a box on to the end */
static inline void oil_box_queue_push(BoxQueue *q, MedianCutBox *box) {
    MedianCutBoxList *cell = malloc(sizeof(MedianCutBoxList));
    cell->box = box;
    cell->next = NULL;
    if (q->last) {
        q->last->next = cell;
        q->last = cell;
    } else {
        q->first = q->last = cell;
    }
    q->length++;
}

/* find the box most fit to be split */
static inline MedianCutBox *oil_box_queue_find_best(BoxQueue *q) {
    MedianCutBox *box = NULL;
    unsigned int box_score = 0;
    MedianCutBoxList *cell;
    
    for (cell = q->first; cell != NULL; cell = cell->next) {
        /* refuse to split volume-1 boxes */
        if (cell->box->volume == 1)
            continue;
        
        if (cell->box->volume > box_score) {
            box = cell->box;
            box_score = cell->box->volume;
        }
    }
    
    return box;
}

/* remove the given box from the queue */
static inline void oil_box_queue_remove(BoxQueue *q, MedianCutBox *box) {
    MedianCutBoxList *cell = q->first;
    MedianCutBoxList *last = NULL;
    MedianCutBoxList **lastlink = &(q->first);
    while (cell) {
        if (cell->box == box) {
            if (q->last == cell)
                q->last = last;
            *lastlink = cell->next;
            free(cell);
            
            q->length--;
            return;
        }
        
        last = cell;
        cell = cell->next;
        lastlink = &(last->next);
    }
}

/* helper for creating and populating a box */
static inline MedianCutBox *oil_median_cut_box_new(unsigned int width, unsigned int height, const OILPixel *data, unsigned int lr, unsigned int lg, unsigned int lb, unsigned int la, unsigned int ur, unsigned int ug, unsigned int ub, unsigned int ua) {
    /* new, calculated values for lg, ug, etc.
       the initializers are swapped on purpose! see below! */
    unsigned int nlr = ur, nlg = ug, nlb = ub, nla = ua;
    unsigned int nur = lr, nug = lg, nub = lb, nua = la;
    /* the length of the box sides */
    unsigned int sr, sg, sb, sa;
    /* temprorary histograms for all 4 axes */
    unsigned int *histr, *histg, *histb, *hista;
    /* accumulators for average color */
    unsigned long ar = 0, ag = 0, ab = 0, aa = 0;
    /* iteration variables */
    unsigned int x, y;
    
    MedianCutBox *box = malloc(sizeof(MedianCutBox));
    
    /* temporary box sizes for the temporary histograms */
    sr = ur - lr;
    sg = ug - lg;
    sb = ub - lb;
    sa = ua - la;
    
    /* and the histograms proper */
    histr = malloc(sizeof(*histr) * sr);
    histg = malloc(sizeof(*histg) * sg);
    histb = malloc(sizeof(*histb) * sb);
    hista = malloc(sizeof(*hista) * sa);
    
    /* bail now if we have no memory */
    if (!box || !histr || !histg || !histb || !hista) {
        if (box)
            free(box);
        if (histr)
            free(histr);
        if (histg)
            free(histg);
        if (histb)
            free(histb);
        if (hista)
            free(hista);
        return NULL;
    }
    
    /* first, we must calculate n[lu][rgba], and the 4 histograms
       while we're at it, count pixels and do averages */
    box->pixels = 0;
    for (x = 0; x < width; x++) {
        for (y = 0; y < height; y++) {
            OILPixel p = data[y * width + x];
            /* check for membership! */
            if (p.r >= lr && p.r < ur &&
                p.g >= lg && p.g < ug &&
                p.b >= lb && p.b < ub &&
                p.a >= la && p.a < ua) {
                
                /* averages */
                ar += p.r;
                ag += p.g;
                ab += p.b;
                aa += p.a;
                
                /* pixel count */
                box->pixels++;
                
                /* histograms */
                histr[p.r - lr]++;
                histg[p.g - lg]++;
                histb[p.b - lb]++;
                hista[p.a - la]++;

                /* shrinkwrap! */
                if (p.r < nlr)
                    nlr = p.r;
                if (p.r >= nur)
                    nur = p.r + 1;
                
                if (p.g < nlg)
                    nlg = p.g;
                if (p.g >= nug)
                    nug = p.g + 1;
                
                if (p.b < nlb)
                    nlb = p.b;
                if (p.b >= nub)
                    nub = p.b + 1;
                
                if (p.a < nla)
                    nla = p.a;
                if (p.a >= nua)
                    nua = p.a + 1;
            }
        }
    }
    
    /* refuse to create an empty box */
    if (box->pixels == 0) {
        free(histr);
        free(histg);
        free(histb);
        free(hista);
        free(box);
        return NULL;
    }
    
    /* basic copying */
    box->lr = nlr;
    box->lg = nlg;
    box->lb = nlb;
    box->la = nla;
    box->ur = nur;
    box->ug = nug;
    box->ub = nub;
    box->ua = nua;
    
    /* find the side lengths, for real this time */
    sr = nur - nlr;
    sg = nug - nlg;
    sb = nub - nlb;
    sa = nua - nla;
    
    /* now, set the volume */
    box->volume = sr * sg * sb * sa;
    
    if (sr >= OIL_MAX(sa, OIL_MAX(sg, sb))) {
        /* red longest */
        box->longest_side = 0;
        box->longest_side_length = sr;
        box->histogram = calloc(sizeof(*(box->histogram)) * sr, 1);
        memcpy(box->histogram, &(histr[nlr - lr]), sizeof(*histr) * sr);
    } else if (sg >= OIL_MAX(sa, OIL_MAX(sr, sb))) {
        /* green longest */
        box->longest_side = 1;
        box->longest_side_length = sg;
        box->histogram = calloc(sizeof(*(box->histogram)) * sg, 1);
        memcpy(box->histogram, &(histg[nlg - lg]), sizeof(*histg) * sg);
    } else if (sb >= OIL_MAX(sa, OIL_MAX(sr, sg))) {
        /* blue longest */
        box->longest_side = 2;
        box->longest_side_length = sb;
        box->histogram = calloc(sizeof(*(box->histogram)) * sb, 1);
        memcpy(box->histogram, &(histb[nlb - lb]), sizeof(*histb) * sb);
    } else {
        /* alpha longest */
        box->longest_side = 3;
        box->longest_side_length = sa;
        box->histogram = calloc(sizeof(*(box->histogram)) * sa, 1);
        memcpy(box->histogram, &(hista[nla - la]), sizeof(*hista) * sa);
    }
    
    /* set the average */
    box->average.r = ar / box->pixels;
    box->average.g = ag / box->pixels;
    box->average.b = ab / box->pixels;
    box->average.a = aa / box->pixels;
    
    /* free our temporary histograms */
    free(histr);
    free(histg);
    free(histb);
    free(hista);
    
    return box;
}

/* helper for freeing a box */
static inline void oil_median_cut_box_free(MedianCutBox *box) {
    free(box->histogram);
    free(box);
}

OILPalette *oil_palette_median_cut(OILImage *im, unsigned int size) {
    unsigned int width, height;
    const OILPixel *data;
    BoxQueue queue = {NULL, NULL, 0};
    OILPalette *pal;
    
    if (!im || size == 0)
        return NULL;
    
    oil_image_get_size(im, &width, &height);
    data = oil_image_get_data(im);
    
    /* start out with a box covering all colors */
    oil_box_queue_push(&queue, oil_median_cut_box_new(width, height, data, 0, 0, 0, 0, 256, 256, 256, 256));
    
    while (queue.length < size) {
        MedianCutBox *box;
        MedianCutBox *a, *b;
        MedianCutBoxList *cell;
        unsigned int i, accum;

        /* choose a box to operate on */
        box = oil_box_queue_find_best(&queue);
        
        /* if we didn't find any suitable boxes, we're done */
        if (!box)
            break;
        
        /* we did find a box, remove it from the queue */
        oil_box_queue_remove(&queue, box);
        
        /* operate on this box, first find the median */
        accum = 0;
        for (i = 0; i < box->longest_side_length; i++) {
            accum += box->histogram[i];
            if (accum >= box->pixels / 2) {
                /* if the left side has less pixels than the right,
                   give the current bin to the left */
                if (accum - box->histogram[i] < box->pixels - accum)
                    i++;
                break;
            }
        }
        
        /* i now contains the median along the longest side
           so split! */
        switch (box->longest_side) {
        case 0:
            a = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb, box->la,
                                       box->lr + i, box->ug, box->ub, box->ua);
            b = oil_median_cut_box_new(width, height, data,
                                       box->lr + i, box->lg, box->lb, box->la,
                                       box->ur, box->ug, box->ub, box->ua);
            break;
        case 1:
            a = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb, box->la,
                                       box->ur, box->lg + i, box->ub, box->ua);
            b = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg + i, box->lb, box->la,
                                       box->ur, box->ug, box->ub, box->ua);
            break;
        case 2:
            a = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb, box->la,
                                       box->ur, box->ug, box->lb + i, box->ua);
            b = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb + i, box->la,
                                       box->ur, box->ug, box->ub, box->ua);
            break;
        case 3:
            a = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb, box->la,
                                       box->ur, box->ug, box->ub, box->la + i);
            b = oil_median_cut_box_new(width, height, data,
                                       box->lr, box->lg, box->lb, box->la + i,
                                       box->ur, box->ug, box->ub, box->ua);
            break;
        }
        
        /* we're done with the box, so free it */
        oil_median_cut_box_free(box);
        
        /* now we add a and b */
        oil_box_queue_push(&queue, a);
        oil_box_queue_push(&queue, b);
    }
    
    /* ooookay, we have all our boxes present,
       now to turn that into a palette */
    pal = malloc(sizeof(OILPalette));
    pal->size = queue.length;
    pal->table = malloc(sizeof(OILPixel) * queue.length);
    
    while (queue.length > 0) {
        MedianCutBox *box = oil_box_queue_pop(&queue);
        pal->table[queue.length] = box->average;
        oil_median_cut_box_free(box);
    }
    
    return pal;
}

void oil_palette_free(OILPalette *p) {
    if (p) {
        if (p->table)
            free(p->table);
        free(p);
    }
}
