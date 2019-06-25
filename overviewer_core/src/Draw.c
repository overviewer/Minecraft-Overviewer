/*
 * The Python Imaging Library.
 * $Id$
 *
 * a simple drawing package for the Imaging library
 *
 * history:
 * 1996-04-13 fl  Created.
 * 1996-04-30 fl  Added transforms and polygon support.
 * 1996-08-12 fl  Added filled polygons.
 * 1996-11-05 fl  Fixed float/int confusion in polygon filler
 * 1997-07-04 fl  Support 32-bit images (C++ would have been nice)
 * 1998-09-09 fl  Eliminated qsort casts; improved rectangle clipping
 * 1998-09-10 fl  Fixed fill rectangle to include lower edge (!)
 * 1998-12-29 fl  Added arc, chord, and pieslice primitives
 * 1999-01-10 fl  Added some level 2 ("arrow") stuff (experimental)
 * 1999-02-06 fl  Added bitmap primitive
 * 1999-07-26 fl  Eliminated a compiler warning
 * 1999-07-31 fl  Pass ink as void* instead of int
 * 2002-12-10 fl  Added experimental RGBA-on-RGB drawing
 * 2004-09-04 fl  Support simple wide lines (no joins)
 * 2005-05-25 fl  Fixed line width calculation
 * 2011-04-01     Modified for use in Minecraft-Overviewer
 *
 * Copyright (c) 1996-2006 by Fredrik Lundh
 * Copyright (c) 1997-2006 by Secret Labs AB.
 *
 * This file is part of the Python Imaging Library
 *
 * By obtaining, using, and/or copying this software and/or its associated
 * documentation, you agree that you have read, understood, and will comply
 * with the following terms and conditions:
 *
 * Permission to use, copy, modify, and distribute this software and its
 * associated documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appears in all copies, and that
 * both that copyright notice and this permission notice appear in supporting
 * documentation, and that the name of Secret Labs AB or the author not be used
 * in advertising or publicity pertaining to distribution of the software
 * without specific, written prior permission.
 *
 * SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. 
 * IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
 * INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 */

/* FIXME: support fill/outline attribute for all filled shapes */
/* FIXME: support zero-winding fill */
/* FIXME: add drawing context, support affine transforms */
/* FIXME: support clip window (and mask?) */

#include "Imaging.h"
#include "utils.h"

#include <math.h>

#define CEIL(v) (int)ceil(v)
#define FLOOR(v) ((v) >= 0.0 ? (int)(v) : (int)floor(v))

#define INK8(ink) (*(UINT8*)ink)
#define INK32(ink) (*(INT32*)ink)

/* -------------------------------------------------------------------- */
/* Primitives                                                           */
/* -------------------------------------------------------------------- */

typedef struct {
    /* edge descriptor for polygon engine */
    int32_t d;
    int32_t x0, y0;
    int32_t xmin, ymin, xmax, ymax;
    float dx;
} Edge;

static inline void
point8(Imaging im, int32_t x, int32_t y, int32_t ink) {
    if (x >= 0 && x < im->xsize && y >= 0 && y < im->ysize)
        im->image8[y][x] = (UINT8)ink;
}

static inline void
point32(Imaging im, int32_t x, int32_t y, int32_t ink) {
    if (x >= 0 && x < im->xsize && y >= 0 && y < im->ysize)
        im->image32[y][x] = ink;
}

static inline void
point32rgba(Imaging im, int32_t x, int32_t y, int32_t ink) {
    uint32_t tmp1, tmp2;

    if (x >= 0 && x < im->xsize && y >= 0 && y < im->ysize) {
        UINT8* out = (UINT8*)im->image[y] + x * 4;
        UINT8* in = (UINT8*)&ink;
        out[0] = OV_BLEND(in[3], out[0], in[0], tmp1, tmp2);
        out[1] = OV_BLEND(in[3], out[1], in[1], tmp1, tmp2);
        out[2] = OV_BLEND(in[3], out[2], in[2], tmp1, tmp2);
    }
}

static inline void
hline8(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t ink) {
    int32_t tmp;

    if (y0 >= 0 && y0 < im->ysize) {
        if (x0 > x1)
            tmp = x0, x0 = x1, x1 = tmp;
        if (x0 < 0)
            x0 = 0;
        else if (x0 >= im->xsize)
            return;
        if (x1 < 0)
            return;
        else if (x1 >= im->xsize)
            x1 = im->xsize - 1;
        if (x0 <= x1)
            memset(im->image8[y0] + x0, (UINT8)ink, x1 - x0 + 1);
    }
}

static inline void
hline32(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t ink) {
    int32_t tmp;
    INT32* p;

    if (y0 >= 0 && y0 < im->ysize) {
        if (x0 > x1)
            tmp = x0, x0 = x1, x1 = tmp;
        if (x0 < 0)
            x0 = 0;
        else if (x0 >= im->xsize)
            return;
        if (x1 < 0)
            return;
        else if (x1 >= im->xsize)
            x1 = im->xsize - 1;
        p = im->image32[y0];
        while (x0 <= x1)
            p[x0++] = ink;
    }
}

static inline void
hline32rgba(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t ink) {
    int32_t tmp;
    uint32_t tmp1, tmp2;

    if (y0 >= 0 && y0 < im->ysize) {
        if (x0 > x1)
            tmp = x0, x0 = x1, x1 = tmp;
        if (x0 < 0)
            x0 = 0;
        else if (x0 >= im->xsize)
            return;
        if (x1 < 0)
            return;
        else if (x1 >= im->xsize)
            x1 = im->xsize - 1;
        if (x0 <= x1) {
            UINT8* out = (UINT8*)im->image[y0] + x0 * 4;
            UINT8* in = (UINT8*)&ink;
            while (x0 <= x1) {
                out[0] = OV_BLEND(in[3], out[0], in[0], tmp1, tmp2);
                out[1] = OV_BLEND(in[3], out[1], in[1], tmp1, tmp2);
                out[2] = OV_BLEND(in[3], out[2], in[2], tmp1, tmp2);
                x0++;
                out += 4;
            }
        }
    }
}

static inline void
line8(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t ink) {
    int32_t i, n, e;
    int32_t dx, dy;
    int32_t xs, ys;

    /* normalize coordinates */
    dx = x1 - x0;
    if (dx < 0)
        dx = -dx, xs = -1;
    else
        xs = 1;
    dy = y1 - y0;
    if (dy < 0)
        dy = -dy, ys = -1;
    else
        ys = 1;

    n = (dx > dy) ? dx : dy;

    if (dx == 0)

        /* vertical */
        for (i = 0; i < dy; i++) {
            point8(im, x0, y0, ink);
            y0 += ys;
        }

    else if (dy == 0)

        /* horizontal */
        for (i = 0; i < dx; i++) {
            point8(im, x0, y0, ink);
            x0 += xs;
        }

    else if (dx > dy) {

        /* bresenham, horizontal slope */
        n = dx;
        dy += dy;
        e = dy - dx;
        dx += dx;

        for (i = 0; i < n; i++) {
            point8(im, x0, y0, ink);
            if (e >= 0) {
                y0 += ys;
                e -= dx;
            }
            e += dy;
            x0 += xs;
        }

    } else {

        /* bresenham, vertical slope */
        n = dy;
        dx += dx;
        e = dx - dy;
        dy += dy;

        for (i = 0; i < n; i++) {
            point8(im, x0, y0, ink);
            if (e >= 0) {
                x0 += xs;
                e -= dy;
            }
            e += dx;
            y0 += ys;
        }
    }
}

static inline void
line32(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t ink) {
    int32_t i, n, e;
    int32_t dx, dy;
    int32_t xs, ys;

    /* normalize coordinates */
    dx = x1 - x0;
    if (dx < 0)
        dx = -dx, xs = -1;
    else
        xs = 1;
    dy = y1 - y0;
    if (dy < 0)
        dy = -dy, ys = -1;
    else
        ys = 1;

    n = (dx > dy) ? dx : dy;

    if (dx == 0)

        /* vertical */
        for (i = 0; i < dy; i++) {
            point32(im, x0, y0, ink);
            y0 += ys;
        }

    else if (dy == 0)

        /* horizontal */
        for (i = 0; i < dx; i++) {
            point32(im, x0, y0, ink);
            x0 += xs;
        }

    else if (dx > dy) {

        /* bresenham, horizontal slope */
        n = dx;
        dy += dy;
        e = dy - dx;
        dx += dx;

        for (i = 0; i < n; i++) {
            point32(im, x0, y0, ink);
            if (e >= 0) {
                y0 += ys;
                e -= dx;
            }
            e += dy;
            x0 += xs;
        }

    } else {

        /* bresenham, vertical slope */
        n = dy;
        dx += dx;
        e = dx - dy;
        dy += dy;

        for (i = 0; i < n; i++) {
            point32(im, x0, y0, ink);
            if (e >= 0) {
                x0 += xs;
                e -= dy;
            }
            e += dx;
            y0 += ys;
        }
    }
}

static inline void
line32rgba(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t ink) {
    int32_t i, n, e;
    int32_t dx, dy;
    int32_t xs, ys;

    /* normalize coordinates */
    dx = x1 - x0;
    if (dx < 0)
        dx = -dx, xs = -1;
    else
        xs = 1;
    dy = y1 - y0;
    if (dy < 0)
        dy = -dy, ys = -1;
    else
        ys = 1;

    n = (dx > dy) ? dx : dy;

    if (dx == 0)

        /* vertical */
        for (i = 0; i < dy; i++) {
            point32rgba(im, x0, y0, ink);
            y0 += ys;
        }

    else if (dy == 0)

        /* horizontal */
        for (i = 0; i < dx; i++) {
            point32rgba(im, x0, y0, ink);
            x0 += xs;
        }

    else if (dx > dy) {

        /* bresenham, horizontal slope */
        n = dx;
        dy += dy;
        e = dy - dx;
        dx += dx;

        for (i = 0; i < n; i++) {
            point32rgba(im, x0, y0, ink);
            if (e >= 0) {
                y0 += ys;
                e -= dx;
            }
            e += dy;
            x0 += xs;
        }

    } else {

        /* bresenham, vertical slope */
        n = dy;
        dx += dx;
        e = dx - dy;
        dy += dy;

        for (i = 0; i < n; i++) {
            point32rgba(im, x0, y0, ink);
            if (e >= 0) {
                x0 += xs;
                e -= dy;
            }
            e += dx;
            y0 += ys;
        }
    }
}

static int32_t
x_cmp(const void* x0, const void* x1) {
    float diff = *((float*)x0) - *((float*)x1);
    if (diff < 0)
        return -1;
    else if (diff > 0)
        return 1;
    else
        return 0;
}

static inline int32_t
polygon8(Imaging im, int32_t n, Edge* e, int32_t ink, int32_t eofill) {
    int32_t i, j;
    float* xx;
    int32_t ymin, ymax;
    float y;

    if (n <= 0)
        return 0;

    /* Find upper and lower polygon boundary (within image) */

    ymin = e[0].ymin;
    ymax = e[0].ymax;
    for (i = 1; i < n; i++) {
        if (e[i].ymin < ymin)
            ymin = e[i].ymin;
        if (e[i].ymax > ymax)
            ymax = e[i].ymax;
    }

    if (ymin < 0)
        ymin = 0;
    if (ymax >= im->ysize)
        ymax = im->ysize - 1;

    /* Process polygon edges */

    xx = malloc(n * sizeof(float));
    if (!xx)
        return -1;

    for (; ymin <= ymax; ymin++) {
        y = ymin + 0.5F;
        for (i = j = 0; i < n; i++)
            if (y >= e[i].ymin && y <= e[i].ymax) {
                if (e[i].d == 0)
                    hline8(im, e[i].xmin, ymin, e[i].xmax, ink);
                else
                    xx[j++] = (y - e[i].y0) * e[i].dx + e[i].x0;
            }
        if (j == 2) {
            if (xx[0] < xx[1])
                hline8(im, CEIL(xx[0] - 0.5), ymin, FLOOR(xx[1] + 0.5), ink);
            else
                hline8(im, CEIL(xx[1] - 0.5), ymin, FLOOR(xx[0] + 0.5), ink);
        } else {
            qsort(xx, j, sizeof(float), x_cmp);
            for (i = 0; i < j - 1; i += 2)
                hline8(im, CEIL(xx[i] - 0.5), ymin, FLOOR(xx[i + 1] + 0.5), ink);
        }
    }

    free(xx);

    return 0;
}

static inline int32_t
polygon32(Imaging im, int32_t n, Edge* e, int32_t ink, int32_t eofill) {
    int32_t i, j;
    float* xx;
    int32_t ymin, ymax;
    float y;

    if (n <= 0)
        return 0;

    /* Find upper and lower polygon boundary (within image) */

    ymin = e[0].ymin;
    ymax = e[0].ymax;
    for (i = 1; i < n; i++) {
        if (e[i].ymin < ymin)
            ymin = e[i].ymin;
        if (e[i].ymax > ymax)
            ymax = e[i].ymax;
    }

    if (ymin < 0)
        ymin = 0;
    if (ymax >= im->ysize)
        ymax = im->ysize - 1;

    /* Process polygon edges */

    xx = malloc(n * sizeof(float));
    if (!xx)
        return -1;

    for (; ymin <= ymax; ymin++) {
        y = ymin + 0.5F;
        for (i = j = 0; i < n; i++) {
            if (y >= e[i].ymin && y <= e[i].ymax) {
                if (e[i].d == 0)
                    hline32(im, e[i].xmin, ymin, e[i].xmax, ink);
                else
                    xx[j++] = (y - e[i].y0) * e[i].dx + e[i].x0;
            }
        }
        if (j == 2) {
            if (xx[0] < xx[1])
                hline32(im, CEIL(xx[0] - 0.5), ymin, FLOOR(xx[1] + 0.5), ink);
            else
                hline32(im, CEIL(xx[1] - 0.5), ymin, FLOOR(xx[0] + 0.5), ink);
        } else {
            qsort(xx, j, sizeof(float), x_cmp);
            for (i = 0; i < j - 1; i += 2)
                hline32(im, CEIL(xx[i] - 0.5), ymin, FLOOR(xx[i + 1] + 0.5), ink);
        }
    }

    free(xx);

    return 0;
}

static inline int32_t
polygon32rgba(Imaging im, int32_t n, Edge* e, int32_t ink, int32_t eofill) {
    int32_t i, j;
    float* xx;
    int32_t ymin, ymax;
    float y;

    if (n <= 0)
        return 0;

    /* Find upper and lower polygon boundary (within image) */

    ymin = e[0].ymin;
    ymax = e[0].ymax;
    for (i = 1; i < n; i++) {
        if (e[i].ymin < ymin)
            ymin = e[i].ymin;
        if (e[i].ymax > ymax)
            ymax = e[i].ymax;
    }

    if (ymin < 0)
        ymin = 0;
    if (ymax >= im->ysize)
        ymax = im->ysize - 1;

    /* Process polygon edges */

    xx = malloc(n * sizeof(float));
    if (!xx)
        return -1;

    for (; ymin <= ymax; ymin++) {
        y = ymin + 0.5F;
        for (i = j = 0; i < n; i++) {
            if (y >= e[i].ymin && y <= e[i].ymax) {
                if (e[i].d == 0)
                    hline32rgba(im, e[i].xmin, ymin, e[i].xmax, ink);
                else
                    xx[j++] = (y - e[i].y0) * e[i].dx + e[i].x0;
            }
        }
        if (j == 2) {
            if (xx[0] < xx[1])
                hline32rgba(im, CEIL(xx[0] - 0.5), ymin, FLOOR(xx[1] + 0.5), ink);
            else
                hline32rgba(im, CEIL(xx[1] - 0.5), ymin, FLOOR(xx[0] + 0.5), ink);
        } else {
            qsort(xx, j, sizeof(float), x_cmp);
            for (i = 0; i < j - 1; i += 2)
                hline32rgba(im, CEIL(xx[i] - 0.5), ymin, FLOOR(xx[i + 1] + 0.5), ink);
        }
    }

    free(xx);

    return 0;
}

static inline void
add_edge(Edge* e, int32_t x0, int32_t y0, int32_t x1, int32_t y1) {
    /* printf("edge %d %d %d %d\n", x0, y0, x1, y1); */

    if (x0 <= x1)
        e->xmin = x0, e->xmax = x1;
    else
        e->xmin = x1, e->xmax = x0;

    if (y0 <= y1)
        e->ymin = y0, e->ymax = y1;
    else
        e->ymin = y1, e->ymax = y0;

    if (y0 == y1) {
        e->d = 0;
        e->dx = 0.0;
    } else {
        e->dx = ((float)(x1 - x0)) / (y1 - y0);
        if (y0 == e->ymin)
            e->d = 1;
        else
            e->d = -1;
    }

    e->x0 = x0;
    e->y0 = y0;
}

typedef struct {
    void (*point)(Imaging im, int32_t x, int32_t y, int32_t ink);
    void (*hline)(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t ink);
    void (*line)(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1, int32_t ink);
    int32_t (*polygon)(Imaging im, int32_t n, Edge* e, int32_t ink, int32_t eofill);
} DRAW;

DRAW draw8 = {point8, hline8, line8, polygon8};
DRAW draw32 = {point32, hline32, line32, polygon32};
DRAW draw32rgba = {point32rgba, hline32rgba, line32rgba, polygon32rgba};

/* -------------------------------------------------------------------- */
/* Interface                                                            */
/* -------------------------------------------------------------------- */

#define DRAWINIT()                           \
    if (im->image8) {                        \
        draw = &draw8;                       \
        ink = INK8(ink_);                    \
    } else {                                 \
        draw = (op) ? &draw32rgba : &draw32; \
        ink = INK32(ink_);                   \
    }

int32_t ImagingDrawPoint(Imaging im, int32_t x0, int32_t y0, const void* ink_, int32_t op) {
    DRAW* draw;
    INT32 ink;

    DRAWINIT();

    draw->point(im, x0, y0, ink);

    return 0;
}

int32_t ImagingDrawLine(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1, const void* ink_,
                        int32_t op) {
    DRAW* draw;
    INT32 ink;

    DRAWINIT();

    draw->line(im, x0, y0, x1, y1, ink);

    return 0;
}

int32_t ImagingDrawWideLine(Imaging im, int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                            const void* ink_, int32_t width, int32_t op) {
    DRAW* draw;
    INT32 ink;

    Edge e[4];

    int32_t dx, dy;
    double d;

    DRAWINIT();

    if (width <= 1) {
        draw->line(im, x0, y0, x1, y1, ink);
        return 0;
    }

    dx = x1 - x0;
    dy = y1 - y0;

    if (dx == 0 && dy == 0) {
        draw->point(im, x0, y0, ink);
        return 0;
    }

    d = width / sqrt((float)(dx * dx + dy * dy)) / 2.0;

    dx = (int)floor(d * (y1 - y0) + 0.5);
    dy = (int)floor(d * (x1 - x0) + 0.5);

    add_edge(e + 0, x0 - dx, y0 + dy, x1 - dx, y1 + dy);
    add_edge(e + 1, x1 - dx, y1 + dy, x1 + dx, y1 - dy);
    add_edge(e + 2, x1 + dx, y1 - dy, x0 + dx, y0 - dy);
    add_edge(e + 3, x0 + dx, y0 - dy, x0 - dx, y0 + dy);

    draw->polygon(im, 4, e, ink, 0);

    return 0;
}

/* -------------------------------------------------------------------- */
/* standard shapes */

#define ARC 0
#define CHORD 1
#define PIESLICE 2

/* -------------------------------------------------------------------- */

/* experimental level 2 ("arrow") graphics stuff.  this implements
   portions of the arrow api on top of the Edge structure.  the
   semantics are ok, except that "curve" flattens the bezier curves by
   itself */

#if 1 /* ARROW_GRAPHICS */

struct ImagingOutlineInstance {

    float x0, y0;

    float x, y;

    int32_t count;
    Edge* edges;

    int32_t size;
};

void ImagingOutlineDelete(ImagingOutline outline) {
    if (!outline)
        return;

    if (outline->edges)
        free(outline->edges);

    free(outline);
}

static Edge*
allocate(ImagingOutline outline, int32_t extra) {
    Edge* e;

    if (outline->count + extra > outline->size) {
        /* expand outline buffer */
        outline->size += extra + 25;
        if (!outline->edges)
            e = malloc(outline->size * sizeof(Edge));
        else
            e = realloc(outline->edges, outline->size * sizeof(Edge));
        if (!e)
            return NULL;
        outline->edges = e;
    }

    e = outline->edges + outline->count;

    outline->count += extra;

    return e;
}

int32_t ImagingOutlineMove(ImagingOutline outline, float x0, float y0) {
    outline->x = outline->x0 = x0;
    outline->y = outline->y0 = y0;

    return 0;
}

int32_t ImagingOutlineLine(ImagingOutline outline, float x1, float y1) {
    Edge* e;

    e = allocate(outline, 1);
    if (!e)
        return -1; /* out of memory */

    add_edge(e, (int)outline->x, (int)outline->y, (int)x1, (int)y1);

    outline->x = x1;
    outline->y = y1;

    return 0;
}

int32_t ImagingOutlineCurve(ImagingOutline outline, float x1, float y1,
                            float x2, float y2, float x3, float y3) {
    Edge* e;
    int32_t i;
    float xo, yo;

#define STEPS 32

    e = allocate(outline, STEPS);
    if (!e)
        return -1; /* out of memory */

    xo = outline->x;
    yo = outline->y;

    /* flatten the bezier segment */

    for (i = 1; i <= STEPS; i++) {

        float t = ((float)i) / STEPS;
        float t2 = t * t;
        float t3 = t2 * t;

        float u = 1.0F - t;
        float u2 = u * u;
        float u3 = u2 * u;

        float x = outline->x * u3 + 3 * (x1 * t * u2 + x2 * t2 * u) + x3 * t3 + 0.5;
        float y = outline->y * u3 + 3 * (y1 * t * u2 + y2 * t2 * u) + y3 * t3 + 0.5;

        add_edge(e++, xo, yo, (int)x, (int)y);

        xo = x, yo = y;
    }

    outline->x = xo;
    outline->y = yo;

    return 0;
}

int32_t ImagingOutlineCurve2(ImagingOutline outline, float cx, float cy,
                             float x3, float y3) {
    /* add bezier curve based on three control points (as
       in the Flash file format) */

    return ImagingOutlineCurve(
        outline,
        (outline->x + cx + cx) / 3, (outline->y + cy + cy) / 3,
        (cx + cx + x3) / 3, (cy + cy + y3) / 3,
        x3, y3);
}

int32_t ImagingOutlineClose(ImagingOutline outline) {
    if (outline->x == outline->x0 && outline->y == outline->y0)
        return 0;
    return ImagingOutlineLine(outline, outline->x0, outline->y0);
}

int32_t ImagingDrawOutline(Imaging im, ImagingOutline outline, const void* ink_,
                           int32_t fill, int32_t op) {
    DRAW* draw;
    INT32 ink;

    DRAWINIT();

    draw->polygon(im, outline->count, outline->edges, ink, 0);

    return 0;
}

#endif
