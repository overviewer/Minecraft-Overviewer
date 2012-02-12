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
 * This is a general include file for the Overviewer C extension. It
 * lists useful, defined functions as well as those that are exported
 * to python, so all files can use them.
 */

#ifndef __OVERVIEWER_H_INCLUDED__
#define __OVERVIEWER_H_INCLUDED__

// increment this value if you've made a change to the c extesion
// and want to force users to rebuild
#define OVERVIEWER_EXTENSION_VERSION 20

/* Python PIL, and numpy headers */
#include <Python.h>
#include <Imaging.h>
#include <numpy/arrayobject.h>

/* macro for getting a value out of various numpy arrays */
#define getArrayByte3D(array, x,y,z) (*(unsigned char *)(PyArray_GETPTR3((array), (x), (y), (z))))
#define getArrayShort2D(array, x,y) (*(unsigned short *)(PyArray_GETPTR2((array), (x), (y))))

/* generally useful MAX / MIN macros */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* in composite.c */
Imaging imaging_python_to_c(PyObject *obj);
PyObject *alpha_over(PyObject *dest, PyObject *src, PyObject *mask,
                     int dx, int dy, int xsize, int ysize);
PyObject *alpha_over_full(PyObject *dest, PyObject *src, PyObject *mask, float overall_alpha,
                          int dx, int dy, int xsize, int ysize);
PyObject *alpha_over_wrap(PyObject *self, PyObject *args);
PyObject *tint_with_mask(PyObject *dest, unsigned char sr, unsigned char sg,
                         unsigned char sb, unsigned char sa,
                         PyObject *mask, int dx, int dy, int xsize, int ysize);
PyObject *draw_triangle(PyObject *dest, int inclusive,
                        int x0, int y0,
                        unsigned char r0, unsigned char g0, unsigned char b0,
                        int x1, int y1,
                        unsigned char r1, unsigned char g1, unsigned char b1,
                        int x2, int y2,
                        unsigned char r2, unsigned char g2, unsigned char b2,
                        int tux, int tuy, int *touchups, unsigned int num_touchups);

/* forward declaration of RenderMode object */
typedef struct _RenderMode RenderMode;

/* in iterate.c */
typedef struct {
    /* the regionset object, and chunk coords */
    PyObject *regionset;
    int chunkx, chunkz;
    
    /* the tile image and destination */
    PyObject *img;
    int imgx, imgy;
    
    /* the current render mode in use */
    RenderMode *rendermode;
    
    /* the Texture object */
    PyObject *textures;
    
    /* the block position and type, and the block array */
    int x, y, z;
    unsigned char block;
    unsigned char block_data;
    unsigned char block_pdata;

    /* useful information about this, and neighboring, chunks */
    PyObject *blockdatas;
    PyObject *blocks;
    PyObject *up_left_blocks;
    PyObject *up_right_blocks;
    PyObject *left_blocks;
    PyObject *right_blocks;
} RenderState;
PyObject *init_chunk_render(void);
PyObject *chunk_render(PyObject *self, PyObject *args);
typedef enum
{
    KNOWN,
    TRANSPARENT,
    SOLID,
    FLUID,
    NOSPAWN,
    NODATA,
} BlockProperty;
/* globals set in init_chunk_render, here because they're used
   in block_has_property */
extern unsigned int max_blockid;
extern unsigned int max_data;
extern unsigned char *block_properties;
static inline int
block_has_property(unsigned char b, BlockProperty prop) {
    if (b >= max_blockid || !(block_properties[b] & (1 << KNOWN))) {
        /* block is unknown, return defaults */
        if (prop == TRANSPARENT)
            return 1;
        return 0;
    }
    
    return block_properties[b] & (1 << prop);
}
#define is_transparent(b) block_has_property((b), TRANSPARENT)

/* helper for getting chunk data arrays */
typedef enum
{
    BLOCKS,
    BLOCKDATA,
    BLOCKLIGHT,
    SKYLIGHT,
} ChunkDataType;
typedef enum
{
    CURRENT,
    DOWN_RIGHT, /*  0, +1 */
    DOWN_LEFT,  /* -1,  0 */
    UP_RIGHT,   /* +1,  0 */
    UP_LEFT,    /*  0, -1 */
} ChunkNeighborName;
PyObject *get_chunk_data(RenderState *state, ChunkNeighborName neighbor, ChunkDataType type,
        unsigned char clearexception);

/* pull in the rendermode info */
#include "rendermodes.h"

/* in endian.c */
void init_endian(void);
unsigned short big_endian_ushort(unsigned short in);
unsigned int big_endian_uint(unsigned int in);

#endif /* __OVERVIEWER_H_INCLUDED__ */
