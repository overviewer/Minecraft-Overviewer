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


#define WINVER 0x0601
#define _WIN32_WINNT 0x0601

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


// increment this value if you've made a change to the c extesion
// and want to force users to rebuild
#define OVERVIEWER_EXTENSION_VERSION 52

/* Python PIL, and numpy headers */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <Imaging.h>
/* Fix Pillow on mingw-w64 which includes windows.h in Imaging.h */
#undef TRANSPARENT

/* like (a * b + 127) / 255), but much faster on most platforms
   from PIL's _imaging.c */
#define MULDIV255(a, b, tmp)								\
	(tmp = (a) * (b) + 128, ((((tmp) >> 8) + (tmp)) >> 8))

/* macro for getting a value out of various numpy arrays the 3D arrays have
   interesting, swizzled coordinates because minecraft (anvil) stores blocks
   in y/z/x order for 3D, z/x order for 2D */
#define getArrayByte3D(array, x,y,z) (*(unsigned char *)(PyArray_GETPTR3((array), (y), (z), (x))))
#define getArrayShort3D(array, x,y,z) (*(unsigned short *)(PyArray_GETPTR3((array), (y), (z), (x))))
#define getArrayByte2D(array, x,y) (*(unsigned char *)(PyArray_GETPTR2((array), (y), (x))))


/* generally useful MAX / MIN macros */
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, a, b) (MIN(MAX(x, a), b))

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
PyObject *resize_half(PyObject *dest, PyObject *src);
PyObject *resize_half_wrap(PyObject *self, PyObject *args);

/* forward declaration of RenderMode object */
typedef struct _RenderMode RenderMode;

/* in iterate.c */
#define SECTIONS_PER_CHUNK 16
typedef struct {
    /* whether this chunk is loaded: use load_chunk to load */
    int loaded;
    /* chunk biome array */
    PyArrayObject *biomes;
    /* all the sections in a given chunk */
    struct {
        /* all there is to know about each section */
        PyArrayObject *blocks, *data, *skylight, *blocklight;
    } sections[SECTIONS_PER_CHUNK];
} ChunkData;
typedef struct {
    /* the regionset object, and chunk coords */
    PyObject *world;
    PyObject *regionset;
    int chunkx, chunky, chunkz;
    
    /* the tile image and destination */
    PyObject *img;
    int imgx, imgy;
    
    /* the current render mode in use */
    RenderMode *rendermode;
    
    /* the Texture object */
    PyObject *textures;
    
    /* the block position and type, and the block array */
    int x, y, z;
    unsigned short block;
    unsigned char block_data;
    unsigned short block_pdata;

    /* useful information about this, and neighboring, chunks */
    PyArrayObject *blockdatas;
    PyArrayObject *blocks;
    
    /* 3x3 array of this and neighboring chunk columns */
    ChunkData chunks[3][3];    
} RenderState;
PyObject *init_chunk_render(void);
/* returns true on error, x,z relative */
int load_chunk(RenderState* state, int x, int z, unsigned char required);
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
block_has_property(unsigned short b, BlockProperty prop) {
    if (b >= max_blockid || !(block_properties[b] & (1 << KNOWN))) {
        /* block is unknown, return defaults */
        if (prop == TRANSPARENT)
            return 1;
        return 0;
    }
    
    return block_properties[b] & (1 << prop);
}
#define is_transparent(b) block_has_property((b), TRANSPARENT)
#define is_known_transparent(b) block_has_property((b), TRANSPARENT) && block_has_property((b), KNOWN)

/* helper for indexing section data possibly across section boundaries */
typedef enum
{
    BLOCKS,
    DATA,
    BLOCKLIGHT,
    SKYLIGHT,
    BIOMES,
} DataType;
static inline unsigned int get_data(RenderState *state, DataType type, int x, int y, int z)
{
    int chunkx = 1, chunky = state->chunky, chunkz = 1;
    PyArrayObject *data_array = NULL;
    unsigned int def = 0;
    if (type == SKYLIGHT)
        def = 15;
    
    if (x >= 16) {
        x -= 16;
        chunkx++;
    } else if (x < 0) {
        x += 16;
        chunkx--;
    }
    if (z >= 16) {
        z -= 16;
        chunkz++;
    } else if (z < 0) {
        z += 16;
        chunkz--;
    }

    while (y >= 16) {
        y -= 16;
        chunky++;
    }
    while (y < 0) {
        y += 16;
        chunky--;
    }
    if (chunky < 0 || chunky >= SECTIONS_PER_CHUNK)
        return def;
    
    if (!(state->chunks[chunkx][chunkz].loaded))
    {
        if (load_chunk(state, chunkx - 1, chunkz - 1, 0))
            return def;
    }
    
    switch (type)
    {
    case BLOCKS:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].blocks;
        break;
    case DATA:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].data;
        break;
    case BLOCKLIGHT:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].blocklight;
        break;
    case SKYLIGHT:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].skylight;
        break;
    case BIOMES:
        data_array = state->chunks[chunkx][chunkz].biomes;
    };
    
    if (data_array == NULL)
        return def;
    
    if (type == BLOCKS)
        return getArrayShort3D(data_array, x, y, z);
    if (type == BIOMES)
        return getArrayByte2D(data_array, x, z);
    return getArrayByte3D(data_array, x, y, z);
}

/* pull in the rendermode info */
#include "rendermodes.h"

/* in endian.c */
void init_endian(void);
unsigned short big_endian_ushort(unsigned short in);
unsigned int big_endian_uint(unsigned int in);

#endif /* __OVERVIEWER_H_INCLUDED__ */
