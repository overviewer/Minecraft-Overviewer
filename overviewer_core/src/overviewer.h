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

// increment this value if you've made a change to the c extension
// and want to force users to rebuild
#define OVERVIEWER_EXTENSION_VERSION 86

#include <stdbool.h>
#include <stdint.h>

/* Python PIL, and numpy headers */
#include <Imaging.h>
#include <Python.h>
#include <numpy/arrayobject.h>
/* Fix Pillow on mingw-w64 which includes windows.h in Imaging.h */
#undef TRANSPARENT
/* Utility macros */
#include "mc_id.h"
#include "utils.h"

/* macro for getting a value out of various numpy arrays the 3D arrays have
   interesting, swizzled coordinates because minecraft (anvil) stores blocks
   in y/z/x order for 3D, z/x order for 2D */
#define getArrayByte3D(array, x, y, z) (*(uint8_t*)(PyArray_GETPTR3((array), (y), (z), (x))))
#define getArrayShort3D(array, x, y, z) (*(uint16_t*)(PyArray_GETPTR3((array), (y), (z), (x))))
#define getArrayByte2D(array, x, y) (*(uint8_t*)(PyArray_GETPTR2((array), (y), (x))))

/* in composite.c */
Imaging imaging_python_to_c(PyObject* obj);
PyObject* alpha_over(PyObject* dest, PyObject* src, PyObject* mask,
                     int32_t dx, int32_t dy, int32_t xsize, int32_t ysize);
PyObject* alpha_over_full(PyObject* dest, PyObject* src, PyObject* mask, float overall_alpha,
                          int32_t dx, int32_t dy, int32_t xsize, int32_t ysize);
PyObject* alpha_over_wrap(PyObject* self, PyObject* args);
PyObject* tint_with_mask(PyObject* dest, uint8_t sr, uint8_t sg,
                         uint8_t sb, uint8_t sa,
                         PyObject* mask, int32_t dx, int32_t dy, int32_t xsize, int32_t ysize);
PyObject* draw_triangle(PyObject* dest, int32_t inclusive,
                        int32_t x0, int32_t y0,
                        uint8_t r0, uint8_t g0, uint8_t b0,
                        int32_t x1, int32_t y1,
                        uint8_t r1, uint8_t g1, uint8_t b1,
                        int32_t x2, int32_t y2,
                        uint8_t r2, uint8_t g2, uint8_t b2,
                        int32_t tux, int32_t tuy, int32_t* touchups, uint32_t num_touchups);
PyObject* resize_half(PyObject* dest, PyObject* src);
PyObject* resize_half_wrap(PyObject* self, PyObject* args);

/* forward declaration of RenderMode object */
typedef struct _RenderMode RenderMode;

/* in iterate.c */
#define SECTIONS_PER_CHUNK 16
typedef struct {
    /* whether this chunk is loaded: use load_chunk to load */
    int32_t loaded;
    /* chunk biome array */
    PyArrayObject* biomes;
    /* whether this is a 3d biome array */
    bool new_biomes;
    /* all the sections in a given chunk */
    struct {
        /* all there is to know about each section */
        PyArrayObject *blocks, *data, *skylight, *blocklight;
    } sections[SECTIONS_PER_CHUNK];
} ChunkData;
typedef struct {
    /* the regionset object, and chunk coords */
    PyObject* world;
    PyObject* regionset;
    int32_t chunkx, chunky, chunkz;

    /* the tile image and destination */
    PyObject* img;
    int32_t imgx, imgy;

    /* the current render mode in use */
    RenderMode* rendermode;

    /* the Texture object */
    PyObject* textures;

    /* the block position and type, and the block array */
    int32_t x, y, z;
    mc_block_t block;
    uint8_t block_data;
    uint16_t block_pdata;

    /* useful information about this, and neighboring, chunks */
    PyArrayObject* blockdatas;
    PyArrayObject* blocks;

    /* 3x3 array of this and neighboring chunk columns */
    ChunkData chunks[3][3];
} RenderState;
PyObject* init_chunk_render(void);
/* returns true on error, x,z relative */
bool load_chunk(RenderState* state, int32_t x, int32_t z, uint8_t required);
PyObject* chunk_render(PyObject* self, PyObject* args);
typedef enum {
    KNOWN,
    TRANSPARENT,
    SOLID,
    FLUID,
    NOSPAWN,
    NODATA,
} BlockProperty;
/* globals set in init_chunk_render, here because they're used
   in block_has_property */
extern uint32_t max_blockid;
extern uint32_t max_data;
extern uint8_t* block_properties;
static inline bool
block_has_property(mc_block_t b, BlockProperty prop) {
    if (b >= max_blockid || !(block_properties[b] & (1 << KNOWN))) {
        /* block is unknown, return defaults */
        if (prop == TRANSPARENT)
            return true;
        return false;
    }

    return block_properties[b] & (1 << prop);
}
#define is_transparent(b) block_has_property((b), TRANSPARENT)
#define is_known_transparent(b) block_has_property((b), TRANSPARENT) && block_has_property((b), KNOWN)

/* helper for indexing section data possibly across section boundaries */
typedef enum {
    BLOCKS,
    DATA,
    BLOCKLIGHT,
    SKYLIGHT,
    BIOMES,
} DataType;
static inline uint32_t get_data(RenderState* state, DataType type, int32_t x, int32_t y, int32_t z) {
    int32_t chunkx = 1, chunky = state->chunky, chunkz = 1;
    PyArrayObject* data_array = NULL;
    uint32_t def = 0;
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

    if (!(state->chunks[chunkx][chunkz].loaded)) {
        if (load_chunk(state, chunkx - 1, chunkz - 1, 0))
            return def;
    }

    switch (type) {
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
    if (type == BIOMES) {
        if (state->chunks[chunkx][chunkz].new_biomes) {
            return getArrayByte3D(data_array, x / 4, y / 4, z / 4);
        } else {
            return getArrayByte2D(data_array, x, z);
        }
    }
    return getArrayByte3D(data_array, x, y, z);
}

/* pull in the rendermode info */
#include "rendermodes.h"

/* in endian.c */
void init_endian(void);
uint16_t big_endian_ushort(uint16_t in);
uint32_t big_endian_uint(uint32_t in);

#endif /* __OVERVIEWER_H_INCLUDED__ */
