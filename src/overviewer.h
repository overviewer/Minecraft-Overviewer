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
#define OVERVIEWER_EXTENSION_VERSION 5

/* Python PIL, and numpy headers */
#include <Python.h>
#include <Imaging.h>
#include <numpy/arrayobject.h>

/* macro for getting a value out of various numpy arrays */
#define getArrayByte3D(array, x,y,z) (*(unsigned char *)(PyArray_GETPTR3((array), (x), (y), (z))))
#define getArrayShort1D(array, x) (*(unsigned short *)(PyArray_GETPTR1((array), (x))))

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

/* in iterate.c */
typedef struct {
    /* the ChunkRenderer object */
    PyObject *self;
    
    /* important modules, for convenience */
    PyObject *textures;
    PyObject *chunk;
    
    /* the rest only make sense for occluded() and draw() !! */
    
    /* the tile image and destination */
    PyObject *img;
    int imgx, imgy;
    
    /* the block position and type, and the block array */
    int x, y, z;
    unsigned char block;
    PyObject *blocks;
    PyObject *up_left_blocks;
    PyObject *up_right_blocks;
    PyObject *left_blocks;
    PyObject *right_blocks;
} RenderState;
int init_chunk_render(void);
int is_transparent(unsigned char b);
PyObject *chunk_render(PyObject *self, PyObject *args);

/* pull in the rendermode info */
#include "rendermodes.h"

/* in endian.c */
void init_endian(void);
unsigned short big_endian_ushort(unsigned short in);
unsigned int big_endian_uint(unsigned int in);

#endif /* __OVERVIEWER_H_INCLUDED__ */
