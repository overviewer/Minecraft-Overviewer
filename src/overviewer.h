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

/* Python PIL, and numpy headers */
#include <Python.h>
#include <Imaging.h>
#include <numpy/arrayobject.h>

/* macro for getting a value out of a 3D numpy byte array */
#define getArrayByte3D(array, x,y,z) (*(unsigned char *)(PyArray_GETPTR3((array), (x), (y), (z))))

/* in composite.c */
Imaging imaging_python_to_c(PyObject *obj);
PyObject *alpha_over(PyObject *dest, PyObject *src, PyObject *mask, int dx,
                     int dy, int xsize, int ysize);
PyObject *alpha_over_wrap(PyObject *self, PyObject *args);
PyObject *brightness(PyObject *img, double factor);

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
} RenderState;
int init_chunk_render(void);
int is_transparent(unsigned char b);
PyObject *chunk_render(PyObject *self, PyObject *args);

/* in rendermode.c */
typedef struct {
    /* the size of the local storage for this rendermode */
    unsigned int data_size;
    
    /* may return non-zero on error */
    int (*start)(void *, RenderState *);
    void (*finish)(void *, RenderState *);
    /* returns non-zero to skip rendering this block */
    int (*occluded)(void *, RenderState *);
    /* last two arguments are img and mask, from texture lookup */
    void (*draw)(void *, RenderState *, PyObject *, PyObject *);
} RenderModeInterface;
/* figures out the render mode to use from the given ChunkRenderer */
RenderModeInterface *get_render_mode(RenderState *state);

#endif /* __OVERVIEWER_H_INCLUDED__ */
