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
 * To make a new render mode (the C part, at least):
 *
 *     * add a data struct and extern'd interface declaration below
 *
 *     * fill in this interface struct in rendermode-(yourmode).c
 *         (see rendermodes-normal.c for an example: the "normal" mode)
 *
 *     * if you want to derive from (say) the "normal" mode, put
 *       a RenderModeNormal entry at the top of your data struct, and
 *       be sure to call your parent's functions in your own!
 *         (see rendermode-night.c for a simple example derived from
 *          the "lighting" mode)
 *
 *     * add a condition to get_render_mode() in rendermodes.c
 */

#ifndef __RENDERMODES_H_INCLUDED__
#define __RENDERMODES_H_INCLUDED__

#include <Python.h>

/* rendermode interface */
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

/* individual rendermode interface declarations follow */

/* NORMAL */
typedef struct {
    /* normal mode does not have any special data, so just use a dummy int
       this way, normal mode is just like any other type of render mode */
    int dummy;
} RenderModeNormal;
extern RenderModeInterface rendermode_normal;

/* LIGHTING */
typedef struct {
    /* inherits from normal render mode */
    RenderModeNormal parent;
    
    PyObject *black_color, *facemasks_py;
    PyObject *facemasks[3];
    
    /* extra block data, loaded off the chunk class */
    PyObject *skylight, *blocklight;
    PyObject *left_blocks, *left_skylight, *left_blocklight;
    PyObject *right_blocks, *right_skylight, *right_blocklight;
    
    /* can be overridden in derived rendermodes to control lighting
       arguments are skylight, blocklight */
    float (*calculate_darkness)(unsigned char, unsigned char);
} RenderModeLighting;
extern RenderModeInterface rendermode_lighting;

/* NIGHT */
typedef struct {
    /* inherits from lighting */
    RenderModeLighting parent;
} RenderModeNight;
extern RenderModeInterface rendermode_night;

#endif /* __RENDERMODES_H_INCLUDED__ */
