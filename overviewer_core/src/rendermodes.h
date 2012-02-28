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

/* To make a new render primitive:
 *
 *  * add a new class to rendermodes.py
 *        there are a ton of examples there, the syntax is pretty simple. If
 *        you need any extra objects that are easy to create in python, this
 *        is where you put them.
 *
 *  * create a file in src/primitives with the same name
 *        so, Nether (named "nether") goes in `nether.c`.
 *
 *  * declare a RenderPrimitiveInterface with the name primitive_name
 *        if you have an underscore in the name, replace it with a
 *        hyphen. height-fading uses primitive_height_fading.
 *
 *  * fill in the entries of this struct
 *        the name should match, and you should declare an 'instance' struct
 *        to use as the self argument to each function. See nether.c and
 *        height-fading.c for simple examples.
 *
 * setup.py will pick up your primitive, add it to the global list, and build
 * it for you if you follow these conventions.
 */

#ifndef __RENDERMODES_H_INCLUDED__
#define __RENDERMODES_H_INCLUDED__

#include <Python.h>
#include "overviewer.h"

/* render primitive interface */
typedef struct {
    /* the name of this mode */
    const char *name;    
    /* the size of the local storage for this rendermode */
    unsigned int data_size;
    
    /* may return non-zero on error, last arg is the python support object */
    int (*start)(void *, RenderState *, PyObject *);
    void (*finish)(void *, RenderState *);
    /* returns non-zero to skip rendering this block because it's not visible */
    int (*occluded)(void *, RenderState *, int, int, int);
    /* returns non-zero to skip rendering this block because the user doesn't
     * want it visible */
    int (*hidden)(void *, RenderState *, int, int, int);
    /* last two arguments are img and mask, from texture lookup */
    void (*draw)(void *, RenderState *, PyObject *, PyObject *, PyObject *);
} RenderPrimitiveInterface;

/* A quick note about the difference between occluded and hidden:
 *
 * Occluded should be used to tell the renderer that a block will not be
 * visible in the final image because other blocks will be drawn on top of
 * it. This is a potentially *expensive* check that should be used rarely,
 * usually only once per block. The idea is this check is expensive, but not
 * as expensive as drawing the block itself.
 *
 * Hidden is used to tell the renderer not to draw the block, usually because
 * the current rendermode depends on those blocks being hidden to do its
 * job. For example, cave mode uses this to hide non-cave blocks. This check
 * should be *cheap*, as it's potentially called many times per block. For
 * example, in lighting mode it is called at most 4 times per block.
 */

/* convenience wrapper for a single primitive + interface */
typedef struct {
    void *primitive;
    RenderPrimitiveInterface *iface;
} RenderPrimitive;

/* wrapper for passing around rendermodes */
struct _RenderMode {
    unsigned int num_primitives;
    RenderPrimitive **primitives;
    RenderState *state;
};

/* functions for creating / using rendermodes */
RenderMode *render_mode_create(PyObject *mode, RenderState *state);
void render_mode_destroy(RenderMode *self);
int render_mode_occluded(RenderMode *self, int x, int y, int z);
int render_mode_hidden(RenderMode *self, int x, int y, int z);
void render_mode_draw(RenderMode *self, PyObject *img, PyObject *mask, PyObject *mask_light);

/* helper function for reading in rendermode options
   works like PyArg_ParseTuple on a support object */
int render_mode_parse_option(PyObject *support, const char *name, const char *format, ...);

#endif /* __RENDERMODES_H_INCLUDED__ */
