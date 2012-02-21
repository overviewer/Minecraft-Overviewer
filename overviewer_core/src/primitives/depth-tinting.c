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

#include "../overviewer.h"
#include <math.h>

typedef struct {
    /* list of colors used for tinting */
    PyObject *depth_colors;
} RenderPrimitiveDepthTinting;

static int
depth_tinting_start(void *data, RenderState *state, PyObject *support) {
    RenderPrimitiveDepthTinting* self;
    self = (RenderPrimitiveDepthTinting *)data;

    self->depth_colors = PyObject_GetAttrString(support, "depth_colors");
    if (self->depth_colors == NULL)
        return 1;

    return 0;
}

static void
depth_tinting_finish(void *data, RenderState *state) {
    RenderPrimitiveDepthTinting* self;
    self = (RenderPrimitiveDepthTinting *)data;

    Py_DECREF(self->depth_colors);
}

static void
depth_tinting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderPrimitiveDepthTinting* self;
    int y, r, g, b;
    self = (RenderPrimitiveDepthTinting *)data;

    y = state->chunky * 16 + state->y;
    r = 0, g = 0, b = 0;
    
    /* the colors array assumes y is between 0 and 127, so we scale it */
    y = (y * 128) / (16 * SECTIONS_PER_CHUNK);

    /* get the colors and tint and tint */
    r = PyInt_AsLong(PyList_GetItem(self->depth_colors, 0 + y*3));
    g = PyInt_AsLong(PyList_GetItem(self->depth_colors, 1 + y*3));
    b = PyInt_AsLong(PyList_GetItem(self->depth_colors, 2 + y*3));
    
    tint_with_mask(state->img, r, g, b, 255, mask, state->imgx, state->imgy, 0, 0);
}

RenderPrimitiveInterface primitive_depth_tinting = {
    "depth-tinting", sizeof(RenderPrimitiveDepthTinting),
    depth_tinting_start,
    depth_tinting_finish,
    NULL,
    NULL,
    depth_tinting_draw,
};
