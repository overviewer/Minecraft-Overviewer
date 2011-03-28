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

#include "overviewer.h"

static int
rendermode_overlay_start(void *data, RenderState *state) {
    PyObject *facemasks_py;
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    
    facemasks_py = PyObject_GetAttrString(state->chunk, "facemasks");
    /* borrowed reference, needs to be incref'd if we keep it */
    self->facemask_top = PyTuple_GetItem(facemasks_py, 0);
    Py_INCREF(self->facemask_top);
    Py_DECREF(facemasks_py);
    
    self->white_color = PyObject_GetAttrString(state->chunk, "white_color");
    
    return 0;
}

static void
rendermode_overlay_finish(void *data, RenderState *state) {
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    
    Py_XDECREF(self->facemask_top);
    Py_XDECREF(self->white_color);
}

static int
rendermode_overlay_occluded(void *data, RenderState *state) {
    int x = state->x, y = state->y, z = state->z;
    
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }

    return 0;
}

static void
rendermode_overlay_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    
    /* clear the draw space -- set alpha to 0 within mask */
    tint_with_mask(state->img, 255, 255, 255, 0, mask, state->imgx, state->imgy, 0, 0);
    
    /* do the overlay */
    alpha_over_full(state->img, self->white_color, self->facemask_top, 0.5, state->imgx, state->imgy, 0, 0);
}

RenderModeInterface rendermode_overlay = {
    sizeof(RenderModeOverlay),
    rendermode_overlay_start,
    rendermode_overlay_finish,
    rendermode_overlay_occluded,
    rendermode_overlay_draw,
};
