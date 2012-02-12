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

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    *r = 200;
    *g = 200;
    *b = 255;
    *a = 155;
}

static int
rendermode_overlay_start(void *data, RenderState *state, PyObject *options) {
    PyObject *facemasks_py;
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    
    facemasks_py = PyObject_GetAttrString(state->support, "facemasks");
    /* borrowed reference, needs to be incref'd if we keep it */
    self->facemask_top = PyTuple_GetItem(facemasks_py, 0);
    Py_INCREF(self->facemask_top);
    Py_DECREF(facemasks_py);
    
    self->white_color = PyObject_GetAttrString(state->support, "white_color");
    self->get_color = get_color;
    
    return 0;
}

static void
rendermode_overlay_finish(void *data, RenderState *state) {
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    
    Py_DECREF(self->facemask_top);
    Py_DECREF(self->white_color);
}

static int
rendermode_overlay_occluded(void *data, RenderState *state, int x, int y, int z) {
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !render_mode_hidden(state->rendermode, x-1, y, z) &&
         !render_mode_hidden(state->rendermode, x, y, z+1) &&
         !render_mode_hidden(state->rendermode, x, y+1, z) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }
    
    return 0;
}

static int
rendermode_overlay_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* overlays hide nothing by default */
    return 0;
}

static void
rendermode_overlay_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderModeOverlay *self = (RenderModeOverlay *)data;
    unsigned char r, g, b, a;

    // exactly analogous to edge-line code for these special blocks
    int increment=0;
    if (state->block == 44)  // half-step
        increment=6;
    else if (state->block == 78) // snow
        increment=9;
    
    /* clear the draw space -- set alpha to 0 within mask */
    tint_with_mask(state->img, 255, 255, 255, 0, mask, state->imgx, state->imgy, 0, 0);

    /* skip rendering the overlay if we can't see it */
    if (state->z != 127) {
        unsigned char top_block = getArrayByte3D(state->blocks, state->x, state->y, state->z+1);
        if (!is_transparent(top_block)) {
            return;
        }
        
        /* check to be sure this block is solid/fluid */
        if (block_has_property(top_block, SOLID) || block_has_property(top_block, FLUID)) {
            
            /* top block is fluid or solid, skip drawing */
            return;
        }
    }
    
    /* check to be sure this block is solid/fluid */
    if (!block_has_property(state->block, SOLID) && !block_has_property(state->block, FLUID)) {
        
        /* not fluid or solid, skip drawing the overlay */
        return;
    }

    /* get our color info */
    self->get_color(data, state, &r, &g, &b, &a);
    
    /* do the overlay */
    if (a > 0) {
        alpha_over(state->img, self->white_color, self->facemask_top, state->imgx, state->imgy + increment, 0, 0);
        tint_with_mask(state->img, r, g, b, a, self->facemask_top, state->imgx, state->imgy + increment, 0, 0);
    }
}

RenderModeInterface rendermode_overlay = {
    "overlay", "Overlay",
    "base rendermode for informational overlays",
    NULL,
    NULL,
    sizeof(RenderModeOverlay),
    rendermode_overlay_start,
    rendermode_overlay_finish,
    rendermode_overlay_occluded,
    rendermode_overlay_hidden,
    rendermode_overlay_draw,
};
