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

#include "overlay.h"

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    RenderPrimitiveOverlay* self = (RenderPrimitiveOverlay *)data;

    *r = self->color->r;
    *g = self->color->g;
    *b = self->color->b;
    *a = self->color->a;
}

static int
overlay_start(void *data, RenderState *state, PyObject *support) {
    PyObject *opt = NULL;
    OverlayColor *color = NULL;
    RenderPrimitiveOverlay *self = (RenderPrimitiveOverlay *)data;
    
    self->facemask_top = PyObject_GetAttrString(support, "facemask_top");
    self->white_color = PyObject_GetAttrString(support, "whitecolor");
    self->get_color = get_color;
    
    color = self->color = calloc(1, sizeof(OverlayColor));

    if (color == NULL) {
        return 1;
    }
    
    self->default_color.r = 200;
    self->default_color.g = 200;
    self->default_color.b = 255;
    self->default_color.a = 155;
    
    if(!render_mode_parse_option(support, "overlay_color", "bbbb", &(color->r), &(color->g), &(color->b), &(color->a))) {
        if(PyErr_Occurred())
            PyErr_Clear();
        free(color);
        self->color = &self->default_color;
        // Check if it is None, if it is, continue and use the default, if it isn't, return an error
        if(render_mode_parse_option(support, "overlay_color", "O", &(opt))) {
            // If it is an object, check to see if it is None, if it is, use the default.
            if(opt && opt != Py_None) {
                return 1;
            }
        }
    }

    return 0;
}

static void
overlay_finish(void *data, RenderState *state) {
    RenderPrimitiveOverlay *self = (RenderPrimitiveOverlay *)data;

    if (self->color && self->color != &self->default_color) {
        free(self->color);
    }
    
    Py_DECREF(self->facemask_top);
    Py_DECREF(self->white_color);
}

void
overlay_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderPrimitiveOverlay *self = (RenderPrimitiveOverlay *)data;
    unsigned char r, g, b, a;
    unsigned short top_block;

    // exactly analogous to edge-line code for these special blocks
    int increment=0;
    if (state->block == 44)  // half-step
        increment=6;
    else if (state->block == 78) // snow
        increment=9;
    
    /* skip rendering the overlay if we can't see it */
    top_block = get_data(state, BLOCKS, state->x, state->y+1, state->z);
    if (!is_transparent(top_block)) {
        return;
    }
    
    /* check to be sure this block is solid/fluid */
    if (block_has_property(top_block, SOLID) || block_has_property(top_block, FLUID)) {
        
        /* top block is fluid or solid, skip drawing */
        return;
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
        alpha_over_full(state->img, self->white_color, self->facemask_top, a/255.f, state->imgx, state->imgy + increment, 0, 0);
        tint_with_mask(state->img, r, g, b, 255, self->facemask_top, state->imgx, state->imgy + increment, 0, 0);
    }
}

RenderPrimitiveInterface primitive_overlay = {
    "overlay",
    sizeof(RenderPrimitiveOverlay),
    overlay_start,
    overlay_finish,
    NULL,
    NULL,
    overlay_draw,
};
