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

typedef struct {
    PyObject *black_color;
    PyObject *white_color;
    unsigned int sealevel;
} PrimitiveHeightFading;

static int
height_fading_start(void *data, RenderState *state, PyObject *support) {
    PrimitiveHeightFading *self = (PrimitiveHeightFading *)data;
    
    if (!render_mode_parse_option(support, "sealevel", "I", &(self->sealevel)))
        return 1;
    
    self->black_color = PyObject_GetAttrString(support, "black_color");
    self->white_color = PyObject_GetAttrString(support, "white_color");
    
    return 0;
}

static void
height_fading_finish(void *data, RenderState *state) {
    PrimitiveHeightFading *self = (PrimitiveHeightFading *)data;

    Py_DECREF(self->black_color);
    Py_DECREF(self->white_color);
}

static void
height_fading_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    float alpha;
    PrimitiveHeightFading *self = (PrimitiveHeightFading *)data;
    int y = 16 * state->chunky + state->y;

    /* do some height fading */
    PyObject *height_color = self->white_color;

    /* current formula requires y to be between 0 and 127, so scale it */
    y = (y * 128) / (2 * self->sealevel);
    
    /* negative alpha => darkness, positive => light */
    alpha = (1.0 / (1 + expf((70 - y) / 11.0))) * 0.6 - 0.55;
    
    if (alpha < 0.0) {
        alpha *= -1;
        height_color = self->black_color;
    }
    
    alpha_over_full(state->img, height_color, mask_light, alpha, state->imgx, state->imgy, 0, 0);
}

RenderPrimitiveInterface primitive_height_fading = {
    "height-fading", sizeof(PrimitiveHeightFading),
    height_fading_start,
    height_fading_finish,
    NULL,
    NULL,
    height_fading_draw,
};
