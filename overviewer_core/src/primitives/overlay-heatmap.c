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

#include <math.h>
#include "overlay.h"

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    int t_invisible;
    int delta_t;
} RenderPrimitiveHeatmap;

static void get_color(void* data, RenderState* state,
                      uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) {
    RenderPrimitiveHeatmap* self = (RenderPrimitiveHeatmap*)data;
    long mtime;
    float _value_f;
    char value;
    PyObject *mtime_pyobj;

    // Set default values (will get overridden based on self->mode)
    *r = 255;
    *g = 0;
    *b = 0;

    // Get the chunk modified time
    mtime_pyobj = PyObject_CallMethod(state->regionset, "get_chunk_mtime", "ii", state->chunkx, state->chunkz);
    if (mtime_pyobj == NULL || mtime_pyobj == Py_None) {
        *a = 0;
        return;
    }
    mtime = PyLong_AsLong(mtime_pyobj);
    Py_XDECREF(mtime_pyobj);

    // Convert the time to a value in the range [0,255] based on t_invisible and delta_t
    _value_f = (mtime - self->t_invisible) / (float)self->delta_t;
    value = _value_f <= 0 ? 0 : (_value_f >= 1 ? 255 : 255*_value_f);
    *a = value;
}

static bool
overlay_heatmap_start(void* data, RenderState* state, PyObject* support) {
    RenderPrimitiveHeatmap* self;
    int t_full;

    /* first, chain up */
    bool ret = primitive_overlay.start(data, state, support);
    if (ret != false)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveHeatmap*)data;
    self->parent.get_color = get_color;

    if (!render_mode_parse_option(support, "t_invisible", "I", &(self->t_invisible)))
        return true;

    if (!render_mode_parse_option(support, "t_full", "I", &t_full))
        return true;

    self->delta_t = t_full - self->t_invisible;
    return false;
}

static void
overlay_heatmap_finish(void* data, RenderState* state) {
    /* chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_heatmap = {
    "overlay-heatmap",
    sizeof(RenderPrimitiveHeatmap),
    overlay_heatmap_start,
    overlay_heatmap_finish,
    NULL,
    NULL,
    overlay_draw,
};
