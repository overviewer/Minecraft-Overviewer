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
    int32_t min;
    int32_t max;
} PrimitiveDepth;

static bool
depth_start(void* data, RenderState* state, PyObject* support) {
    PrimitiveDepth* self = (PrimitiveDepth*)data;

    if (!render_mode_parse_option(support, "min", "I", &(self->min)))
        return true;
    if (!render_mode_parse_option(support, "max", "I", &(self->max)))
        return true;

    return false;
}

static bool
depth_hidden(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    PrimitiveDepth* self = (PrimitiveDepth*)data;
    y += 16 * state->chunky;
    if (y > self->max || y < self->min) {
        return true;
    }
    return false;
}

RenderPrimitiveInterface primitive_depth = {
    "depth",
    sizeof(PrimitiveDepth),
    depth_start,
    NULL,
    NULL,
    depth_hidden,
    NULL,
};
