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

static int
no_fluids_start(void *data, RenderState *state, PyObject *support) {
    return 0;
}

static int
no_fluids_hidden(void *data, RenderState *state, int x, int y, int z) {
    return block_has_property(state->block, FLUID);
}

RenderPrimitiveInterface primitive_no_fluids = {
    "no-fluids", 0,
    no_fluids_start,
    NULL,
    NULL,
    no_fluids_hidden,
    NULL,
};
