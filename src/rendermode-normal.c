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
rendermode_normal_start(void *data, RenderState *state) {
    /* do nothing */
    return 0;
}

static void
rendermode_normal_finish(void *data, RenderState *state) {
    /* do nothing */
}

static int
rendermode_normal_occluded(void *data, RenderState *state) {
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
rendermode_normal_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    alpha_over(state->img, src, mask, state->imgx, state->imgy, 0, 0);
}

RenderModeInterface rendermode_normal = {
    sizeof(RenderModeNormal),
    rendermode_normal_start,
    rendermode_normal_finish,
    rendermode_normal_occluded,
    rendermode_normal_draw,
};
