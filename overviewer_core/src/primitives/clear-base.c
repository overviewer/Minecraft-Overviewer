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
clear_base_occluded(void *data, RenderState *state, int x, int y, int z) {
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !render_mode_hidden(state->rendermode, x-1, y, z) &&
         !render_mode_hidden(state->rendermode, x, y, z+1) &&
         !render_mode_hidden(state->rendermode, x, y+1, z) &&
         !is_transparent(getArrayShort3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayShort3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayShort3D(state->blocks, x, y+1, z))) {
        return 1;
    }
    
    return 0;
}

static void
clear_base_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    /* clear the draw space -- set alpha to 0 within mask */
    tint_with_mask(state->img, 255, 255, 255, 0, mask, state->imgx, state->imgy, 0, 0);
}

RenderPrimitiveInterface primitive_clear_base = {
    "clear-base",
    0,
    NULL,
    NULL,
    clear_base_occluded,
    NULL,
    clear_base_draw,
};
