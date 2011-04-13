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
#include <math.h>
//~ 
//~ /* figures out the black_coeff from a given skylight and blocklight, used in
   //~ lighting calculations -- note this is *different* from the one in
   //~ rendermode-lighting.c (the "skylight - 11" part) */
//~ static float calculate_darkness(unsigned char skylight, unsigned char blocklight) {
    //~ return 1.0f - powf(0.8f, 15.0 - MAX(blocklight, skylight - 11));
//~ }

static int
rendermode_cave_occluded(void *data, RenderState *state) {
    int x = state->x, y = state->y, z = state->z;
    RenderModeCave* self;
    self = (RenderModeCave *)data;

    /* check if the block is touching skylight */
    if (z != 127) { 
        
        if (getArrayByte3D(self->skylight, x, y, z+1) != 0) {
            return 1;
        }

        if ((x == 15)) {
            if  (self->up_right_skylight != Py_None) {
                if (getArrayByte3D(self->up_right_skylight, 0, y, z) != 0) {
                    return 1;
                }
            }
        } else {
            if (getArrayByte3D(self->skylight, x+1, y, z) != 0) {
                return 1;
            }
        }
        
        if (x == 0) {
            if  (self->left_skylight != Py_None) {
                if (getArrayByte3D(self->left_skylight, 15, y, z) != 0) {
                    return 1;
                }
            }
        } else {
            if (getArrayByte3D(self->skylight, x-1, y, z) != 0) {
                return 1;
            }
        }

        if (y == 15) {
            if  (self->right_skylight != Py_None) {
                if (getArrayByte3D(self->right_skylight, 0, y, z) != 0) {
                    return 1;
                }
            }
        } else {
            if (getArrayByte3D(self->skylight, x, y+1, z) != 0) {
                return 1;
            }
        }

        if (y == 0) {
            if  (self->up_left_skylight != Py_None) {
                if (getArrayByte3D(self->up_left_skylight, 15, y, z) != 0) {
                    return 1;
                }
            }
        } else {
            if (getArrayByte3D(self->skylight, x, y-1, z) != 0) {
                return 1;
            }
        }

        /* check for normal occlusion */
        /* use ajacent chunks, if not you get blocks spreaded in chunk edges */
        if ( (x == 0) && (y != 15) ) {
            if (state->left_blocks != Py_None) {
                if (!is_transparent(getArrayByte3D(state->left_blocks, 15, y, z)) &&
                    !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
                    !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
                    return 1;
                }
            } else {
                return 1;
            }
        }
        
        if ( (x != 0) && (y == 15) ) {
            if (state->right_blocks != Py_None) {
                if (!is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
                    !is_transparent(getArrayByte3D(state->right_blocks, x, 0, z)) &&
                    !is_transparent(getArrayByte3D(state->blocks, x, y, z+1))) {
                    return 1;
                }
            } else {
                return 1;
            }
        }
        
        if ( (x == 0) && (y == 15) ) {
            if ((state->left_blocks != Py_None) &&
                (state->right_blocks != Py_None)) {
                if (!is_transparent(getArrayByte3D(state->left_blocks, 15, y, z)) &&
                    !is_transparent(getArrayByte3D(state->right_blocks, x, 0, z)) &&
                    !is_transparent(getArrayByte3D(state->blocks, x, y, z+1))) {
                    return 1;
                }
            } else {
                return 1;
            }
        }
        
        if ( (x != 0) && (y != 15) &&
            !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
            !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
            !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
            return 1;
        }

    } else { /* if z == 127 don't skip */
        return 1;
    }

    return 0;
}

static int
rendermode_cave_start(void *data, RenderState *state) {
    RenderModeCave* self;
    self = (RenderModeCave *)data;

    /* first, chain up */
    int ret = rendermode_normal.start(data, state);
    if (ret != 0)
        return ret;

    /* if there's skylight we are in the surface! */
    self->skylight = PyObject_GetAttrString(state->self, "skylight");
    self->left_skylight = PyObject_GetAttrString(state->self, "left_skylight");
    self->right_skylight = PyObject_GetAttrString(state->self, "right_skylight");
    self->up_left_skylight = PyObject_GetAttrString(state->self, "up_left_skylight");
    self->up_right_skylight = PyObject_GetAttrString(state->self, "up_right_skylight");

    return 0;
}

static void
rendermode_cave_finish(void *data, RenderState *state) {
    RenderModeCave* self;
    self = (RenderModeCave *)data;

    Py_DECREF(self->skylight);
    Py_DECREF(self->left_skylight);
    Py_DECREF(self->right_skylight);
    Py_DECREF(self->up_left_skylight);
    Py_DECREF(self->up_right_skylight);

    rendermode_normal.finish(data, state);
}

static void
rendermode_cave_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    /* nothing special to do */
    rendermode_normal.draw(data, state, src, mask);
    
}

RenderModeInterface rendermode_cave = {
    "cave", "render only caves in normal mode",
    sizeof(RenderModeCave),
    rendermode_cave_start,
    rendermode_cave_finish,
    rendermode_cave_occluded,
    rendermode_cave_draw,
};
