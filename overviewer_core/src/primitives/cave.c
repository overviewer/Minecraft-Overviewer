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
#include <math.h>

typedef struct {
    /* data used to know where the surface is */
    PyObject *skylight;
    PyObject *left_skylight;
    PyObject *right_skylight;
    PyObject *up_left_skylight;
    PyObject *up_right_skylight;

    /* data used to know where the lit caves are */
    PyObject *blocklight;
    PyObject *left_blocklight;
    PyObject *right_blocklight;
    PyObject *up_left_blocklight;
    PyObject *up_right_blocklight;
    
    int only_lit;
} RenderPrimitiveCave;

static inline int
touches_light(unsigned int x, unsigned int y, unsigned int z,
              PyObject *light, PyObject *left_light, PyObject *right_light,
              PyObject *up_left_light, PyObject *up_right_light) {
    
    if (getArrayByte3D(light, x, y, z+1) != 0) {
        return 1;
    }

    if ((x == 15)) {
        if  (up_right_light) {
            if (getArrayByte3D(up_right_light, 0, y, z) != 0) {
                return 1;
            }
        }
    } else {
        if (getArrayByte3D(light, x+1, y, z) != 0) {
            return 1;
        }
    }
        
    if (x == 0) {
        if  (left_light) {
            if (getArrayByte3D(left_light, 15, y, z) != 0) {
                return 1;
            }
        }
    } else {
        if (getArrayByte3D(light, x-1, y, z) != 0) {
            return 1;
        }
    }

    if (y == 15) {
        if  (right_light) {
            if (getArrayByte3D(right_light, 0, y, z) != 0) {
                return 1;
            }
        }
    } else {
        if (getArrayByte3D(light, x, y+1, z) != 0) {
            return 1;
        }
    }

    if (y == 0) {
        if  (up_left_light) {
            if (getArrayByte3D(up_left_light, 15, y, z) != 0) {
                return 1;
            }
        }
    } else {
        if (getArrayByte3D(light, x, y-1, z) != 0) {
            return 1;
        }
    }
        
    return 0;
}

static int
cave_occluded(void *data, RenderState *state, int x, int y, int z) { 
    /* check for normal occlusion */
    /* use ajacent chunks, if not you get blocks spreaded in chunk edges */

    if (z != 127) {
        if ( (x == 0) && (y != 15) ) {
            if (state->left_blocks != NULL) {
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
            if (state->right_blocks != NULL) {
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
            if ((state->left_blocks != NULL) &&
                (state->right_blocks != NULL)) {
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
    }
    
    return 0;
}

static int
cave_hidden(void *data, RenderState *state, int x, int y, int z) {
    RenderPrimitiveCave* self;
    int dz = 0;
    self = (RenderPrimitiveCave *)data;
    
    /* check if the block is touching skylight */
    if (z != 127) { 
        
        if (touches_light(x, y, z, self->skylight, self->left_skylight, self->right_skylight, self->up_left_skylight, self->up_right_skylight)) {
            return 1;
        }

        if (self->only_lit && !touches_light(x, y, z, self->blocklight, self->left_blocklight, self->right_blocklight, self->up_left_blocklight, self->up_right_blocklight)) {
            return 1;
        }
    } else { /* if z == 127 skip */
        return 1;
    }

    /* check for lakes and seas and don't render them
     * at this point of the code the block has no skylight
     * but a deep sea can be completely dark
     */
    
    if ((getArrayByte3D(state->blocks, x, y, z) == 9) ||
        (getArrayByte3D(state->blocks, x, y, z+1) == 9)) {

        for (dz = z+1; dz < 127; dz++) { /* go up and check for skylight */
            if (getArrayByte3D(self->skylight, x, y, dz) != 0) {
                return 1;
            }
            if (getArrayByte3D(state->blocks, x, y, dz) != 9) {
            /* we are out of the water! and there's no skylight
             * , i.e. is a cave lake or something similar */
                return 0;
            }
        }
    }
    
    /* unfortunate side-effect of lit cave mode: we need to count occluded
     * blocks as hidden for the lighting to look right, since technically our
     * hiding depends on occlusion as well
     */
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }
    
    return cave_occluded(data, state, x, y, z);
}

static int
cave_start(void *data, RenderState *state, PyObject *support) {
    RenderPrimitiveCave* self;
    int ret;
    self = (RenderPrimitiveCave *)data;

    if (!render_mode_parse_option(support, "only_lit", "i", &(self->only_lit)))
        return 1;
    
    /* if there's skylight we are in the surface! */
    self->skylight = get_chunk_data(state, CURRENT, SKYLIGHT, 1);
    self->left_skylight = get_chunk_data(state, DOWN_LEFT, SKYLIGHT, 1);
    self->right_skylight = get_chunk_data(state, DOWN_RIGHT, SKYLIGHT, 1);
    self->up_left_skylight = get_chunk_data(state, UP_LEFT, SKYLIGHT, 1);
    self->up_right_skylight = get_chunk_data(state, UP_RIGHT, SKYLIGHT, 1);
    
    if (self->only_lit) {
        self->blocklight = get_chunk_data(state, CURRENT, BLOCKLIGHT, 1);
        self->left_blocklight = get_chunk_data(state, DOWN_LEFT, BLOCKLIGHT, 1);
        self->right_blocklight = get_chunk_data(state, DOWN_RIGHT, BLOCKLIGHT, 1);
        self->up_left_blocklight = get_chunk_data(state, UP_LEFT, BLOCKLIGHT, 1);
        self->up_right_blocklight = get_chunk_data(state, UP_RIGHT, BLOCKLIGHT, 1);
    }

    return 0;
}

static void
cave_finish(void *data, RenderState *state) {
    RenderPrimitiveCave* self;
    self = (RenderPrimitiveCave *)data;
    
    Py_DECREF(self->skylight);
    Py_XDECREF(self->left_skylight);
    Py_XDECREF(self->right_skylight);
    Py_XDECREF(self->up_left_skylight);
    Py_XDECREF(self->up_right_skylight);
    
    if (self->only_lit) {
        Py_DECREF(self->blocklight);
        Py_XDECREF(self->left_blocklight);
        Py_XDECREF(self->right_blocklight);
        Py_XDECREF(self->up_left_blocklight);
        Py_XDECREF(self->up_right_blocklight);
    }
}

RenderPrimitiveInterface primitive_cave = {
    "cave", sizeof(RenderPrimitiveCave),
    cave_start,
    cave_finish,
    cave_occluded,
    cave_hidden,
    NULL,
};
