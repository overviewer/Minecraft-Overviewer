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

static inline int
touches_light(unsigned int x, unsigned int y, unsigned int z,
              PyObject *light, PyObject *left_light, PyObject *right_light,
              PyObject *up_left_light, PyObject *up_right_light) {
    
    
    if (getArrayByte3D(light, x, y, z+1) != 0) {
        return 1;
    }

    if ((x == 15)) {
        if  (up_right_light != Py_None) {
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
        if  (left_light != Py_None) {
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
        if  (right_light != Py_None) {
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
        if  (up_left_light != Py_None) {
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

static inline int
rendermode_cave_adjacent_occluded(void *data, RenderState *state, int x, int y, int z) {
    /* check for occlusion of edge blocks, using adjacent block data */
    
    if (z != 127) {
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
    }
    
    return 0;
}

static int
rendermode_cave_occluded(void *data, RenderState *state, int x, int y, int z) { 
    /* first, check to see if it's "normally" occluded */
    if (rendermode_lighting.occluded(data, state, x, y, z))
        return 1;

    /* check for normal occlusion */
    /* use ajacent chunks, if not you get blocks spreaded in chunk edges */
    return rendermode_cave_adjacent_occluded(data, state, x, y, z);
}

static int
rendermode_cave_hidden(void *data, RenderState *state, int x, int y, int z) {
    RenderModeCave* self;
    int dz = 0;
    self = (RenderModeCave *)data;
    
    /* first, check to see if it's "normally" hidden */
    if (rendermode_lighting.hidden(data, state, x, y, z))
        return 1;

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
     *
     * We leave out this check otherwise because it's fairly expensive.
     */
    if (self->lighting) {
        if ( (x != 0) && (y != 15) && (z != 127) &&
             !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
             !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
             !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
            return 1;
        }
        
        return rendermode_cave_adjacent_occluded(data, state, x, y, z);
    }
    
    return 0;
}

static int
rendermode_cave_start(void *data, RenderState *state, PyObject *options) {
    RenderModeCave* self;
    int ret;
    self = (RenderModeCave *)data;

    /* first, chain up */
    ret = rendermode_lighting.start(data, state, options);
    if (ret != 0)
        return ret;
    
    self->depth_tinting = 1;
    if (!render_mode_parse_option(options, "depth_tinting", "i", &(self->depth_tinting)))
        return 1;

    self->only_lit = 0;
    if (!render_mode_parse_option(options, "only_lit", "i", &(self->only_lit)))
        return 1;
    
    self->lighting = 0;
    if (!render_mode_parse_option(options, "lighting", "i", &(self->lighting)))
        return 1;
    
    if (self->lighting)
    {
        /* we can't skip lighting the sides in cave mode, it looks too weird */
        self->parent.skip_sides = 0;
    }
    
    /* if there's skylight we are in the surface! */
    self->skylight = PyObject_GetAttrString(state->self, "skylight");
    self->left_skylight = PyObject_GetAttrString(state->self, "left_skylight");
    self->right_skylight = PyObject_GetAttrString(state->self, "right_skylight");
    self->up_left_skylight = PyObject_GetAttrString(state->self, "up_left_skylight");
    self->up_right_skylight = PyObject_GetAttrString(state->self, "up_right_skylight");
    
    if (self->only_lit) {
        self->blocklight = PyObject_GetAttrString(state->self, "blocklight");
        self->left_blocklight = PyObject_GetAttrString(state->self, "left_blocklight");
        self->right_blocklight = PyObject_GetAttrString(state->self, "right_blocklight");
        self->up_left_blocklight = PyObject_GetAttrString(state->self, "up_left_blocklight");
        self->up_right_blocklight = PyObject_GetAttrString(state->self, "up_right_blocklight");
    }

    /* colors for tinting */
    self->depth_colors = PyObject_GetAttrString(state->chunk, "depth_colors");

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
    
    if (self->only_lit) {
        Py_DECREF(self->blocklight);
        Py_DECREF(self->left_blocklight);
        Py_DECREF(self->right_blocklight);
        Py_DECREF(self->up_left_blocklight);
        Py_DECREF(self->up_right_blocklight);
    }
    
    Py_DECREF(self->depth_colors);

    rendermode_lighting.finish(data, state);
}

static void
rendermode_cave_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderModeCave* self;
    int z, r, g, b;
    self = (RenderModeCave *)data;

    z = state->z;
    r = 0, g = 0, b = 0;

    /* draw the normal block */
    if (self->lighting) {
        rendermode_lighting.draw(data, state, src, mask, mask_light);
    } else {
        rendermode_normal.draw(data, state, src, mask, mask_light);
    }

    if (self->depth_tinting) {
        /* get the colors and tint and tint */
        r = PyInt_AsLong(PyList_GetItem(self->depth_colors, 0 + z*3));
        g = PyInt_AsLong(PyList_GetItem(self->depth_colors, 1 + z*3));
        b = PyInt_AsLong(PyList_GetItem(self->depth_colors, 2 + z*3));
        
        tint_with_mask(state->img, r, g, b, 255, mask, state->imgx, state->imgy, 0, 0);
    }

}

const RenderModeOption rendermode_cave_options[] = {
    {"depth_tinting", "tint caves based on how deep they are (default: True)"},
    {"only_lit", "only render lit caves (default: False)"},
    {"lighting", "render caves with lighting enabled (default: False)"},
    {NULL, NULL}
};

RenderModeInterface rendermode_cave = {
    "cave", "Cave",
    "render only caves",
    rendermode_cave_options,
    &rendermode_lighting,
    sizeof(RenderModeCave),
    rendermode_cave_start,
    rendermode_cave_finish,
    rendermode_cave_occluded,
    rendermode_cave_hidden,
    rendermode_cave_draw,
};
