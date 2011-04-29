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

/* figures out the black_coeff from a given skylight and blocklight,
   used in lighting calculations */
static float calculate_darkness(unsigned char skylight, unsigned char blocklight) {
    return 1.0f - powf(0.8f, 15.0 - MAX(blocklight, skylight));
}

/* loads the appropriate light data for the given (possibly non-local)
 * coordinates, and returns a black_coeff this is exposed, so other (derived)
 * rendermodes can use it
 *  
 * authoratative is a return slot for whether or not this lighting calculation
 * is true, or a guess. If we guessed, *authoratative will be false, but if it
 * was calculated correctly from available light data, it will be true. You
 * may (and probably should) pass NULL.
 */
inline float
get_lighting_coefficient(RenderModeLighting *self, RenderState *state,
                         int x, int y, int z, int *authoratative) {
    
    /* placeholders for later data arrays, coordinates */
    PyObject *blocks = NULL;
    PyObject *skylight = NULL;
    PyObject *blocklight = NULL;
    int local_x = x, local_y = y, local_z = z;
    unsigned char block, skylevel, blocklevel;
    
    /* defaults to "guess" until told otherwise */
    if (authoratative)
        *authoratative = 0;
    
    /* find out what chunk we're in, and translate accordingly */
    if (x >= 0 && y < 16) {
        blocks = state->blocks;
        skylight = self->skylight;
        blocklight = self->blocklight;
    } else if (x < 0) {
        local_x += 16;        
        blocks = state->left_blocks;
        skylight = self->left_skylight;
        blocklight = self->left_blocklight;
    } else if (y >= 16) {
        local_y -= 16;
        blocks = state->right_blocks;
        skylight = self->right_skylight;
        blocklight = self->right_blocklight;
    }
    
    /* make sure we have correctly-ranged coordinates */
    if (!(local_x >= 0 && local_x < 16 &&
          local_y >= 0 && local_y < 16 &&
          local_z >= 0 && local_z < 128)) {
        
        return self->calculate_darkness(15, 0);
    }
    
    /* also, make sure we have enough info to correctly calculate lighting */
    if (blocks == Py_None || blocks == NULL ||
        skylight == Py_None || skylight == NULL ||
        blocklight == Py_None || blocklight == NULL) {
        
        return self->calculate_darkness(15, 0);
    }
    
    block = getArrayByte3D(blocks, local_x, local_y, local_z);
    
    /* if this block is opaque, use a fully-lit coeff instead
       to prevent stippled lines along chunk boundaries! */
    if (!is_transparent(block)) {
        return self->calculate_darkness(15, 0);
    }
    
    /* only do special half-step handling if no authoratative pointer was
       passed in, which is a sign that we're recursing */
    if (block == 44 && authoratative == NULL) {
        float average_gather = 0.0f;
        unsigned int average_count = 0;
        int auth;
        float coeff;
        
        /* iterate through all surrounding blocks to take an average */
        int dx, dy, dz;
        for (dx = -1; dx <= 1; dx += 2) {
            for (dy = -1; dy <= 1; dy += 2) {
                for (dz = -1; dz <= 1; dz += 2) {
                    coeff = get_lighting_coefficient(self, state, x+dx, y+dy, z+dz, &auth);
                    if (auth) {
                        average_gather += coeff;
                        average_count++;
                    }
                }
            }
        }
        
        /* only return the average if at least one was authoratative */
        if (average_count > 0)
            return average_gather / average_count;
    }
    
    if (block == 10 || block == 11) {
        /* lava blocks should always be lit! */
        return 0.0f;
    }
    
    skylevel = getArrayByte3D(skylight, local_x, local_y, local_z);
    blocklevel = getArrayByte3D(blocklight, local_x, local_y, local_z);
    
    /* no longer a guess */
    if (authoratative)
        *authoratative = 1;
    
    return self->calculate_darkness(skylevel, blocklevel);
}

/* shades the drawn block with the given facemask/black_color, based on the
   lighting results from (x, y, z) */
static inline void
do_shading_with_mask(RenderModeLighting *self, RenderState *state,
                     int x, int y, int z, PyObject *mask) {
    float black_coeff;

    /* first, check for occlusion if the block is in the local chunk */
    if (x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 128) {
        unsigned char block = getArrayByte3D(state->blocks, x, y, z);
        if (!is_transparent(block)) {
            /* this face isn't visible, so don't draw anything */
            return;
        }
    }
    
    black_coeff = get_lighting_coefficient(self, state, x, y, z, NULL);
    alpha_over_full(state->img, self->black_color, mask, black_coeff, state->imgx, state->imgy, 0, 0);
}

static int
rendermode_lighting_start(void *data, RenderState *state) {
    RenderModeLighting* self;

    /* first, chain up */
    int ret = rendermode_normal.start(data, state);
    if (ret != 0)
        return ret;
    
    self = (RenderModeLighting *)data;
    
    self->black_color = PyObject_GetAttrString(state->chunk, "black_color");
    self->facemasks_py = PyObject_GetAttrString(state->chunk, "facemasks");
    // borrowed references, don't need to be decref'd
    self->facemasks[0] = PyTuple_GetItem(self->facemasks_py, 0);
    self->facemasks[1] = PyTuple_GetItem(self->facemasks_py, 1);
    self->facemasks[2] = PyTuple_GetItem(self->facemasks_py, 2);
    
    self->skylight = PyObject_GetAttrString(state->self, "skylight");
    self->blocklight = PyObject_GetAttrString(state->self, "blocklight");
    self->left_skylight = PyObject_GetAttrString(state->self, "left_skylight");
    self->left_blocklight = PyObject_GetAttrString(state->self, "left_blocklight");
    self->right_skylight = PyObject_GetAttrString(state->self, "right_skylight");
    self->right_blocklight = PyObject_GetAttrString(state->self, "right_blocklight");
    
    self->calculate_darkness = calculate_darkness;
    
    return 0;
}

static void
rendermode_lighting_finish(void *data, RenderState *state) {
    RenderModeLighting *self = (RenderModeLighting *)data;
    
    Py_DECREF(self->black_color);
    Py_DECREF(self->facemasks_py);
    
    Py_DECREF(self->skylight);
    Py_DECREF(self->blocklight);
    Py_DECREF(self->left_skylight);
    Py_DECREF(self->left_blocklight);
    Py_DECREF(self->right_skylight);
    Py_DECREF(self->right_blocklight);
    
    /* now chain up */
    rendermode_normal.finish(data, state);
}

static int
rendermode_lighting_occluded(void *data, RenderState *state) {
    /* no special occlusion here */
    return rendermode_normal.occluded(data, state);
}

static void
rendermode_lighting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    RenderModeLighting* self;
    int x, y, z;

    /* first, chain up */
    rendermode_normal.draw(data, state, src, mask);
    
    self = (RenderModeLighting *)data;
    x = state->x, y = state->y, z = state->z;
    
    if (is_transparent(state->block)) {
        /* transparent: do shading on whole block */
        do_shading_with_mask(self, state, x, y, z, mask);
    } else {
        /* opaque: do per-face shading */
        do_shading_with_mask(self, state, x, y, z+1, self->facemasks[0]);
        do_shading_with_mask(self, state, x-1, y, z, self->facemasks[1]);
        do_shading_with_mask(self, state, x, y+1, z, self->facemasks[2]);
    }
}

RenderModeInterface rendermode_lighting = {
    "lighting", "draw shadows from the lighting data",
    &rendermode_normal,
    sizeof(RenderModeLighting),
    rendermode_lighting_start,
    rendermode_lighting_finish,
    rendermode_lighting_occluded,
    rendermode_lighting_draw,
};
