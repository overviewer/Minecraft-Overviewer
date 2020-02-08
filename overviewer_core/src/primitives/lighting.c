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
#include "lighting.h"
#include "../block_class.h"
#include "../mc_id.h"
#include "../overviewer.h"

/* figures out the color from a given skylight and blocklight,
   used in lighting calculations */
static void
calculate_light_color(void* data,
                      uint8_t skylight, uint8_t blocklight,
                      uint8_t* r, uint8_t* g, uint8_t* b) {
    uint8_t v = 255 * powf(0.8f, 15.0 - OV_MAX(blocklight, skylight));
    *r = v;
    *g = v;
    *b = v;
}

/* fancy version that uses the colored light texture */
static void
calculate_light_color_fancy(void* data,
                            uint8_t skylight, uint8_t blocklight,
                            uint8_t* r, uint8_t* g, uint8_t* b) {
    RenderPrimitiveLighting* mode = (RenderPrimitiveLighting*)(data);
    uint32_t index;
    PyObject* color;

    blocklight = OV_MAX(blocklight, skylight);

    index = skylight + blocklight * 16;
    color = PySequence_GetItem(mode->lightcolor, index);

    *r = PyLong_AsLong(PyTuple_GET_ITEM(color, 0));
    *g = PyLong_AsLong(PyTuple_GET_ITEM(color, 1));
    *b = PyLong_AsLong(PyTuple_GET_ITEM(color, 2));

    Py_DECREF(color);
}

/* figures out the color from a given skylight and blocklight, used in
   lighting calculations -- note this is *different* from the one above
   (the "skylight - 11" part)
*/
static void
calculate_light_color_night(void* data,
                            uint8_t skylight, uint8_t blocklight,
                            uint8_t* r, uint8_t* g, uint8_t* b) {
    uint8_t v = 255 * powf(0.8f, 15.0 - OV_MAX(blocklight, skylight - 11));
    *r = v;
    *g = v;
    *b = v;
}

/* fancy night version that uses the colored light texture */
static void
calculate_light_color_fancy_night(void* data,
                                  uint8_t skylight, uint8_t blocklight,
                                  uint8_t* r, uint8_t* g, uint8_t* b) {
    RenderPrimitiveLighting* mode = (RenderPrimitiveLighting*)(data);
    uint32_t index;
    PyObject* color;

    index = skylight + blocklight * 16;
    color = PySequence_GetItem(mode->lightcolor, index);

    *r = PyLong_AsLong(PyTuple_GET_ITEM(color, 0));
    *g = PyLong_AsLong(PyTuple_GET_ITEM(color, 1));
    *b = PyLong_AsLong(PyTuple_GET_ITEM(color, 2));

    Py_DECREF(color);
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

uint8_t
estimate_blocklevel(RenderPrimitiveLighting* self, RenderState* state,
                    int32_t x, int32_t y, int32_t z, bool* authoratative) {

    /* placeholders for later data arrays, coordinates */
    mc_block_t block;
    uint8_t blocklevel;
    uint32_t average_count = 0, average_gather = 0, coeff = 0;

    /* defaults to "guess" until told otherwise */
    if (authoratative)
        *authoratative = false;

    block = get_data(state, BLOCKS, x, y, z);

    if (authoratative == NULL) {
        bool auth;

        /* iterate through all surrounding blocks to take an average */
        int32_t dx, dy, dz, local_block;
        for (dx = -1; dx <= 1; dx += 2) {
            for (dy = -1; dy <= 1; dy += 2) {
                for (dz = -1; dz <= 1; dz += 2) {
                    coeff = estimate_blocklevel(self, state, x + dx, y + dy, z + dz, &auth);
                    local_block = get_data(state, BLOCKS, x + dx, y + dy, z + dz);
                    /* only add if the block is transparent, this seems to look better than
                       using every block */
                    if (auth && is_transparent(local_block)) {
                        average_gather += coeff;
                        average_count++;
                    }
                }
            }
        }
    }

    /* only return the average if at least one was authoratative */
    if (average_count > 0) {
        return average_gather / average_count;
    }

    blocklevel = get_data(state, BLOCKLIGHT, x, y, z);

    /* no longer a guess */
    if (!block_class_is_subset(block, block_class_alt_height, block_class_alt_height_len) && authoratative) {
        *authoratative = 1;
    }

    return blocklevel;
}

inline void
get_lighting_color(RenderPrimitiveLighting* self, RenderState* state,
                   int32_t x, int32_t y, int32_t z,
                   uint8_t* r, uint8_t* g, uint8_t* b) {

    /* placeholders for later data arrays, coordinates */
    mc_block_t block;
    uint8_t skylevel, blocklevel;

    block = get_data(state, BLOCKS, x, y, z);
    skylevel = get_data(state, SKYLIGHT, x, y, z);
    blocklevel = get_data(state, BLOCKLIGHT, x, y, z);

    /* special half-step handling, stairs handling */
    /* Anvil also needs to be here, blockid 145 */
    if (block_class_is_subset(block, block_class_alt_height, block_class_alt_height_len) || block == block_anvil) {
        uint32_t upper_block;

        /* stairs and half-blocks take the skylevel from the upper block if it's transparent */
        int32_t upper_counter = 0;
        /* but if the upper_block is one of these special half-steps, we need to look at *its* upper_block */
        do {
            upper_counter++;
            upper_block = get_data(state, BLOCKS, x, y + upper_counter, z);
        } while (block_class_is_subset(upper_block, block_class_alt_height, block_class_alt_height_len));
        if (is_transparent(upper_block)) {
            skylevel = get_data(state, SKYLIGHT, x, y + upper_counter, z);
        } else {
            skylevel = 15;
        }

        /* the block has a bad blocklevel, estimate it from neigborhood
         * use given coordinates, no local ones! */
        blocklevel = estimate_blocklevel(self, state, x, y, z, NULL);
    }

    if (block_class_is_subset(block, (mc_block_t[]){block_flowing_lava, block_lava}, 2)) {
        /* lava blocks should always be lit! */
        *r = 255;
        *g = 255;
        *b = 255;
        return;
    }

    self->calculate_light_color(self, OV_MIN(skylevel, 15), OV_MIN(blocklevel, 15), r, g, b);
}

/* does per-face occlusion checking for do_shading_with_mask */
inline bool
lighting_is_face_occluded(RenderState* state, bool skip_sides, int32_t x, int32_t y, int32_t z) {
    /* first, check for occlusion if the block is in the local chunk */
    if (x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16) {
        mc_block_t block = getArrayShort3D(state->blocks, x, y, z);

        if (!is_transparent(block) && !render_mode_hidden(state->rendermode, x, y, z)) {
            /* this face isn't visible, so don't draw anything */
            return true;
        }
    } else if (!skip_sides) {
        mc_block_t block = get_data(state, BLOCKS, x, y, z);
        if (!is_transparent(block)) {
            /* the same thing but for adjacent chunks, this solves an
               ugly black doted line between chunks in night rendermode.
               This wouldn't be necessary if the textures were truly
               tessellate-able */
            return true;
        }
    }
    return false;
}

/* shades the drawn block with the given facemask, based on the
   lighting results from (x, y, z) */
static inline void
do_shading_with_mask(RenderPrimitiveLighting* self, RenderState* state,
                     int32_t x, int32_t y, int32_t z, PyObject* mask) {
    uint8_t r, g, b;
    float comp_strength;

    /* check occlusion */
    if (lighting_is_face_occluded(state, self->skip_sides, x, y, z))
        return;

    get_lighting_color(self, state, x, y, z, &r, &g, &b);
    comp_strength = 1.0 - self->strength;

    r += (255 - r) * comp_strength;
    g += (255 - g) * comp_strength;
    b += (255 - b) * comp_strength;

    tint_with_mask(state->img, r, g, b, 255, mask, state->imgx, state->imgy, 0, 0);
}

static bool
lighting_start(void* data, RenderState* state, PyObject* support) {
    RenderPrimitiveLighting* self;
    self = (RenderPrimitiveLighting*)data;

    /* don't skip sides by default */
    self->skip_sides = false;

    if (!render_mode_parse_option(support, "strength", "f", &(self->strength)))
        return true;
    if (!render_mode_parse_option(support, "night", "p", &(self->night)))
        return true;
    if (!render_mode_parse_option(support, "color", "p", &(self->color)))
        return true;

    self->facemasks_py = PyObject_GetAttrString(support, "facemasks");
    // borrowed references, don't need to be decref'd
    self->facemasks[0] = PyTuple_GetItem(self->facemasks_py, 0);
    self->facemasks[1] = PyTuple_GetItem(self->facemasks_py, 1);
    self->facemasks[2] = PyTuple_GetItem(self->facemasks_py, 2);

    if (self->night) {
        self->calculate_light_color = calculate_light_color_night;
    } else {
        self->calculate_light_color = calculate_light_color;
    }

    if (self->color) {
        self->lightcolor = PyObject_CallMethod(state->textures, "load_light_color", "");
        if (self->lightcolor == Py_None) {
            Py_DECREF(self->lightcolor);
            self->lightcolor = NULL;
            self->color = false;
        } else {
            if (self->night) {
                self->calculate_light_color = calculate_light_color_fancy_night;
            } else {
                self->calculate_light_color = calculate_light_color_fancy;
            }
        }
    } else {
        self->lightcolor = NULL;
    }

    return false;
}

static void
lighting_finish(void* data, RenderState* state) {
    RenderPrimitiveLighting* self = (RenderPrimitiveLighting*)data;

    Py_DECREF(self->facemasks_py);
}

static void
lighting_draw(void* data, RenderState* state, PyObject* src, PyObject* mask, PyObject* mask_light) {
    RenderPrimitiveLighting* self;
    int32_t x, y, z;

    self = (RenderPrimitiveLighting*)data;
    x = state->x, y = state->y, z = state->z;

    if (block_class_is_subset(state->block, (mc_block_t[]){block_flowing_water, block_water}, 2)) { /* special case for water */
        /* looks like we need a new case for lighting, there are
         * blocks that are transparent for occlusion calculations and
         * need per-face shading if the face is drawn. */
        if ((state->block_pdata & 16) == 16) {
            do_shading_with_mask(self, state, x, y + 1, z, self->facemasks[0]);
        }
        if ((state->block_pdata & 2) == 2) { /* bottom left */
            do_shading_with_mask(self, state, x - 1, y, z, self->facemasks[1]);
        }
        if ((state->block_pdata & 4) == 4) { /* bottom right */
            do_shading_with_mask(self, state, x, y, z + 1, self->facemasks[2]);
        }
        /* leaves, ice, and pistons are transparent for occlusion calculations
         * but they need per face-shading to look as in game */
    } else if (is_transparent(state->block) &&
               !block_class_is_subset(state->block, (mc_block_t[]){block_leaves, block_ice, block_piston, block_sticky_piston}, 4)) {
        /* transparent: do shading on whole block */
        do_shading_with_mask(self, state, x, y, z, mask_light);
    } else {
        /* opaque: do per-face shading */
        do_shading_with_mask(self, state, x, y + 1, z, self->facemasks[0]);
        do_shading_with_mask(self, state, x - 1, y, z, self->facemasks[1]);
        do_shading_with_mask(self, state, x, y, z + 1, self->facemasks[2]);
    }
}

RenderPrimitiveInterface primitive_lighting = {
    "lighting",
    sizeof(RenderPrimitiveLighting),
    lighting_start,
    lighting_finish,
    NULL,
    NULL,
    lighting_draw,
};
