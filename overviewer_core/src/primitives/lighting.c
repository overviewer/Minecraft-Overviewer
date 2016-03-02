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
#include "lighting.h"
#include <math.h>

/* figures out the color from a given skylight and blocklight,
   used in lighting calculations */
static void
calculate_light_color(void *data,
                      unsigned char skylight, unsigned char blocklight,
                      unsigned char *r, unsigned char *g, unsigned char *b) {
    unsigned char v = 255 * powf(0.8f, 15.0 - MAX(blocklight, skylight));
    *r = v;
    *g = v;
    *b = v;
}

/* fancy version that uses the colored light texture */
static void
calculate_light_color_fancy(void *data,
                            unsigned char skylight, unsigned char blocklight,
                            unsigned char *r, unsigned char *g, unsigned char *b) {
    RenderPrimitiveLighting *mode = (RenderPrimitiveLighting *)(data);
    unsigned int index;
    PyObject *color;
    
    blocklight = MAX(blocklight, skylight);
    
    index = skylight + blocklight * 16;
    color = PySequence_GetItem(mode->lightcolor, index);
    
    *r = PyInt_AsLong(PyTuple_GET_ITEM(color, 0));
    *g = PyInt_AsLong(PyTuple_GET_ITEM(color, 1));
    *b = PyInt_AsLong(PyTuple_GET_ITEM(color, 2));
    
    Py_DECREF(color);
}

/* figures out the color from a given skylight and blocklight, used in
   lighting calculations -- note this is *different* from the one above
   (the "skylight - 11" part)
*/
static void
calculate_light_color_night(void *data,
                            unsigned char skylight, unsigned char blocklight,
                            unsigned char *r, unsigned char *g, unsigned char *b) {
    unsigned char v = 255 * powf(0.8f, 15.0 - MAX(blocklight, skylight - 11));
    *r = v;
    *g = v;
    *b = v;
}

/* fancy night version that uses the colored light texture */
static void
calculate_light_color_fancy_night(void *data,
                                  unsigned char skylight, unsigned char blocklight,
                                  unsigned char *r, unsigned char *g, unsigned char *b) {
    RenderPrimitiveLighting *mode = (RenderPrimitiveLighting *)(data);
    unsigned int index;
    PyObject *color;
    
    index = skylight + blocklight * 16;
    color = PySequence_GetItem(mode->lightcolor, index);
    
    *r = PyInt_AsLong(PyTuple_GET_ITEM(color, 0));
    *g = PyInt_AsLong(PyTuple_GET_ITEM(color, 1));
    *b = PyInt_AsLong(PyTuple_GET_ITEM(color, 2));
    
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

unsigned char
estimate_blocklevel(RenderPrimitiveLighting *self, RenderState *state,
                         int x, int y, int z, int *authoratative) {

    /* placeholders for later data arrays, coordinates */
    unsigned char block, blocklevel;
    unsigned int average_count = 0, average_gather = 0, coeff = 0;

    /* defaults to "guess" until told otherwise */
    if (authoratative)
        *authoratative = 0;
    
    block = get_data(state, BLOCKS, x, y, z);
    
    if (authoratative == NULL) {
        int auth;
        
        /* iterate through all surrounding blocks to take an average */
        int dx, dy, dz, local_block;
        for (dx = -1; dx <= 1; dx += 2) {
            for (dy = -1; dy <= 1; dy += 2) {
                for (dz = -1; dz <= 1; dz += 2) {
                    coeff = estimate_blocklevel(self, state, x+dx, y+dy, z+dz, &auth);
                    local_block = get_data(state, BLOCKS, x+dx, y+dy, z+dz);
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
    if (!(block == 44 || block == 53 || block == 67 || block == 108 || block == 109 || block == 180 || block == 182 || block == 205) && authoratative) {
        *authoratative = 1;
    }
    
    return blocklevel;
}

inline void
get_lighting_color(RenderPrimitiveLighting *self, RenderState *state,
                   int x, int y, int z,
                   unsigned char *r, unsigned char *g, unsigned char *b) {

    /* placeholders for later data arrays, coordinates */
    unsigned char block, skylevel, blocklevel;
    
    block = get_data(state, BLOCKS, x, y, z);
    skylevel = get_data(state, SKYLIGHT, x, y, z);
    blocklevel = get_data(state, BLOCKLIGHT, x, y, z);

    /* special half-step handling, stairs handling */
    /* Anvil also needs to be here, blockid 145 */
    if (block == 44 || block == 53 || block == 67 || block == 108 || block == 109 || block == 114 ||
        block == 128 || block == 134 || block == 135 || block == 136 || block == 145 || block == 156 ||
        block == 163 || block == 164 || block == 180 || block == 182 || block == 203 || block == 205) {
        unsigned int upper_block;
        
        /* stairs and half-blocks take the skylevel from the upper block if it's transparent */
        int upper_counter = 0;
        /* but if the upper_block is one of these special half-steps, we need to look at *its* upper_block */
        do {
            upper_counter++; 
            upper_block = get_data(state, BLOCKS, x, y + upper_counter, z);
        } while (upper_block == 44 || upper_block == 53 || upper_block == 67 || upper_block == 108 ||
                 upper_block == 109 || upper_block == 114 || upper_block == 128 || upper_block == 134 ||
                 upper_block == 135 || upper_block == 136 || upper_block == 156 || upper_block == 163 ||
                 upper_block == 164 || upper_block == 180 || upper_block == 182 || upper_block == 203 || upper_block == 205);
        if (is_transparent(upper_block)) {
            skylevel = get_data(state, SKYLIGHT, x, y + upper_counter, z);
        } else {
            skylevel = 15;
        }
        
        /* the block has a bad blocklevel, estimate it from neigborhood
         * use given coordinates, no local ones! */
        blocklevel = estimate_blocklevel(self, state, x, y, z, NULL);

    }
    
    if (block == 10 || block == 11) {
        /* lava blocks should always be lit! */
        *r = 255;
        *g = 255;
        *b = 255;
        return;
    }
    
    self->calculate_light_color(self, MIN(skylevel, 15), MIN(blocklevel, 15), r, g, b);
}

/* does per-face occlusion checking for do_shading_with_mask */
inline int
lighting_is_face_occluded(RenderState *state, int skip_sides, int x, int y, int z) {
    /* first, check for occlusion if the block is in the local chunk */
    if (x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16) {
        unsigned short block = getArrayShort3D(state->blocks, x, y, z);
        
        if (!is_transparent(block) && !render_mode_hidden(state->rendermode, x, y, z)) {
            /* this face isn't visible, so don't draw anything */
            return 1;
        }
    } else if (!skip_sides) {
        unsigned short block = get_data(state, BLOCKS, x, y, z);
        if (!is_transparent(block)) {
            /* the same thing but for adjacent chunks, this solves an
               ugly black doted line between chunks in night rendermode.
               This wouldn't be necessary if the textures were truly
               tessellate-able */
               return 1;
        }
    }
    return 0;
}

/* shades the drawn block with the given facemask, based on the
   lighting results from (x, y, z) */
static inline void
do_shading_with_mask(RenderPrimitiveLighting *self, RenderState *state,
                     int x, int y, int z, PyObject *mask) {
    unsigned char r, g, b;
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

static int
lighting_start(void *data, RenderState *state, PyObject *support) {
    RenderPrimitiveLighting* self;
    self = (RenderPrimitiveLighting *)data;
    
    /* don't skip sides by default */
    self->skip_sides = 0;

    if (!render_mode_parse_option(support, "strength", "f", &(self->strength)))
        return 1;
    if (!render_mode_parse_option(support, "night", "i", &(self->night)))
        return 1;
    if (!render_mode_parse_option(support, "color", "i", &(self->color)))
        return 1;
    
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
            self->color = 0;
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
    
    return 0;
}

static void
lighting_finish(void *data, RenderState *state) {
    RenderPrimitiveLighting *self = (RenderPrimitiveLighting *)data;
    
    Py_DECREF(self->facemasks_py);
}

static void
lighting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderPrimitiveLighting* self;
    int x, y, z;

    self = (RenderPrimitiveLighting *)data;
    x = state->x, y = state->y, z = state->z;
    
    if ((state->block == 8) || (state->block == 9)) { /* special case for water */
        /* looks like we need a new case for lighting, there are
         * blocks that are transparent for occlusion calculations and
         * need per-face shading if the face is drawn. */
        if ((state->block_pdata & 16) == 16) {
            do_shading_with_mask(self, state, x, y+1, z, self->facemasks[0]);
        }
        if ((state->block_pdata & 2) == 2) { /* bottom left */
            do_shading_with_mask(self, state, x-1, y, z, self->facemasks[1]);
        }
        if ((state->block_pdata & 4) == 4) { /* bottom right */
            do_shading_with_mask(self, state, x, y, z+1, self->facemasks[2]);
        }
        /* leaves and ice are transparent for occlusion calculations but they 
         * per face-shading to look as in game */
    } else if (is_transparent(state->block) && (state->block != 18) && (state->block != 79)) {
        /* transparent: do shading on whole block */
        do_shading_with_mask(self, state, x, y, z, mask_light);
    } else {
        /* opaque: do per-face shading */
        do_shading_with_mask(self, state, x, y+1, z, self->facemasks[0]);
        do_shading_with_mask(self, state, x-1, y, z, self->facemasks[1]);
        do_shading_with_mask(self, state, x, y, z+1, self->facemasks[2]);
    }
}

RenderPrimitiveInterface primitive_lighting = {
    "lighting", sizeof(RenderPrimitiveLighting),
    lighting_start,
    lighting_finish,
    NULL,
    NULL,
    lighting_draw,
};
