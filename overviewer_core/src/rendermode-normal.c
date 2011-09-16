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
rendermode_normal_start(void *data, RenderState *state, PyObject *options) {
    PyObject *chunk_x_py, *chunk_y_py, *world, *use_biomes, *worlddir;
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    /* load up the given options, first */
    
    self->edge_opacity = 0.15;
    if (!render_mode_parse_option(options, "edge_opacity", "f", &(self->edge_opacity)))
        return 1;
    
    self->min_depth = 0;
    if (!render_mode_parse_option(options, "min_depth", "I", &(self->min_depth)))
        return 1;

    self->max_depth = 127;
    if (!render_mode_parse_option(options, "max_depth", "I", &(self->max_depth)))
        return 1;

    self->height_fading = 0;
    if (!render_mode_parse_option(options, "height_fading", "i", &(self->height_fading)))
        return 1;
    
    if (self->height_fading) {
        self->black_color = PyObject_GetAttrString(state->chunk, "black_color");
        self->white_color = PyObject_GetAttrString(state->chunk, "white_color");
    }
    
    /* biome-compliant grass mask (includes sides!) */
    self->grass_texture = PyObject_GetAttrString(state->textures, "biome_grass_texture");

    chunk_x_py = PyObject_GetAttrString(state->self, "chunkX");
    chunk_y_py = PyObject_GetAttrString(state->self, "chunkY");
    
    /* careful now -- C's % operator works differently from python's
       we can't just do x % 32 like we did before */
    self->chunk_x = PyInt_AsLong(chunk_x_py);
    self->chunk_y = PyInt_AsLong(chunk_y_py);
    
    while (self->chunk_x < 0)
        self->chunk_x += 32;
    while (self->chunk_y < 0)
        self->chunk_y += 32;
    
    self->chunk_x %= 32;
    self->chunk_y %= 32;
    
    /* fetch the biome data from textures.py, if needed */
    world = PyObject_GetAttrString(state->self, "world");
    worlddir = PyObject_GetAttrString(world, "worlddir");
    use_biomes = PyObject_GetAttrString(world, "useBiomeData");
    Py_DECREF(world);
    
    if (PyObject_IsTrue(use_biomes)) {
        self->biome_data = PyObject_CallMethod(state->textures, "getBiomeData", "OOO",
                                               worlddir, chunk_x_py, chunk_y_py);
        if (self->biome_data == Py_None) {
            Py_DECREF(self->biome_data);
            self->biome_data = NULL;
            self->foliagecolor = NULL;
            self->grasscolor = NULL;
        } else {
            self->foliagecolor = PyObject_GetAttrString(state->textures, "foliagecolor");
            self->grasscolor = PyObject_GetAttrString(state->textures, "grasscolor");
        }
    } else {
        self->biome_data = NULL;
        self->foliagecolor = NULL;
        self->grasscolor = NULL;
    }
    
    Py_DECREF(use_biomes);
    Py_DECREF(worlddir);
    Py_DECREF(chunk_x_py);
    Py_DECREF(chunk_y_py);
    
    return 0;
}

static void
rendermode_normal_finish(void *data, RenderState *state) {
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    Py_XDECREF(self->biome_data);
    Py_XDECREF(self->foliagecolor);
    Py_XDECREF(self->grasscolor);
    Py_XDECREF(self->grass_texture);
    Py_XDECREF(self->black_color);
    Py_XDECREF(self->white_color);
}

static int
rendermode_normal_occluded(void *data, RenderState *state, int x, int y, int z) {
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !render_mode_hidden(state->rendermode, x-1, y, z) &&
         !render_mode_hidden(state->rendermode, x, y, z+1) &&
         !render_mode_hidden(state->rendermode, x, y+1, z) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }

    return 0;
}

static int
rendermode_normal_hidden(void *data, RenderState *state, int x, int y, int z) {
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    if (z > self->max_depth || z < self->min_depth) {
        return 1;
    }

    return 0;
}

static void
rendermode_normal_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    RenderModeNormal *self = (RenderModeNormal *)data;

    /* draw the block! */
    alpha_over(state->img, src, mask, state->imgx, state->imgy, 0, 0);
    
    /* check for biome-compatible blocks
     *
     * NOTES for maintainers:
     *
     * To add a biome-compatible block, add an OR'd condition to this
     * following if block, a case to the first switch statement to handle when
     * biome info IS available, and another case to the second switch
     * statement for when biome info ISN'T available.
     *
     * Make sure that in textures.py, the generated textures are the
     * biome-compliant ones! The tinting is now all done here.
     */
    if (/* grass, but not snowgrass */
        (state->block == 2 && !(state->z < 127 && getArrayByte3D(state->blocks, state->x, state->y, state->z+1) == 78)) ||
        /* leaves */
        state->block == 18 ||
        /* tallgrass, but not dead shrubs */
        (state->block == 31 && state->block_data != 0) ||
        /* pumpkin/melon stem, not fully grown. Fully grown stems
         * get constant brown color (see textures.py) */
        (((state->block == 104) || (state->block == 105)) && (state->block_data != 7)) ||
        /* vines */
        state->block == 106)
    {
        /* do the biome stuff! */
        PyObject *facemask = mask;
        unsigned char r, g, b;
        
        if (state->block == 2) {
            /* grass needs a special facemask */
            facemask = self->grass_texture;
        }
        
        if (self->biome_data) {
            /* we have data, so use it! */
            unsigned int index;
            PyObject *color = NULL;
            
            index = ((self->chunk_y * 16) + state->y) * 16 * 32 + (self->chunk_x * 16) + state->x;
            index = big_endian_ushort(getArrayShort1D(self->biome_data, index));
            
            switch (state->block) {
            case 2:
                /* grass */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 18:
                /* leaves */
                color = PySequence_GetItem(self->foliagecolor, index);
                break;
            case 31:
                /* tall grass */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 104:
                /* pumpkin stem */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 105:
                /* melon stem */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 106:
                /* vines */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            default:
                break;
            };
            
            if (color)
            {
                /* we've got work to do */
                
                r = PyInt_AsLong(PyTuple_GET_ITEM(color, 0));
                g = PyInt_AsLong(PyTuple_GET_ITEM(color, 1));
                b = PyInt_AsLong(PyTuple_GET_ITEM(color, 2));
                Py_DECREF(color);
            }
        } else {
           if (state->block == 2 || state->block == 31 ||
               state->block == 104 || state->block == 105)
               /* grass and pumpkin/melon stems */
            {
                r = 115;
                g = 175;
                b = 71;
            }
            if (state->block == 18 || state->block == 106) /* leaves and vines */
            {
                r = 37;
                g = 118;
                b = 25;
            }
        }
        
        tint_with_mask(state->img, r, g, b, 255, facemask, state->imgx, state->imgy, 0, 0);
    }
    
    if (self->height_fading) {
        /* do some height fading */
        PyObject *height_color = self->white_color;
        /* negative alpha => darkness, positive => light */
        float alpha = (1.0 / (1 + expf((70 - state->z) / 11.0))) * 0.6 - 0.55;
        
        if (alpha < 0.0) {
            alpha *= -1;
            height_color = self->black_color;
        }
        
        alpha_over_full(state->img, height_color, mask_light, alpha, state->imgx, state->imgy, 0, 0);
    }
    

    /* Draw some edge lines! */
    // draw.line(((imgx+12,imgy+increment), (imgx+22,imgy+5+increment)), fill=(0,0,0), width=1)
    if (state->block == 44 || state->block == 78 || !is_transparent(state->block)) {
        Imaging img_i = imaging_python_to_c(state->img);
        unsigned char ink[] = {0, 0, 0, 255 * self->edge_opacity};

        int increment=0;
        if (state->block == 44)  // half-step
            increment=6;
        else if ((state->block == 78) || (state->block == 93) || (state->block == 94)) // snow, redstone repeaters (on and off)
            increment=9;

        if ((state->x == 15) && (state->up_right_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_right_blocks, 0, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        } else if (state->x != 15) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x+1, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        }
        // if y != 0 and blocks[x,y-1,z] == 0

        // chunk boundries are annoying
        if ((state->y == 0) && (state->up_left_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_left_blocks, state->x, 15, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        } else if (state->y != 0) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x, state->y-1, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                // draw.line(((imgx,imgy+6+increment), (imgx+12,imgy+increment)), fill=(0,0,0), width=1)
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        }
    }
}

const RenderModeOption rendermode_normal_options[] = {
    {"edge_opacity", "darkness of the edge lines, from 0.0 to 1.0 (default: 0.15)"},
    {"min_depth", "lowest level of blocks to render (default: 0)"},
    {"max_depth", "highest level of blocks to render (default: 127)"},
    {"height_fading", "darken or lighten blocks based on height (default: False)"},
    {NULL, NULL}
};

RenderModeInterface rendermode_normal = {
    "normal", "Normal",
    "nothing special, just render the blocks",
    rendermode_normal_options,
    NULL,
    sizeof(RenderModeNormal),
    rendermode_normal_start,
    rendermode_normal_finish,
    rendermode_normal_occluded,
    rendermode_normal_hidden,
    rendermode_normal_draw,
};
