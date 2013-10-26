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
#include "biomes.h"

typedef struct {
    int use_biomes;
    /* grasscolor and foliagecolor lookup tables */
    PyObject *grasscolor, *foliagecolor, *watercolor;
    /* biome-compatible grass/leaf textures */
    PyObject *grass_texture;
} PrimitiveBase;


static int
base_start(void *data, RenderState *state, PyObject *support) {
    PrimitiveBase *self = (PrimitiveBase *)data;
    
    if (!render_mode_parse_option(support, "biomes", "i", &(self->use_biomes)))
        return 1;
    
    /* biome-compliant grass mask (includes sides!) */
    self->grass_texture = PyObject_GetAttrString(state->textures, "biome_grass_texture");
    
    /* color lookup tables */
    self->foliagecolor = PyObject_CallMethod(state->textures, "load_foliage_color", "");
    self->grasscolor = PyObject_CallMethod(state->textures, "load_grass_color", "");
    self->watercolor = PyObject_CallMethod(state->textures, "load_water_color", "");
    
    return 0;
}

static void
base_finish(void *data, RenderState *state) {
    PrimitiveBase *self = (PrimitiveBase *)data;
    
    Py_DECREF(self->foliagecolor);
    Py_DECREF(self->grasscolor);
    Py_DECREF(self->watercolor);
    Py_DECREF(self->grass_texture);
}

static int
base_occluded(void *data, RenderState *state, int x, int y, int z) {
    if ( (x != 0) && (y != 15) && (z != 15) &&
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
base_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    PrimitiveBase *self = (PrimitiveBase *)data;

    /* in order to detect top parts of doublePlant grass & ferns */
    unsigned char below_block = get_data(state, BLOCKS, state->x, state->y-1, state->z);
    unsigned char below_data = get_data(state, DATA, state->x, state->y-1, state->z);

    /* draw the block! */
    alpha_over(state->img, src, mask, state->imgx, state->imgy, 0, 0);
    
    /* check for biome-compatible blocks
     *
     * NOTES for maintainers:
     *
     * To add a biome-compatible block, add an OR'd condition to this
     * following if block, and a case to the switch statement to handle biome
     * coloring.
     *
     * Make sure that in textures.py, the generated textures are the
     * biome-compliant ones! The tinting is now all done here.
     */
    if (/* grass, but not snowgrass */
        (state->block == 2 && get_data(state, BLOCKS, state->x, state->y+1, state->z) != 78) ||
        /* water */
        state->block == 8 || state->block == 9 ||
        /* leaves */
        state->block == 18 || state->block == 161 ||
        /* tallgrass, but not dead shrubs */
        (state->block == 31 && state->block_data != 0) ||
        /* pumpkin/melon stem, not fully grown. Fully grown stems
         * get constant brown color (see textures.py) */
        (((state->block == 104) || (state->block == 105)) && (state->block_data != 7)) ||
        /* vines */
        state->block == 106 ||
        /* lily pads */
        state->block == 111 ||
        /* doublePlant grass & ferns */
        (state->block == 175 && (state->block_data == 2 || state->block_data == 3)) ||
        /* doublePlant grass & ferns tops */
        (state->block == 175 && below_block == 175 && (below_data == 2 || below_data == 3)) )
    {
        /* do the biome stuff! */
        PyObject *facemask = mask;
        unsigned char r = 255, g = 255, b = 255;
        PyObject *color_table = NULL;
        unsigned char flip_xy = 0;
        
        if (state->block == 2) {
            /* grass needs a special facemask */
            facemask = self->grass_texture;
        }

        switch (state->block) {
        case 2:
            /* grass */
            color_table = self->grasscolor;
            break;
        case 8:
        case 9:
            /* water */
            color_table = self->watercolor;
            break;
        case 18:
        case 161:
            /* leaves */
            color_table = self->foliagecolor;
            if (state->block_data == 2)
            {
                /* birch!
                   birch foliage color is flipped XY-ways */
                flip_xy = 1;
            }
            break;
        case 31:
            /* tall grass */
            color_table = self->grasscolor;
            break;
        case 104:
            /* pumpkin stem */
            color_table = self->grasscolor;
            break;
        case 105:
            /* melon stem */
            color_table = self->grasscolor;
            break;
        case 106:
            /* vines */
            color_table = self->grasscolor;
            break;
        case 111:
            /* lily pads */
            color_table = self->grasscolor;
            break;
        case 175:
            /* doublePlant grass & ferns */
            color_table = self->grasscolor;
            break;
        default:
            break;
        };
            
        if (color_table) {
            unsigned char biome;
            int dx, dz;
            unsigned char tablex, tabley;
            float temp = 0.0, rain = 0.0;
            unsigned int multr = 0, multg = 0, multb = 0;
            int tmp;
            PyObject *color = NULL;
            
            if (self->use_biomes) {
                /* average over all neighbors */
                for (dx = -1; dx <= 1; dx++) {
                    for (dz = -1; dz <= 1; dz++) {
                        biome = get_data(state, BIOMES, state->x + dx, state->y, state->z + dz);
                        if (biome >= NUM_BIOMES) {
                            /* note -- biome 255 shows up on map borders.
                               who knows what it is? certainly not I.
                            */
                            biome = DEFAULT_BIOME; /* forest -- reasonable default */
                        }
                    
                        temp += biome_table[biome].temperature;
                        rain += biome_table[biome].rainfall;
                        multr += biome_table[biome].r;
                        multg += biome_table[biome].g;
                        multb += biome_table[biome].b;
                    }
                }
                
                temp /= 9.0;
                rain /= 9.0;
                multr /= 9;
                multg /= 9;
                multb /= 9;
            } else {
                /* don't use biomes, just use the default */
                temp = biome_table[DEFAULT_BIOME].temperature;
                rain = biome_table[DEFAULT_BIOME].rainfall;
                multr = biome_table[DEFAULT_BIOME].r;
                multg = biome_table[DEFAULT_BIOME].g;
                multb = biome_table[DEFAULT_BIOME].b;
            }
            
            /* second coordinate is actually scaled to fit inside the triangle
               so store it in rain */
            rain *= temp;
            
            /* make sure they're sane */
            temp = CLAMP(temp, 0.0, 1.0);
            rain = CLAMP(rain, 0.0, 1.0);
            
            /* convert to x/y coordinates in color table */
            tablex = 255 - (255 * temp);
            tabley = 255 - (255 * rain);
            if (flip_xy) {
                unsigned char tmp = 255 - tablex;
                tablex = 255 - tabley;
                tabley = tmp;
            }
            
            /* look up color! */
            color = PySequence_GetItem(color_table, tabley * 256 + tablex);
            r = PyInt_AsLong(PyTuple_GET_ITEM(color, 0));
            g = PyInt_AsLong(PyTuple_GET_ITEM(color, 1));
            b = PyInt_AsLong(PyTuple_GET_ITEM(color, 2));
            Py_DECREF(color);
            
            /* do the after-coloration */
            r = MULDIV255(r, multr, tmp);
            g = MULDIV255(g, multg, tmp);
            b = MULDIV255(b, multb, tmp);
        }
        
        /* final coloration */
        tint_with_mask(state->img, r, g, b, 255, facemask, state->imgx, state->imgy, 0, 0);
    }
}

RenderPrimitiveInterface primitive_base = {
    "base", sizeof(PrimitiveBase),
    base_start,
    base_finish,
    base_occluded,
    NULL,
    base_draw,
};
