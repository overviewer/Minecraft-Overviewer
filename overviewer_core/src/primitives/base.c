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

#include "../block_class.h"
#include "../mc_id.h"
#include "../overviewer.h"
#include "biomes.h"

typedef struct {
    int32_t use_biomes;
    /* grasscolor and foliagecolor lookup tables */
    PyObject *grasscolor, *foliagecolor, *watercolor;
    /* biome-compatible grass/leaf textures */
    PyObject* grass_texture;
} PrimitiveBase;

static bool
base_start(void* data, RenderState* state, PyObject* support) {
    PrimitiveBase* self = (PrimitiveBase*)data;

    if (!render_mode_parse_option(support, "biomes", "i", &(self->use_biomes)))
        return true;

    /* biome-compliant grass mask (includes sides!) */
    self->grass_texture = PyObject_GetAttrString(state->textures, "biome_grass_texture");

    /* color lookup tables */
    self->foliagecolor = PyObject_CallMethod(state->textures, "load_foliage_color", "");
    self->grasscolor = PyObject_CallMethod(state->textures, "load_grass_color", "");
    self->watercolor = PyObject_CallMethod(state->textures, "load_water_color", "");

    return false;
}

static void
base_finish(void* data, RenderState* state) {
    PrimitiveBase* self = (PrimitiveBase*)data;

    Py_XDECREF(self->foliagecolor);
    Py_XDECREF(self->grasscolor);
    Py_XDECREF(self->watercolor);
    Py_XDECREF(self->grass_texture);
}

static bool
base_occluded(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    if ((x != 0) && (y != 15) && (z != 15) &&
        !render_mode_hidden(state->rendermode, x - 1, y, z) &&
        !render_mode_hidden(state->rendermode, x, y, z + 1) &&
        !render_mode_hidden(state->rendermode, x, y + 1, z) &&
        !is_transparent(getArrayShort3D(state->blocks, x - 1, y, z)) &&
        !is_transparent(getArrayShort3D(state->blocks, x, y, z + 1)) &&
        !is_transparent(getArrayShort3D(state->blocks, x, y + 1, z))) {
        return true;
    }

    return false;
}

static void
base_draw(void* data, RenderState* state, PyObject* src, PyObject* mask, PyObject* mask_light) {
    PrimitiveBase* self = (PrimitiveBase*)data;

    /* in order to detect top parts of doublePlant grass & ferns */
    mc_block_t below_block = get_data(state, BLOCKS, state->x, state->y - 1, state->z);
    uint8_t below_data = get_data(state, DATA, state->x, state->y - 1, state->z);

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
        (state->block == block_grass && get_data(state, BLOCKS, state->x, state->y + 1, state->z) != 78) ||
        block_class_is_subset(state->block, (mc_block_t[]){block_vine, block_waterlily, block_flowing_water, block_water, block_leaves, block_leaves2},
                              6) ||
        /* tallgrass, but not dead shrubs */
        (state->block == block_tallgrass && state->block_data != 0) ||
        /* pumpkin/melon stem, not fully grown. Fully grown stems
         * get constant brown color (see textures.py) */
        (((state->block == block_pumpkin_stem) || (state->block == block_melon_stem)) && (state->block_data != 7)) ||
        /* doublePlant grass & ferns */
        (state->block == block_double_plant && (state->block_data == 2 || state->block_data == 3)) ||
        /* doublePlant grass & ferns tops */
        (state->block == block_double_plant && below_block == block_double_plant && (below_data == 2 || below_data == 3))) {
        /* do the biome stuff! */
        PyObject* facemask = mask;
        uint8_t r = 255, g = 255, b = 255;
        PyObject* color_table = NULL;
        bool flip_xy = false;

        if (state->block == block_grass) {
            /* grass needs a special facemask */
            facemask = self->grass_texture;
        }
        if (block_class_is_subset(state->block, (mc_block_t[]){block_grass, block_tallgrass, block_pumpkin_stem, block_melon_stem, block_vine, block_waterlily, block_double_plant}, 7)) {
            color_table = self->grasscolor;
        } else if (block_class_is_subset(state->block, (mc_block_t[]){block_flowing_water, block_water}, 2)) {
            color_table = self->watercolor;
        } else if (block_class_is_subset(state->block, (mc_block_t[]){block_leaves, block_leaves2}, 2)) {
            color_table = self->foliagecolor;
            /* birch foliage color is flipped XY-ways */
            flip_xy = state->block_data == 2;
        }

        if (color_table) {
            uint8_t biome;
            int32_t dx, dz;
            uint8_t tablex, tabley;
            float temp = 0.0, rain = 0.0;
            uint32_t multr = 0, multg = 0, multb = 0;
            int32_t tmp;
            PyObject* color = NULL;

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
            temp = OV_CLAMP(temp, 0.0, 1.0);
            rain = OV_CLAMP(rain, 0.0, 1.0);

            /* convert to x/y coordinates in color table */
            tablex = 255 - (255 * temp);
            tabley = 255 - (255 * rain);
            if (flip_xy) {
                uint8_t tmp = 255 - tablex;
                tablex = 255 - tabley;
                tabley = tmp;
            }

            /* look up color! */
            color = PySequence_GetItem(color_table, tabley * 256 + tablex);
            r = PyLong_AsLong(PyTuple_GET_ITEM(color, 0));
            g = PyLong_AsLong(PyTuple_GET_ITEM(color, 1));
            b = PyLong_AsLong(PyTuple_GET_ITEM(color, 2));
            Py_DECREF(color);

            /* do the after-coloration */
            r = OV_MULDIV255(r, multr, tmp);
            g = OV_MULDIV255(g, multg, tmp);
            b = OV_MULDIV255(b, multb, tmp);
        }

        /* final coloration */
        tint_with_mask(state->img, r, g, b, 255, facemask, state->imgx, state->imgy, 0, 0);
    }
}

RenderPrimitiveInterface primitive_base = {
    "base",
    sizeof(PrimitiveBase),
    base_start,
    base_finish,
    base_occluded,
    NULL,
    base_draw,
};
