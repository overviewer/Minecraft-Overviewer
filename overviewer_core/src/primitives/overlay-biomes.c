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

#include "overlay.h"
#include "biomes.h"

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;

    void *biomes;
} RenderPrimitiveBiomes;

struct BiomeColor {
    unsigned char biome;
    unsigned char r, g, b;
};

static struct BiomeColor default_biomes[] = {
    {0, 135, 106, 150}, /* Ocean */
    {1, 98, 238, 240}, /* Plains */
    {2, 227, 107, 0}, /* Desert */
    {3, 255, 55, 55}, /* Extreme Hills */
    {4, 10, 200, 200}, /* Forest */
    {5, 10, 100, 240}, /* Taiga */
    {6, 200, 100, 100}, /* Swampland */
    {7, 70, 170, 0}, /* River */
    {8, 255, 0, 0}, /* Hell */
    {9, 255, 255, 255}, /* Sky */
    {10, 155, 55, 255}, /* FrozenOcean */
    {11, 255, 55, 255}, /* FrozenRiver */
    {12, 155, 255, 255}, /* Ice Plains */
    {13, 205, 205, 255}, /* Ice Mountains */
    {14, 255, 0, 155}, /* MushroomIsland */
    {15, 255, 75, 175}, /* MushroomIslandShore */
    {16, 255, 255, 0}, /* Beach */
    {17, 240, 155, 0}, /* DesertHills */
    {18, 100, 200, 200}, /* ForestHills */
    {19, 100, 100, 240}, /* TaigaHills */
    {20, 255, 25, 15}, /* Extreme Hills Edge */
    {21, 155, 155, 55}, /* Jungle */
    {22, 175, 255, 55}, /* Jungle Hills */
    {23, 135, 255, 55}, /* Jungle Edge */
    {24, 135, 106, 150}, /* Deep Ocean */
    {25, 255, 25, 15}, /* Stone Beach */
    {26, 155, 255, 255}, /* Cold Beach */
    {27, 10, 200, 200}, /* Birch Forest */
    {28, 10, 200, 200}, /* Birch Forest Edge */
    {29, 10, 200, 200}, /* Roofed Forest */
    {30, 155, 255, 255}, /* Cold Taiga */
    {31, 155, 200, 255}, /* Cold Taiga Hills */
    {32, 10, 100, 240}, /* Mega Taiga */
    {33, 10, 100, 240}, /* Mega Taiga Hills*/
    {34, 255, 55, 55}, /* Extreme Hills+ */
    {35, 227, 107, 0}, /* Savanna */
    {36, 227, 107, 0}, /* Savanna Plateau */
    {37, 255, 100, 100}, /* Mesa */
    {38, 255, 100, 100}, /* Mesa Plateau F */
    {39, 255, 100, 100}, /* Mesa Plateau */

    /* end of list marker */
    {255, 0, 0, 0}
};

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {

    unsigned char biome;
    int x = state->x, z = state->z, y_max, y;
    int max_i = -1;
    RenderPrimitiveBiomes* self = (RenderPrimitiveBiomes *)data;
    struct BiomeColor *biomes = (struct BiomeColor *)(self->biomes);
    *a = 0;

    y_max = state->y + 1;
    for (y = state->chunky * -16; y <= y_max; y++) {
        int i, tmp;
	biome = get_data(state, BIOMES, x, y, z);

        if (biome >= NUM_BIOMES) {
            biome = DEFAULT_BIOME;
        }

        for (i = 0; (max_i == -1 || i < max_i) && biomes[i].biome != 255; i++) {
            if (biomes[i].biome == biome) {
		//printf("(%d, %d, %d), %d, %s\n", x, y, z, biomes[i].biome, biome_table[biomes[i].biome].name);
                *r = biomes[i].r;
                *g = biomes[i].g;
                *b = biomes[i].b;

                *a = self->parent.color->a;

                max_i = i;
                break;
            }
        }
    }
}

static int
overlay_biomes_start(void *data, RenderState *state, PyObject *support) {
    PyObject *opt;
    RenderPrimitiveBiomes* self;
    unsigned char alpha_tmp=0;

    /* first, chain up */
    int ret = primitive_overlay.start(data, state, support);
    if (ret != 0)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveBiomes *)data;

    // opt is a borrowed reference.  do not deref
    if (!render_mode_parse_option(support, "biomes", "O", &(opt)))
        return 1;
    if (opt && opt != Py_None) {
        struct BiomeColor *biomes = NULL;
        Py_ssize_t biomes_size = 0, i;
        /* create custom biomes */

        if (!PyList_Check(opt)) {
            PyErr_SetString(PyExc_TypeError, "'biomes' must be a list");
            return 1;
        }

        biomes_size = PyList_GET_SIZE(opt);
        biomes = self->biomes = calloc(biomes_size + 1, sizeof(struct BiomeColor));
        if (biomes == NULL) {
            return 1;
        }

        for (i = 0; i < biomes_size; i++) {
            PyObject *biome = PyList_GET_ITEM(opt, i);
	    char *tmpname = NULL;
            int j = 0;

            if (!PyArg_ParseTuple(biome, "s(bbb)", &tmpname, &(biomes[i].r), &(biomes[i].g), &(biomes[i].b))) {
                free(biomes);
                self->biomes = NULL;
                return 1;
            }

            //printf("%s, (%d, %d, %d) ->", tmpname, biomes[i].r, biomes[i].g, biomes[i].b);
            for (j = 0; j < NUM_BIOMES; j++) {
                if (strncmp(biome_table[j].name, tmpname, strlen(tmpname))==0) {
                    //printf("biome_table index=%d", j);
                    biomes[i].biome = j;
                    break;
                }
            }
	    //printf("\n");
        }
	biomes[biomes_size].biome = 255; //Because 0 is a valid biome, have to use 255 as the end of list marker instead. Fragile!

    } else {
        self->biomes = default_biomes;
    }

    if (!render_mode_parse_option(support, "alpha", "b", &(alpha_tmp))) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
	    alpha_tmp = 240;
        }

	self->parent.color->a = alpha_tmp;
    }
    /* setup custom color */
    self->parent.get_color = get_color;

    return 0;
}

static void
overlay_biomes_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderPrimitiveBiomes* self = (RenderPrimitiveBiomes *)data;

    if (self->biomes && self->biomes != default_biomes) {
        free(self->biomes);
    }

    /* now, chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_biomes = {
    "overlay-biomes",
    sizeof(RenderPrimitiveBiomes),
    overlay_biomes_start,
    overlay_biomes_finish,
    NULL,
    NULL,
    overlay_draw,
};
