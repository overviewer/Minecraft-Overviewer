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
    /* 0 */
    {0, 135, 206, 250}, /* Ocean */
    {1, 198, 238, 140}, /* Plains */
    {2, 227, 207, 87}, /* Desert */
    {3, 155, 155, 155}, /* Extreme Hills */
    {4, 184, 239, 134}, /* Forest */
    /* 5 */
    {5, 148, 228, 220}, /* Taiga */
    {6, 0, 100, 90}, /* Swampland */
    {7, 70, 130, 180}, /* River */
    {8, 176, 23, 31}, /* Hell */
    {9, 255, 255, 255}, /* Sky */
    /* 10 */
    {10, 155, 55, 255}, /* FrozenOcean */
    {11, 255, 55, 255}, /* FrozenRiver */
    {12, 155, 255, 255}, /* Ice Plains */
    {13, 205, 205, 255}, /* Ice Mountains */
    {14, 55, 55, 55}, /* MushroomIsland */
    /* 15 */
    {15, 75, 75, 75}, /* MushroomIslandShore */
    {16, 255, 215, 80}, /* Beach */
    {17, 140, 155, 255}, /* DesertHills */
    {18, 105, 255, 155}, /* ForestHills */
    {19, 75, 125, 75}, /* TaigaHills */
    /* 20 */
    {20, 255, 25, 15}, /* Extreme Hills Edge */
    {21, 0, 255, 55}, /* Jungle */
    {22, 75, 255, 255}, /* Jungle Mountains */
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
                
                tmp = (128 - y_max + y) * 2 - 40;
                *a = MIN(MAX(0, tmp), 255);
                
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
