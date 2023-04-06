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

#include "biomes.h"
#include "overlay.h"

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;

    void* biomes;
} RenderPrimitiveBiomes;

struct BiomeColor {
    uint8_t biome;
    uint8_t r, g, b;
};

/* This default list matches (or at least tries to match)
 * the default coloring scheme for Biomes established by
 * the contributors of the Amidst project:
 * https://github.com/toolbox4minecraft/amidst/wiki/Biome-Color-Table
 */
static struct BiomeColor default_biomes[] = {
    {  0,   0,   0, 112},  /* Ocean */
    {  1, 141, 179,  96},  /* Plains */
    {  2, 250, 148,  24},  /* Desert */
    {  3,  96,  96,  96},  /* Mountains */
    {  4,   5, 102,  21},  /* Forest */
    {  5,  11, 102,  33},  /* Taiga */
    {  6,   7, 249, 178},  /* Swamp */
    {  7,   0,   0, 255},  /* River */
    {  8, 190,  60,  60},  /* Nether Wastes; not clear red because we use that for crimson forest instead */
    {  9, 128, 128, 255},  /* The End */
    { 10, 112, 112, 214},  /* Frozen Ocean */
    { 11, 160, 160, 255},  /* Frozen River */
    { 12, 255, 255, 255},  /* Snowy Tundra */
    { 13, 160, 160, 160},  /* Snowy Mountains */
    { 14, 255,   0, 255},  /* Mushroom Fields */
    { 15, 160,   0, 255},  /* Mushroom Field Shore */
    { 16, 250, 222,  85},  /* Beach */
    { 17, 210,  95,  18},  /* Desert Hills */
    { 18,  34,  85,  28},  /* Wooded Hills */
    { 19,  22,  57,  51},  /* Taiga Hills */
    { 20, 114, 120, 154},  /* Mountain Edge */
    { 21,  83, 123,   9},  /* Jungle */
    { 22,  44,  66,   5},  /* Jungle Hills */
    { 23,  98, 139,  23},  /* Jungle Edge */
    { 24,   0,   0,  48},  /* Deep Ocean */
    { 25, 162, 162, 132},  /* Stone Shore */
    { 26, 250, 240, 192},  /* Snowy Beach */
    { 27,  48, 116,  68},  /* Birch Forest */
    { 28,  31,  95,  50},  /* Birch Forest Hills */
    { 29,  64,  81,  26},  /* Dark Forest */
    { 30,  49,  85,  74},  /* Snowy Taiga */
    { 31,  36,  63,  54},  /* Snowy Taiga Hills */
    { 32,  89, 102,  81},  /* Giant Tree Taiga */
    { 33,  69,  79,  62},  /* Giant Tree Taiga Hills */
    { 34,  80, 112,  80},  /* Wooded Mountains */
    { 35, 189, 178,  95},  /* Savanna */
    { 36, 167, 157, 100},  /* Savanna Plateau */
    { 37, 217,  69,  21},  /* Badlands */
    { 38, 176, 151, 101},  /* Wooded Badlands Plateau */
    { 39, 202, 140, 101},  /* Badlands Plateau */
    { 40, 128, 128, 255},  /* Small End Islands */
    { 41, 128, 128, 255},  /* End Midlands */
    { 42, 128, 128, 255},  /* End Highlands */
    { 43, 128, 128, 255},  /* End Barrens */
    { 44,   0,   0, 172},  /* Warm Ocean */
    { 45,   0,   0, 144},  /* Lukewarm Ocean */
    { 46,  32,  32, 112},  /* Cold Ocean */
    { 47,   0,   0,  80},  /* Deep Warm Ocean */
    { 48,   0,   0,  64},  /* Deep Lukewarm Ocean */
    { 49,  32,  32,  56},  /* Deep Cold Ocean */
    { 50,  64,  64, 144},  /* Deep Frozen Ocean */
    /* 127 would be "the void" */
    {129, 181, 219, 136},  /* Sunflower Plains */
    {130, 255, 188,  64},  /* Desert Lakes */
    {131, 136, 136, 136},  /* Gravelly Mountains */
    {132,  45, 142,  73},  /* Flower Forest */
    {133,  51, 142, 129},  /* Taiga Mountains */
    {134,  47, 255, 218},  /* Swamp Hills */
    {140,   0,  70, 120},  /* Ice Spikes */
    {149, 123, 163,  49},  /* Modified Jungle */
    {151, 138, 179,  63},  /* Modified Jungle Edge */
    {155,  88, 156, 108},  /* Tall Birch Forest */
    {156,  71, 135,  90},  /* Tall Birch Hills */
    {157, 104, 121,  66},  /* Dark Forest Hills */
    {158,  89, 125, 114},  /* Snowy Taiga Mountains */
    {160, 129, 142, 121},  /* Giant Spruce Taiga */
    {161, 109, 119, 102},  /* Giant Spruce Taiga Hills */
    {162, 120, 152, 120},  /* Gravelly Mountains+ */
    {163, 229, 218, 135},  /* Shattered Savanna */
    {164, 207, 197, 140},  /* Shattered Savanna Plateau */
    {165, 255, 109,  61},  /* Eroded Badlands */
    {166, 216, 191, 141},  /* Modified Wooded Badlands Plateau */
    {167, 242, 180, 141},  /* Modified Badlands Plateau */
    {168, 118, 142,  20},  /* Bamboo Jungle */
    {169,  59,  71,  10},  /* Bamboo Jungle Hills */
    {170,  70,  50,  40},  /* Soul Sand Valley */
    {171, 255,   0,   0},  /* Crimson Forest */
    {172,  30, 160, 160},  /* Warped Forest */
    {173,  70,  70,  70},  /* Basalt Deltas */

    /* end of list marker */
    {255, 0, 0, 0}};

static void get_color(void* data, RenderState* state,
                      uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) {

    uint8_t biome;
    int32_t x = state->x, z = state->z, y_max, y;
    int32_t max_i = -1;
    RenderPrimitiveBiomes* self = (RenderPrimitiveBiomes*)data;
    struct BiomeColor* biomes = (struct BiomeColor*)(self->biomes);
    *a = 0;

    y_max = state->y + 1;
    for (y = state->chunky * -16; y <= y_max; y++) {
        int32_t i, tmp;
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

static bool
overlay_biomes_start(void* data, RenderState* state, PyObject* support) {
    PyObject* opt;
    RenderPrimitiveBiomes* self;
    uint8_t alpha_tmp = 0;

    /* first, chain up */
    bool ret = primitive_overlay.start(data, state, support);
    if (ret != false)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveBiomes*)data;

    // opt is a borrowed reference.  do not deref
    if (!render_mode_parse_option(support, "biomes", "O", &(opt)))
        return true;
    if (opt && opt != Py_None) {
        struct BiomeColor* biomes = NULL;
        Py_ssize_t biomes_size = 0, i;
        /* create custom biomes */

        if (!PyList_Check(opt)) {
            PyErr_SetString(PyExc_TypeError, "'biomes' must be a list");
            return true;
        }

        biomes_size = PyList_GET_SIZE(opt);
        biomes = self->biomes = calloc(biomes_size + 1, sizeof(struct BiomeColor));
        if (biomes == NULL) {
            return true;
        }

        for (i = 0; i < biomes_size; i++) {
            PyObject* biome = PyList_GET_ITEM(opt, i);
            char* tmpname = NULL;
            uint32_t j = 0;

            if (!PyArg_ParseTuple(biome, "s(bbb)", &tmpname, &(biomes[i].r), &(biomes[i].g), &(biomes[i].b))) {
                free(biomes);
                self->biomes = NULL;
                return true;
            }

            //printf("%s, (%d, %d, %d) ->", tmpname, biomes[i].r, biomes[i].g, biomes[i].b);
            for (j = 0; j < NUM_BIOMES; j++) {
                if (strncmp(biome_table[j].name, tmpname, strlen(tmpname)) == 0) {
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

    return false;
}

static void
overlay_biomes_finish(void* data, RenderState* state) {
    /* first free all *our* stuff */
    RenderPrimitiveBiomes* self = (RenderPrimitiveBiomes*)data;

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
