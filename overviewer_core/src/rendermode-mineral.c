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

struct MineralColor {
    unsigned char blockid;
    unsigned char r, g, b;
};

/* put more valuable ores first -- they take precedence */
static struct MineralColor default_minerals[] = {
    {48 /* Mossy Stone  */, 31, 153, 9},
    
    {56 /* Diamond Ore  */, 32, 230, 220},

    {21 /* Lapis Lazuli */, 0, 23, 176},
    {14 /* Gold Ore     */, 255, 234, 0},

    {15 /* Iron Ore     */, 204, 204, 204},
    {73 /* Redstone     */, 186, 0, 0},
    {74 /* Lit Redstone */, 186, 0, 0},
    {16 /* Coal Ore     */, 54, 54, 54},
    
    /* end of list marker */
    {0, 0, 0, 0}
};

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    
    int x = state->x, y = state->y, z_max = state->z + 1, z;
    int max_i = -1;
    RenderModeMineral* self = (RenderModeMineral *)data;
    struct MineralColor *minerals = (struct MineralColor *)(self->minerals);
    *a = 0;
    
    for (z = 0; z <= z_max; z++) {
        int i, tmp;
        unsigned char blockid = getArrayByte3D(state->blocks, x, y, z);
        
        for (i = 0; (max_i == -1 || i < max_i) && minerals[i].blockid != 0; i++) {
            if (minerals[i].blockid == blockid) {
                *r = minerals[i].r;
                *g = minerals[i].g;
                *b = minerals[i].b;
                
                tmp = (128 - z_max + z) * 2 - 40;
                *a = MIN(MAX(0, tmp), 255);
                
                max_i = i;
                break;
            }
        }
    }
}

static int
rendermode_mineral_start(void *data, RenderState *state, PyObject *options) {
    PyObject *opt;
    RenderModeMineral* self;

    /* first, chain up */
    int ret = rendermode_overlay.start(data, state, options);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderModeMineral *)data;
    
    opt = PyDict_GetItemString(options, "minerals");
    if (opt) {
        struct MineralColor *minerals = NULL;
        Py_ssize_t minerals_size = 0, i;
        /* create custom minerals */
        
        if (!PyList_Check(opt)) {
            PyErr_SetString(PyExc_TypeError, "'minerals' must be a list");
            return 1;
        }
        
        minerals_size = PyList_GET_SIZE(opt);
        minerals = self->minerals = calloc(minerals_size + 1, sizeof(struct MineralColor));
        if (minerals == NULL) {
            return 1;
        }
        
        for (i = 0; i < minerals_size; i++) {
            PyObject *mineral = PyList_GET_ITEM(opt, i);
            if (!PyArg_ParseTuple(mineral, "b(bbb)", &(minerals[i].blockid), &(minerals[i].r), &(minerals[i].g), &(minerals[i].b))) {
                free(minerals);
                self->minerals = NULL;
                return 1;
            }
        }
    } else {
        self->minerals = default_minerals;
    }
    
    /* setup custom color */
    self->parent.get_color = get_color;
    
    return 0;
}

static void
rendermode_mineral_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderModeMineral* self = (RenderModeMineral *)data;
    
    if (self->minerals && self->minerals != default_minerals) {
        free(self->minerals);
    }
    
    /* now, chain up */
    rendermode_overlay.finish(data, state);
}

static int
rendermode_mineral_occluded(void *data, RenderState *state, int x, int y, int z) {
    /* no special occlusion here */
    return rendermode_overlay.occluded(data, state, x, y, z);
}

static int
rendermode_mineral_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* no special hiding here */
    return rendermode_overlay.hidden(data, state, x, y, z);
}

static void
rendermode_mineral_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    /* draw normally */
    rendermode_overlay.draw(data, state, src, mask, mask_light);
}

const RenderModeOption rendermode_mineral_options[] = {
    {"minerals", "a list of (blockid, (r, g, b)) tuples for coloring minerals"},
    {NULL, NULL}
};

RenderModeInterface rendermode_mineral = {
    "mineral", "Mineral",
    "draws a colored overlay showing where ores are located",
    rendermode_mineral_options,
    &rendermode_overlay,
    sizeof(RenderModeMineral),
    rendermode_mineral_start,
    rendermode_mineral_finish,
    rendermode_mineral_occluded,
    rendermode_mineral_hidden,
    rendermode_mineral_draw,
};
