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

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    
    void *minerals;
} RenderPrimitiveMineral;

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
    
    int x = state->x, z = state->z, y_max, y;
    int max_i = -1;
    RenderPrimitiveMineral* self = (RenderPrimitiveMineral *)data;
    struct MineralColor *minerals = (struct MineralColor *)(self->minerals);
    *a = 0;
    
    y_max = state->y + 1;
    for (y = state->chunky * -16; y <= y_max; y++) {
        int i, tmp;
        unsigned short blockid = get_data(state, BLOCKS, x, y, z);
        
        for (i = 0; (max_i == -1 || i < max_i) && minerals[i].blockid != 0; i++) {
            if (minerals[i].blockid == blockid) {
                *r = minerals[i].r;
                *g = minerals[i].g;
                *b = minerals[i].b;
                
                tmp = (128 - y_max + y) * 2 - 40;
                *a = MIN(MAX(0, tmp), 255);
                
                max_i = i;
                break;
            }
        }
    }
}

static int
overlay_mineral_start(void *data, RenderState *state, PyObject *support) {
    PyObject *opt;
    RenderPrimitiveMineral* self;

    /* first, chain up */
    int ret = primitive_overlay.start(data, state, support);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderPrimitiveMineral *)data;
    
    // opt is a borrowed reference.  do not deref 
    if (!render_mode_parse_option(support, "minerals", "O", &(opt)))
        return 1;
    if (opt && opt != Py_None) {
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
overlay_mineral_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderPrimitiveMineral* self = (RenderPrimitiveMineral *)data;
    
    if (self->minerals && self->minerals != default_minerals) {
        free(self->minerals);
    }
    
    /* now, chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_mineral = {
    "overlay-mineral",
    sizeof(RenderPrimitiveMineral),
    overlay_mineral_start,
    overlay_mineral_finish,
    NULL,
    NULL,
    overlay_draw,
};
