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
#include <math.h>

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    long long seed; // needs to be at least 64-bits
} RenderPrimitiveSlime;

/*
 * random_* are a re-implementation of java's Random() class
 * since Minecraft's slime algorithm depends on it
 * http://docs.oracle.com/javase/1.4.2/docs/api/java/util/Random.html
 */

static void random_set_seed(long long *seed, long long new_seed) {
    *seed = (new_seed ^ 0x5deece66dLL) & ((1LL << 48) - 1);
}

static int random_next(long long *seed, int bits) {
    *seed = (*seed * 0x5deece66dLL + 0xbL) & ((1LL << 48) - 1);
    return (int)(*seed >> (48 - bits));
}

static int random_next_int(long long *seed, int n) {
    int bits, val;
    
    if (n <= 0) {
        /* invalid */
        return 0;
    }
    
    if ((n & -n) == n) {
        /* n is a power of two */
        return (int)((n * (long long)random_next(seed, 31)) >> 31);
    }
    
    do {
        bits = random_next(seed, 31);
        val = bits % n;
    } while (bits - val + (n - 1) < 0);
    return val;
}

static int is_slime(long long map_seed, int chunkx, int chunkz) {
    /* lots of magic numbers, but they're all correct! I swear! */
    long long seed;
    random_set_seed(&seed, (map_seed +
                            (long long)(chunkx * chunkx * 0x4c1906) +
                            (long long)(chunkx * 0x5ac0db) +
                            (long long)(chunkz * chunkz * 0x4307a7LL) +
                            (long long)(chunkz * 0x5f24f)) ^ 0x3ad8025f);
    return (random_next_int(&seed, 10) == 0);
}

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    RenderPrimitiveSlime *self = (RenderPrimitiveSlime *)data;
        
    /* set a nice, pretty green color */
    *r = self->parent.color->r;
    *g = self->parent.color->g;
    *b = self->parent.color->b;

    /* default to no overlay, until told otherwise */
    *a = 0;
    
    if (is_slime(self->seed, state->chunkx, state->chunkz)) {
        /* slimes can spawn! */
        *a = self->parent.color->a;
    }
}

static int
overlay_slime_start(void *data, RenderState *state, PyObject *support) {
    RenderPrimitiveSlime *self;
    PyObject *pyseed;

    /* first, chain up */
    int ret = primitive_overlay.start(data, state, support);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderPrimitiveSlime *)data;
    self->parent.get_color = get_color;
    
    self->parent.default_color.r = 40;
    self->parent.default_color.g = 230;
    self->parent.default_color.b = 40;
    self->parent.default_color.a = 240;
    
    pyseed = PyObject_GetAttrString(state->world, "seed");
    if (!pyseed)
        return 1;
    self->seed = PyLong_AsLongLong(pyseed);
    Py_DECREF(pyseed);
    if (PyErr_Occurred())
        return 1;
    
    return 0;
}

static void
overlay_slime_finish(void *data, RenderState *state) {
    /* chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_slime = {
    "overlay-slime",
    sizeof(RenderPrimitiveSlime),
    overlay_slime_start,
    overlay_slime_finish,
    NULL,
    NULL,
    overlay_draw,
};
