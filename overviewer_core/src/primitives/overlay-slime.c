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

#include <math.h>
#include "overlay.h"

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    int64_t seed; // needs to be at least 64-bits
} RenderPrimitiveSlime;

/*
 * random_* are a re-implementation of java's Random() class
 * since Minecraft's slime algorithm depends on it
 * http://docs.oracle.com/javase/1.4.2/docs/api/java/util/Random.html
 */

static void random_set_seed(int64_t* seed, int64_t new_seed) {
    *seed = (new_seed ^ 0x5deece66dLL) & ((1LL << 48) - 1);
}

static int64_t random_next(int64_t* seed, int32_t bits) {
    *seed = (*seed * 0x5deece66dLL + 0xbL) & ((1LL << 48) - 1);
    return (int64_t)(*seed >> (48 - bits));
}

static int64_t random_next_int(int64_t* seed, uint32_t modulo) {
    int64_t bits, val;

    if ((modulo & -modulo) == modulo) {
        /* modulo is a power of two */
        return (int64_t)((modulo * (int64_t)random_next(seed, 31)) >> 31);
    }

    do {
        bits = random_next(seed, 31);
        val = bits % modulo;
    } while (bits - val + (modulo - 1) < 0);
    return val;
}

static bool is_slime(int64_t map_seed, int32_t chunkx, int32_t chunkz) {
    /* lots of magic numbers, but they're all correct! I swear! */
    int64_t seed;
    random_set_seed(&seed, (map_seed +
                            (int64_t)(chunkx * chunkx * 0x4c1906) +
                            (int64_t)(chunkx * 0x5ac0db) +
                            (int64_t)(chunkz * chunkz * 0x4307a7LL) +
                            (int64_t)(chunkz * 0x5f24f)) ^
                               0x3ad8025f);
    return (random_next_int(&seed, 10) == 0);
}

static void get_color(void* data, RenderState* state,
                      uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) {
    RenderPrimitiveSlime* self = (RenderPrimitiveSlime*)data;

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

static bool
overlay_slime_start(void* data, RenderState* state, PyObject* support) {
    RenderPrimitiveSlime* self;
    PyObject* pyseed;

    /* first, chain up */
    bool ret = primitive_overlay.start(data, state, support);
    if (ret != false)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveSlime*)data;
    self->parent.get_color = get_color;

    self->parent.default_color.r = 40;
    self->parent.default_color.g = 230;
    self->parent.default_color.b = 40;
    self->parent.default_color.a = 240;

    pyseed = PyObject_GetAttrString(state->world, "seed");
    if (!pyseed)
        return true;
    self->seed = PyLong_AsLongLong(pyseed);
    Py_DECREF(pyseed);
    if (PyErr_Occurred())
        return true;

    return false;
}

static void
overlay_slime_finish(void* data, RenderState* state) {
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
