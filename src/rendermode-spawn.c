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
#include <math.h>

static void get_color(void *data, RenderState *state,
                      unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *a) {
    
    RenderModeSpawn* self = (RenderModeSpawn *)data;
    int x = state->x, y = state->y, z = state->z;
    int z_light = z + 1;
    unsigned char blocklight, skylight;
    PyObject *block_py;
    
    /* set a nice, pretty red color */
    *r = 229;
    *g = 36;
    *b = 38;
    
    /* default to no overlay, until told otherwise */
    *a = 0;
    
    block_py = PyInt_FromLong(state->block);
    if (PySequence_Contains(self->nospawn_blocks, block_py)) {
        /* nothing can spawn on this */
        Py_DECREF(block_py);
        return;
    }
    Py_DECREF(block_py);
    
    blocklight = getArrayByte3D(self->blocklight, x, y, MIN(127, z_light));
    
    /* if we're at the top, force 15 (brightest!) skylight */
    if (z_light == 128) {
        skylight = 15;
    } else {
        skylight = getArrayByte3D(self->skylight, x, y, z_light);
    }
    
    if (MAX(blocklight, skylight) <= 7) {
        /* hostile mobs spawn in daylight */
        *a = 240;
    } else if (MAX(blocklight, skylight - 11) <= 7) {
        /* hostile mobs spawn at night */
        *a = 150;
    }
}

static int
rendermode_spawn_start(void *data, RenderState *state) {
    RenderModeSpawn* self;

    /* first, chain up */
    int ret = rendermode_overlay.start(data, state);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderModeSpawn *)data;
    self->nospawn_blocks = PyObject_GetAttrString(state->chunk, "nospawn_blocks");
    self->blocklight = PyObject_GetAttrString(state->self, "blocklight");
    self->skylight = PyObject_GetAttrString(state->self, "skylight");
    
    /* setup custom color */
    self->parent.get_color = get_color;
    
    return 0;
}

static void
rendermode_spawn_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderModeSpawn* self = (RenderModeSpawn *)data;
    
    Py_DECREF(self->nospawn_blocks);
    Py_DECREF(self->blocklight);
    Py_DECREF(self->skylight);
    
    /* now, chain up */
    rendermode_overlay.finish(data, state);
}

static int
rendermode_spawn_occluded(void *data, RenderState *state) {
    /* no special occlusion here */
    return rendermode_overlay.occluded(data, state);
}

static void
rendermode_spawn_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    /* draw normally */
    rendermode_overlay.draw(data, state, src, mask, mask_light);
}

RenderModeInterface rendermode_spawn = {
    "spawn", "draws a red overlay where monsters can spawn at night",
    &rendermode_overlay,
    sizeof(RenderModeSpawn),
    rendermode_spawn_start,
    rendermode_spawn_finish,
    rendermode_spawn_occluded,
    rendermode_spawn_draw,
};
