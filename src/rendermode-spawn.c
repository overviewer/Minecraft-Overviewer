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

static int
rendermode_spawn_start(void *data, RenderState *state) {
    RenderModeSpawn* self;

    /* first, chain up */
    int ret = rendermode_night.start(data, state);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderModeSpawn *)data;
    self->solid_blocks = PyObject_GetAttrString(state->chunk, "solid_blocks");
    self->nospawn_blocks = PyObject_GetAttrString(state->chunk, "nospawn_blocks");
    self->fluid_blocks = PyObject_GetAttrString(state->chunk, "fluid_blocks");
    self->red_color = PyObject_GetAttrString(state->chunk, "red_color");
    
    return 0;
}

static void
rendermode_spawn_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderModeSpawn* self = (RenderModeSpawn *)data;    
    
    Py_DECREF(self->solid_blocks);
    Py_DECREF(self->nospawn_blocks);
    Py_DECREF(self->fluid_blocks);
    
    /* now, chain up */
    rendermode_night.finish(data, state);
}

static int
rendermode_spawn_occluded(void *data, RenderState *state) {
    /* no special occlusion here */
    return rendermode_night.occluded(data, state);
}

static void
rendermode_spawn_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    /* different versions of self (spawn, lighting) */
    RenderModeSpawn* self = (RenderModeSpawn *)data;
    RenderModeLighting *lighting = (RenderModeLighting *)self;
    
    int x = state->x, y = state->y, z = state->z;
    PyObject *old_black_color = NULL;
    
    /* figure out the appropriate darkness:
       this block for transparents, the block above for non-transparent */
    float darkness = 0.0;
    if (is_transparent(state->block)) {
        darkness = get_lighting_coefficient((RenderModeLighting *)self, state, x, y, z, NULL);
    } else {
        darkness = get_lighting_coefficient((RenderModeLighting *)self, state, x, y, z+1, NULL);
    }
    
    /* if it's dark enough... */
    if (darkness > 0.8) {
        PyObject *block_py = PyInt_FromLong(state->block);
        
        /* make sure it's solid */
        if (PySequence_Contains(self->solid_blocks, block_py)) {
            int spawnable = 1;
            
            /* not spawnable if its in the nospawn list */
            if (PySequence_Contains(self->nospawn_blocks, block_py))
                spawnable = 0;
            
            /* check the block above for solid or fluid */
            if (spawnable && z != 127) {
                PyObject *top_block_py = PyInt_FromLong(getArrayByte3D(state->blocks, x, y, z+1));
                if (PySequence_Contains(self->solid_blocks, top_block_py) ||
                    PySequence_Contains(self->fluid_blocks, top_block_py)) {
                    
                    spawnable = 0;
                }
                
                Py_DECREF(top_block_py);
            }
            
            /* if we passed all the checks, replace black_color with red_color */
            if (spawnable) {
                old_black_color = lighting->black_color;
                lighting->black_color = self->red_color;
            }
        }
        
        Py_DECREF(block_py);
    }
    
    /* draw normally */
    rendermode_night.draw(data, state, src, mask);
    
    /* reset black_color, if needed */
    if (old_black_color != NULL) {
        lighting->black_color = old_black_color;
    }
}

RenderModeInterface rendermode_spawn = {
    "spawn", "draws red where monsters can spawn at night",
    sizeof(RenderModeSpawn),
    rendermode_spawn_start,
    rendermode_spawn_finish,
    rendermode_spawn_occluded,
    rendermode_spawn_draw,
};
