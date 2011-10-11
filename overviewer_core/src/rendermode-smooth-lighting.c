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
rendermode_smooth_lighting_start(void *data, RenderState *state, PyObject *options) {
    RenderModeNight* self;    

    /* first, chain up */
    int ret = rendermode_lighting.start(data, state, options);
    if (ret != 0)
        return ret;
    
    return 0;
}

static void
rendermode_smooth_lighting_finish(void *data, RenderState *state) {
    /* nothing special to do */
    rendermode_lighting.finish(data, state);
}

static int
rendermode_smooth_lighting_occluded(void *data, RenderState *state, int x, int y, int z) {
    /* no special occlusion here */
    return rendermode_lighting.occluded(data, state, x, y, z);
}

static int
rendermode_smooth_lighting_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* no special hiding here */
    return rendermode_lighting.hidden(data, state, x, y, z);
}

static void
rendermode_smooth_lighting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    /* nothing special to do */
    rendermode_lighting.draw(data, state, src, mask, mask_light);
}

RenderModeInterface rendermode_smooth_lighting = {
    "smooth-lighting", "Smooth Lighting",
    "like \"lighting\", except smooth",
    NULL,
    &rendermode_lighting,
    sizeof(RenderModeSmoothLighting),
    rendermode_smooth_lighting_start,
    rendermode_smooth_lighting_finish,
    rendermode_smooth_lighting_occluded,
    rendermode_smooth_lighting_hidden,
    rendermode_smooth_lighting_draw,
};
