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

/* figures out the black_coeff from a given skylight and blocklight, used in
   lighting calculations -- note this is *different* from the one in
   rendermode-lighting.c (the "skylight - 11" part) */
static float calculate_darkness(unsigned char skylight, unsigned char blocklight) {
    return 1.0f - powf(0.8f, 15.0 - MAX(blocklight, skylight - 11));
}

static int
rendermode_night_start(void *data, RenderState *state, PyObject *options) {
    RenderModeNight* self;    

    /* first, chain up */
    int ret = rendermode_lighting.start(data, state, options);
    if (ret != 0)
        return ret;
    
    /* override the darkness function with our night version! */
    self = (RenderModeNight *)data;    
    self->parent.calculate_darkness = calculate_darkness;
    
    return 0;
}

static void
rendermode_night_finish(void *data, RenderState *state) {
    /* nothing special to do */
    rendermode_lighting.finish(data, state);
}

static int
rendermode_night_occluded(void *data, RenderState *state, int x, int y, int z) {
    /* no special occlusion here */
    return rendermode_lighting.occluded(data, state, x, y, z);
}

static int
rendermode_night_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* no special hiding here */
    return rendermode_lighting.hidden(data, state, x, y, z);
}

static void
rendermode_night_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    /* nothing special to do */
    rendermode_lighting.draw(data, state, src, mask, mask_light);
}

RenderModeInterface rendermode_night = {
    "night", "Night",
    "like \"lighting\", except at night",
    NULL,
    &rendermode_lighting,
    sizeof(RenderModeNight),
    rendermode_night_start,
    rendermode_night_finish,
    rendermode_night_occluded,
    rendermode_night_hidden,
    rendermode_night_draw,
};
