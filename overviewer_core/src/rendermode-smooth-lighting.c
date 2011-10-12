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
    int x = state->imgx, y = state->imgy;
    
    if (is_transparent(state->block))
    {
        /* transparent blocks are rendered as usual, with flat lighting */
        rendermode_lighting.draw(data, state, src, mask, mask_light);
        return;
    }
    
    /* non-transparent blocks get the special smooth treatment */
    
    /* nothing special to do, but we do want to avoid vanilla
     * lighting mode draws */
    rendermode_normal.draw(data, state, src, mask, mask_light);
    
    /* draw a triangle on top of each block */
    draw_triangle(state->img,
                  x+12, y, 255, 0, 0,
                  x+24, y+6, 0, 255, 0,
                  x, y+6, 0, 0, 255);
    draw_triangle(state->img,
                  x+24, y+6, 255, 0, 0,
                  x, y+6, 0, 255, 0,
                  x+12, y+12, 0, 0, 255);
    
    /* left side... */
    draw_triangle(state->img,
                  x, y+6, 255, 0, 0,
                  x+12, y+12, 0, 255, 0,
                  x+12, y+24, 0, 0, 255);
    draw_triangle(state->img,
                  x+12, y+24, 255, 0, 0,
                  x, y+6, 0, 255, 0,
                  x, y+18, 0, 0, 255);

    /* right side... */
    draw_triangle(state->img,
                  x+24, y+6, 255, 0, 0,
                  x+12, y+12, 0, 255, 0,
                  x+12, y+24, 0, 0, 255);
    draw_triangle(state->img,
                  x+12, y+24, 255, 0, 0,
                  x+24, y+6, 0, 255, 0,
                  x+24, y+18, 0, 0, 255);
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
