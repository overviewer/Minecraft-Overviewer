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

struct OreColor {
    unsigned char blockid;
    unsigned char r, g, b;
};

/* put more valuable ores first -- they take precedence */
static struct OreColor orecolors[] = {
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
    
    int x = state->x, y = state->y, z_max = state->z, z;
    //RenderModeMineral* self = (RenderModeMineral *)data;
    *a = 0;
    
    for (z = 0; z <= z_max; z++) {
        int i, tmp, max_i = sizeof(orecolors) / sizeof(struct OreColor);
        unsigned char blockid = getArrayByte3D(state->blocks, x, y, z);
        
        for (i = 0; i < max_i && orecolors[i].blockid != 0; i++) {
            if (orecolors[i].blockid == blockid) {
                *r = orecolors[i].r;
                *g = orecolors[i].g;
                *b = orecolors[i].b;
                
                tmp = (128 - z_max + z) * 2 - 40;
                *a = MIN(MAX(0, tmp), 255);
                
                max_i = i;
                break;
            }
        }
    }
}

static int
rendermode_mineral_start(void *data, RenderState *state) {
    RenderModeMineral* self;

    /* first, chain up */
    int ret = rendermode_overlay.start(data, state);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderModeMineral *)data;
    
    /* setup custom color */
    self->parent.get_color = get_color;
    
    return 0;
}

static void
rendermode_mineral_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    //RenderModeMineral* self = (RenderModeMineral *)data;
    
    /* now, chain up */
    rendermode_overlay.finish(data, state);
}

static int
rendermode_mineral_occluded(void *data, RenderState *state) {
    /* no special occlusion here */
    return rendermode_overlay.occluded(data, state);
}

static void
rendermode_mineral_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    /* draw normally */
    rendermode_overlay.draw(data, state, src, mask);
}

RenderModeInterface rendermode_mineral = {
    "mineral", "draws a colored overlay showing where ores are located",
    &rendermode_overlay,
    sizeof(RenderModeMineral),
    rendermode_mineral_start,
    rendermode_mineral_finish,
    rendermode_mineral_occluded,
    rendermode_mineral_draw,
};
