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

#include "../overviewer.h"
#include "lighting.h"
#include <math.h>

typedef struct {
    /* inherits from lighting */
    RenderPrimitiveLighting parent;
} RenderPrimitiveSmoothLighting;

/* structure representing one corner of a face (see below) */
struct SmoothLightingCorner {
    /* where this corner shows up on each block texture */
    int imgx, imgy;
    
    /* the two block offsets that (together) determine the 4 blocks to use */
    int dx1, dy1, dz1;
    int dx2, dy2, dz2;
};

/* structure for rule table handling lighting */
struct SmoothLightingFace {
    /* offset from current coordinate to the block this face points towards
       used for occlusion calculations, and as a base for later */
    int dx, dy, dz;
    
    /* the points that form the corners of this face */
    struct SmoothLightingCorner corners[4];
    
    /* pairs of (x,y) in order, as touch-up points, or NULL for none */
    int *touch_up_points;
    unsigned int num_touch_up_points;
};

/* top face touchups, pulled from textures.py (_build_block) */
static int top_touchups[] = {3, 4, 7, 2, 11, 0};

/* the lighting face rule list! */
static struct SmoothLightingFace lighting_rules[] = {
    /* since this is getting a little insane, here's the general layout:
       
    {dx, dy, dz, {        // direction this face is towards
                          // now, a list of 4 corners...
            {imgx, imgy,  // where the corner is on the block image
             x1, y1, z1,  // two vectors, describing the 4 (!!!)
             x2, y2, z2}, // blocks neighboring this corner
            // ...
        },
     {x, y, x, y}, 2}, // touch-up points, and how many there are (may be NULL)
     
    // ...
    
    */
    
    /* top */
    {0, 1, 0, {
            {0, 6,
             -1, 0, 0,
             0, 0, -1},
            {12, 0,
             1, 0, 0,
             0, 0, -1},
            {24, 6,
             1, 0, 0,
             0, 0, 1},
            {12, 12,
             -1, 0, 0,
             0, 0, 1},
        },
     top_touchups, 3},
    
    /* left */
    {-1, 0, 0, {
            {0, 18,
             0, 0, -1,
             0, -1, 0},
            {0, 6,
             0, 0, -1,
             0, 1, 0},
            {12, 12,
             0, 0, 1,
             0, 1, 0},
            {12, 24,
             0, 0, 1,
             0, -1, 0},
        },
     NULL, 0},
    
    /* right */
    {0, 0, 1, {
            {24, 6,
             1, 0, 0,
             0, 1, 0},
            {12, 12,
             -1, 0, 0,
             0, 1, 0},
            {12, 24,
             -1, 0, 0,
             0, -1, 0},
            {24, 18,
             1, 0, 0,
             0, -1, 0},
        },
     NULL, 0},
};

/* helpers for indexing the rule list */
enum
{
    FACE_TOP = 0,
    FACE_LEFT = 1,
    FACE_RIGHT = 2,
};

static void
do_shading_with_rule(RenderPrimitiveSmoothLighting *self, RenderState *state, struct SmoothLightingFace face) {
    int i;
    RenderPrimitiveLighting *lighting = (RenderPrimitiveLighting *)self;
    int x = state->imgx, y = state->imgy;
    struct SmoothLightingCorner *pts = face.corners;
    float comp_shade_strength = 1.0 - lighting->strength;
    unsigned char pts_r[4] = {0, 0, 0, 0};
    unsigned char pts_g[4] = {0, 0, 0, 0};
    unsigned char pts_b[4] = {0, 0, 0, 0};
    int cx = state->x + face.dx;
    int cy = state->y + face.dy;
    int cz = state->z + face.dz;
    
    /* first, check for occlusion if the block is in the local chunk */
    if (lighting_is_face_occluded(state, 0, cx, cy, cz))
        return;
    
    /* calculate the lighting colors for each point */
    for (i = 0; i < 4; i++)
    {
        unsigned char r, g, b;
        unsigned int rgather = 0, ggather = 0, bgather = 0;
        
        get_lighting_color(lighting, state, cx, cy, cz,
                           &r, &g, &b);
        rgather += r; ggather += g; bgather += b;

        get_lighting_color(lighting, state,
                           cx+pts[i].dx1, cy+pts[i].dy1, cz+pts[i].dz1,
                           &r, &g, &b);
        rgather += r; ggather += g; bgather += b;

        get_lighting_color(lighting, state,
                           cx+pts[i].dx2, cy+pts[i].dy2, cz+pts[i].dz2,
                           &r, &g, &b);
        rgather += r; ggather += g; bgather += b;

        /* FIXME special far corner handling */
        get_lighting_color(lighting, state,
                           cx+pts[i].dx1+pts[i].dx2, cy+pts[i].dy1+pts[i].dy2, cz+pts[i].dz1+pts[i].dz2,
                           &r, &g, &b);
        rgather += r; ggather += g; bgather += b;
        
        rgather += (255*4 - rgather) * comp_shade_strength;
        ggather += (255*4 - ggather) * comp_shade_strength;
        bgather += (255*4 - bgather) * comp_shade_strength;
    
        pts_r[i] = rgather / 4;
        pts_g[i] = ggather / 4;
        pts_b[i] = bgather / 4;
    }
    
    /* draw the face */
    draw_triangle(state->img, 1,
                  x+pts[0].imgx, y+pts[0].imgy, pts_r[0], pts_g[0], pts_b[0],
                  x+pts[1].imgx, y+pts[1].imgy, pts_r[1], pts_g[1], pts_b[1],
                  x+pts[2].imgx, y+pts[2].imgy, pts_r[2], pts_g[2], pts_b[2],
                  x, y, face.touch_up_points, face.num_touch_up_points);
    draw_triangle(state->img, 0,
                  x+pts[0].imgx, y+pts[0].imgy, pts_r[0], pts_g[0], pts_b[0],
                  x+pts[2].imgx, y+pts[2].imgy, pts_r[2], pts_g[2], pts_b[2],
                  x+pts[3].imgx, y+pts[3].imgy, pts_r[3], pts_g[3], pts_b[3],
                  x, y, NULL, 0);
}

static int
smooth_lighting_start(void *data, RenderState *state, PyObject *support) {
    /* first, chain up */
    int ret = primitive_lighting.start(data, state, support);
    if (ret != 0)
        return ret;    
    return 0;
}

static void
smooth_lighting_finish(void *data, RenderState *state) {
    /* nothing special to do */
    primitive_lighting.finish(data, state);
}

static void
smooth_lighting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    int light_top = 1;
    int light_left = 1;
    int light_right = 1;
    RenderPrimitiveSmoothLighting *self = (RenderPrimitiveSmoothLighting *)data;
    
    /* special case for leaves, water 8, water 9, ice 79
       -- these are also smooth-lit! */
    if (state->block != 18 && state->block != 8 && state->block != 9 && state->block != 79 && is_transparent(state->block))
    {
        /* transparent blocks are rendered as usual, with flat lighting */
        primitive_lighting.draw(data, state, src, mask, mask_light);
        return;
    }
    
    /* non-transparent blocks get the special smooth treatment */
    
    /* special code for water */
    if (state->block == 9)
    {
        if (!(state->block_pdata & (1 << 4)))
            light_top = 0;
        if (!(state->block_pdata & (1 << 1)))
            light_left = 0;
        if (!(state->block_pdata & (1 << 2)))
            light_right = 0;
    }
    
    if (light_top)
        do_shading_with_rule(self, state, lighting_rules[FACE_TOP]);
    if (light_left)
        do_shading_with_rule(self, state, lighting_rules[FACE_LEFT]);
    if (light_right)
        do_shading_with_rule(self, state, lighting_rules[FACE_RIGHT]);
}

RenderPrimitiveInterface primitive_smooth_lighting = {
    "smooth-lighting", sizeof(RenderPrimitiveSmoothLighting),
    smooth_lighting_start,
    smooth_lighting_finish,
    NULL,
    NULL,
    smooth_lighting_draw,
};
