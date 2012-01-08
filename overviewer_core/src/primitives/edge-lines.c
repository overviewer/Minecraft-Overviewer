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

typedef struct {
    float opacity;
} PrimitiveEdgeLines;

static int
edge_lines_start(void *data, RenderState *state, PyObject *support) {
    PrimitiveEdgeLines *self = (PrimitiveEdgeLines *)data;
    
    self->opacity = 0.15;
    if (!render_mode_parse_option(support, "opacity", "f", &(self->opacity)))
        return 1;
    return 0;
}

static void
edge_lines_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    PrimitiveEdgeLines *self = (PrimitiveEdgeLines *)data;

    /* Draw some edge lines! */
    // draw.line(((imgx+12,imgy+increment), (imgx+22,imgy+5+increment)), fill=(0,0,0), width=1)
    if (state->block == 44 || state->block == 78 || !is_transparent(state->block)) {
        Imaging img_i = imaging_python_to_c(state->img);
        unsigned char ink[] = {0, 0, 0, 255 * self->opacity};

        int increment=0;
        if (state->block == 44)  // half-step
            increment=6;
        else if ((state->block == 78) || (state->block == 93) || (state->block == 94)) // snow, redstone repeaters (on and off)
            increment=9;

        if ((state->x == 15) && (state->up_right_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_right_blocks, 0, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        } else if (state->x != 15) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x+1, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        }
        // if y != 0 and blocks[x,y-1,z] == 0

        // chunk boundries are annoying
        if ((state->y == 0) && (state->up_left_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_left_blocks, state->x, 15, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        } else if (state->y != 0) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x, state->y-1, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                // draw.line(((imgx,imgy+6+increment), (imgx+12,imgy+increment)), fill=(0,0,0), width=1)
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        }
    }
}

RenderPrimitiveInterface primitive_edge_lines = {
    "edge-lines", sizeof(PrimitiveEdgeLines),
    edge_lines_start,
    NULL,
    NULL,
    NULL,
    edge_lines_draw,
};
