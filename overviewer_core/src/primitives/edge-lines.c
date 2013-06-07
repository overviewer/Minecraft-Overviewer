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
    if (!render_mode_parse_option(support, "opacity", "f", &(self->opacity)))
        return 1;
    return 0;
}

static void
edge_lines_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    PrimitiveEdgeLines *self = (PrimitiveEdgeLines *)data;

    /* Draw some edge lines! */
    if (state->block == 44 || state->block == 78 || !is_transparent(state->block)) {
        Imaging img_i = imaging_python_to_c(state->img);
        unsigned char ink[] = {0, 0, 0, 255 * self->opacity};
        unsigned short side_block;
        int x = state->x, y = state->y, z = state->z;

        int increment=0;
        if ((state->block == 44 || state->block == 126) && ((state->block_data & 0x8) == 0 ))  // half-steps BUT no upsidown half-steps
            increment=6;
        else if ((state->block == 78) || (state->block == 93) || (state->block == 94)) // snow, redstone repeaters (on and off)
            increment=9;
        
        /* +X side */
        side_block = get_data(state, BLOCKS, x+1, y, z);
        if (side_block != state->block && (is_transparent(side_block) || render_mode_hidden(state->rendermode, x+1, y, z)) && 
            /* WARNING: ugly special case approaching */
            /* if the block is a slab and the side block is a stair don't draw anything, it can give very ugly results */
            !((state->block == 44 || state->block == 126) && ((side_block == 53) || (side_block == 67) || (side_block == 108) ||
            (side_block == 109) || (side_block == 114) || (side_block == 128) || (side_block == 134) || (side_block == 135) ||
            (side_block == 136)))) {
            ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
            ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
        }
        
        /* -Z side */
        side_block = get_data(state, BLOCKS, x, y, z-1);
        if (side_block != state->block && (is_transparent(side_block) || render_mode_hidden(state->rendermode, x, y, z-1)) &&
            /* WARNING: ugly special case approaching */
            /* if the block is a slab and the side block is a stair don't draw anything, it can give very ugly results */
            !((state->block == 44 || state->block == 126) && ((side_block == 53) || (side_block == 67) || (side_block == 108) ||
            (side_block == 109) || (side_block == 114) || (side_block == 128) || (side_block == 134) || (side_block == 135) ||
            (side_block == 136)))) {
            ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
            ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
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
