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

#include "../block_class.h"
#include "../mc_id.h"
#include "../overviewer.h"

typedef struct {
    float opacity;
} PrimitiveEdgeLines;

static bool
edge_lines_start(void* data, RenderState* state, PyObject* support) {
    PrimitiveEdgeLines* self = (PrimitiveEdgeLines*)data;
    if (!render_mode_parse_option(support, "opacity", "f", &(self->opacity)))
        return true;
    return false;
}

static void
edge_lines_draw(void* data, RenderState* state, PyObject* src, PyObject* mask, PyObject* mask_light) {
    PrimitiveEdgeLines* self = (PrimitiveEdgeLines*)data;

    /* Draw some edge lines! */
    if (block_class_is_subset(state->block, (mc_block_t[]){block_stone_slab, block_snow_layer}, 2) || !is_transparent(state->block)) {
        Imaging img_i = imaging_python_to_c(state->img);
        uint8_t ink[] = {0, 0, 0, 255 * self->opacity};
        mc_block_t side_block;
        int32_t x = state->x, y = state->y, z = state->z;

        int32_t increment = 0;
        if (block_class_is_subset(state->block, (mc_block_t[]){block_wooden_slab, block_stone_slab}, 2) && ((state->block_data & 0x8) == 0)) // half-steps BUT no upsidown half-steps
            increment = 6;
        else if (block_class_is_subset(state->block, (mc_block_t[]){block_snow_layer, block_unpowered_repeater, block_powered_repeater}, 3)) // snow, redstone repeaters (on and off)
            increment = 9;

        /* +X side */
        side_block = get_data(state, BLOCKS, x + 1, y, z);
        if (side_block != state->block && (is_transparent(side_block) || render_mode_hidden(state->rendermode, x + 1, y, z)) &&
            /* WARNING: ugly special case approaching */
            /* if the block is a slab and the side block is a stair don't draw anything, it can give very ugly results */
            !(block_class_is_subset(state->block, (mc_block_t[]){block_wooden_slab, block_stone_slab}, 2) && (block_class_is_subset(side_block, block_class_stair, block_class_stair_len)))) {
            ImagingDrawLine(img_i, state->imgx + 12, state->imgy + 1 + increment, state->imgx + 22 + 1, state->imgy + 5 + 1 + increment, &ink, 1);
            ImagingDrawLine(img_i, state->imgx + 12, state->imgy + increment, state->imgx + 22 + 1, state->imgy + 5 + increment, &ink, 1);
        }

        /* -Z side */
        side_block = get_data(state, BLOCKS, x, y, z - 1);
        if (side_block != state->block && (is_transparent(side_block) || render_mode_hidden(state->rendermode, x, y, z - 1)) &&
            /* WARNING: ugly special case approaching */
            /* if the block is a slab and the side block is a stair don't draw anything, it can give very ugly results */
            !(
                block_class_is_subset(state->block, (mc_block_t[]){block_stone_slab, block_wooden_slab}, 2) && (block_class_is_subset(side_block, block_class_stair, block_class_stair_len)))) {
            ImagingDrawLine(img_i, state->imgx, state->imgy + 6 + 1 + increment, state->imgx + 12 + 1, state->imgy + 1 + increment, &ink, 1);
            ImagingDrawLine(img_i, state->imgx, state->imgy + 6 + increment, state->imgx + 12 + 1, state->imgy + increment, &ink, 1);
        }
    }
}

RenderPrimitiveInterface primitive_edge_lines = {
    "edge-lines",
    sizeof(PrimitiveEdgeLines),
    edge_lines_start,
    NULL,
    NULL,
    NULL,
    edge_lines_draw,
};
