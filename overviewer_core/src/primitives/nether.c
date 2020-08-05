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

#include "nether.h"
#include "../block_class.h"
#include "../mc_id.h"
#include "../overviewer.h"

static void
walk_chunk(RenderState* state, RenderPrimitiveNether* data) {
    int32_t x, y, z;
    mc_block_t blockid;

    for (x = -1; x < WIDTH + 1; x++) {
        for (z = -1; z < DEPTH + 1; z++) {
            blockid = get_data(state, BLOCKS, x, NETHER_ROOF - (state->chunky * 16), z);
            if (blockid == block_bedrock) {
                data->remove_block[x + 1][NETHER_ROOF][z + 1] = true;
                blockid = get_data(state, BLOCKS, x, (NETHER_ROOF + 1) - (state->chunky * 16), z);
                if (blockid == block_brown_mushroom || blockid == block_red_mushroom)
                    data->remove_block[x + 1][NETHER_ROOF + 1][z + 1] = true;
            }

            for (y = NETHER_ROOF - 1; y >= 0; y--) {
                blockid = get_data(state, BLOCKS, x, y - (state->chunky * 16), z);
                if (block_class_is_subset(blockid, block_class_nether_roof, block_class_nether_roof_len))
                    data->remove_block[x + 1][y][z + 1] = true;
                else
                    break;
            }
        }
    }
    data->walked_chunk = true;
}

static bool
nether_hidden(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    RenderPrimitiveNether* self;
    int32_t real_y;

    self = (RenderPrimitiveNether*)data;

    if (!(self->walked_chunk))
        walk_chunk(state, self);

    real_y = y + (state->chunky * 16);
    return self->remove_block[x + 1][real_y][z + 1];
}

RenderPrimitiveInterface primitive_nether = {
    "nether",
    sizeof(RenderPrimitiveNether),
    NULL,
    NULL,
    NULL,
    nether_hidden,
    NULL,
};
