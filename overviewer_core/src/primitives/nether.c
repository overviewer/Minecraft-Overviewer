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
#include "nether.h"

static void
walk_chunk(RenderState *state, RenderPrimitiveNether *data) {
    int x, y, z;
    int id;

    for (x = 0; x < WIDTH; x++) {
        for (z = 0; z < DEPTH; z++) {
            id = get_data(state, BLOCKS, x, NETHER_ROOF - (state->chunky * 16), z);
            if (id == 7) {
                data->remove_block[x][NETHER_ROOF][z] = 1;
                id = get_data(state, BLOCKS, x, (NETHER_ROOF + 1) - (state->chunky * 16), z);
                if (id == 39 || id == 40)
                    data->remove_block[x][NETHER_ROOF + 1][z] = 1;
            }

            for (y = NETHER_ROOF-1; y>=0; y--) {
                id = get_data(state, BLOCKS, x, y - (state->chunky * 16), z);
                if (id == 7 || id == 87)
                    data->remove_block[x][y][z] = 1;
                else
                    break;
            }
        }
    }
    data->walked_chunk = 1;
}

static int
nether_hidden(void *data, RenderState *state, int x, int y, int z) {
    RenderPrimitiveNether* self;
    int real_y;

    self = (RenderPrimitiveNether *)data;

    if (!(self->walked_chunk))
        walk_chunk(state, self);

    real_y = y + (state->chunky * 16);
    return self->remove_block[x][real_y][z];
}

RenderPrimitiveInterface primitive_nether = {
    "nether", sizeof(RenderPrimitiveNether),
    NULL,
    NULL,
    NULL,
    nether_hidden,
    NULL,
};
