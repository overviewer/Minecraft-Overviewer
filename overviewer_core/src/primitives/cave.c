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

#include <math.h>
#include "../overviewer.h"

typedef struct {
    int32_t only_lit;
} RenderPrimitiveCave;

static inline bool
touches_light(RenderState* state, DataType type, uint32_t x, uint32_t y, uint32_t z) {
    if (get_data(state, type, x, y + 1, z))
        return true;
    if (get_data(state, type, x + 1, y, z))
        return true;
    if (get_data(state, type, x - 1, y, z))
        return true;
    if (get_data(state, type, x, y, z + 1))
        return true;
    if (get_data(state, type, x, y, z - 1))
        return true;
    return false;
}

static bool
cave_occluded(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    /* check for normal occlusion */
    /* use ajacent chunks, if not you get blocks spreaded in chunk edges */

    if (!is_known_transparent(get_data(state, BLOCKS, x - 1, y, z)) &&
        !is_known_transparent(get_data(state, BLOCKS, x, y, z + 1)) &&
        !is_known_transparent(get_data(state, BLOCKS, x, y + 1, z))) {
        return true;
    }

    /* special handling for section boundaries */
    if (x == 0 && (!(state->chunks[0][1].loaded) || state->chunks[0][1].sections[state->chunky].blocks == NULL))
        return true;
    if (y == 15 && (state->chunky + 1 >= SECTIONS_PER_CHUNK || state->chunks[1][1].sections[state->chunky + 1].blocks == NULL))
        return true;
    if (z == 15 && (!(state->chunks[1][2].loaded) || state->chunks[1][2].sections[state->chunky].blocks == NULL))
        return true;

    return false;
}

static bool
cave_hidden(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    RenderPrimitiveCave* self;
    int32_t dy = 0;
    uint16_t blockID;
    uint32_t blockUpID;
    self = (RenderPrimitiveCave*)data;

    /* check if the block is touching skylight */
    if (touches_light(state, SKYLIGHT, x, y, z)) {
        return true;
    }

    if (self->only_lit && !touches_light(state, BLOCKLIGHT, x, y, z)) {
        return true;
    }

    /* check for lakes and seas and don't render them
     * at this point of the code the block has no skylight
     * but a deep sea can be completely dark
     */

    blockID = getArrayShort3D(state->blocks, x, y, z);
    blockUpID = get_data(state, BLOCKS, x, y + 1, z);
    if (blockID == 9 || blockID == 8 || blockUpID == 9 || blockUpID == 8) {
        for (dy = y + 1; dy < (SECTIONS_PER_CHUNK - state->chunky) * 16; dy++) {
            /* go up and check for skylight */
            if (get_data(state, SKYLIGHT, x, dy, z) != 0) {
                return true;
            }
            blockUpID = get_data(state, BLOCKS, x, dy, z);
            if (blockUpID != 8 && blockUpID != 9) {
                /* we are out of the water! and there's no skylight
                 * , i.e. is a cave lake or something similar */
                break;
            }
        }
    }

    /* unfortunate side-effect of lit cave mode: we need to count occluded
     * blocks as hidden for the lighting to look right, since technically our
     * hiding depends on occlusion as well
     */
    return cave_occluded(data, state, x, y, z);
}

static bool
cave_start(void* data, RenderState* state, PyObject* support) {
    RenderPrimitiveCave* self;
    self = (RenderPrimitiveCave*)data;

    if (!render_mode_parse_option(support, "only_lit", "i", &(self->only_lit)))
        return true;

    return false;
}

RenderPrimitiveInterface primitive_cave = {
    "cave",
    sizeof(RenderPrimitiveCave),
    cave_start,
    NULL,
    cave_occluded,
    cave_hidden,
    NULL,
};
