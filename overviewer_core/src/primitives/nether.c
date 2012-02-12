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

static int
nether_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* hide all blocks above all air blocks */
    while (z < 128)
    {
        if (getArrayByte3D(state->blocks, x, y, z) == 0)
        {
            return 0;
            break;
        }
        z++;
    }
    return 1;
}

RenderPrimitiveInterface primitive_nether = {
    "nether", 0,
    NULL,
    NULL,
    NULL,
    nether_hidden,
    NULL,
};
