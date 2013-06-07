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
netherold_hidden(void *data, RenderState *state, int x, int y, int z) {
    /* hide all blocks above all air blocks
       
       due to how the nether is currently generated, this will also count
       empty sections as 'solid'
    */
    unsigned char missing_section = 0;
    while (y < (SECTIONS_PER_CHUNK - state->chunky) * 16)
    {
        if (state->chunks[1][1].sections[state->chunky + (y / 16)].blocks == NULL) {
            missing_section = 1;
            y += 16;
            continue;
        } else {
            /* if we passed through a missing section, but now are back in,
               that counts as air */
            if (missing_section)
                return 0;
            missing_section = 0;
        }
        
        if (!missing_section && get_data(state, BLOCKS, x, y, z) == 0)
        {
                return 0;
        } 
        y++;
    }
    return 1;
}

RenderPrimitiveInterface primitive_nether_old = {
    "netherold", 0,
    NULL,
    NULL,
    NULL,
    netherold_hidden,
    NULL,
};
