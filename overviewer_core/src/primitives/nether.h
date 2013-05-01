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

#define NETHER_ROOF 127
#define WIDTH 16
#define DEPTH 16
#define HEIGHT 256

// add two to these because the primative functions should expect to
// deal with x and z values of -1 and 16
typedef struct {
    int walked_chunk;

    int remove_block[WIDTH+2][HEIGHT][DEPTH+2];
    
} RenderPrimitiveNether;
