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


#define DEFAULT_BIOME 4 /* forest, nice and green */

typedef struct {
    const char* name;
    
    float temperature;
    float rainfall;
    
    unsigned int r, g, b;
} Biome;

/* each entry in this table is yanked *directly* out of the minecraft source
 * temp/rainfall are taken from what MCP calls setTemperatureRainfall
 *
 * Some biomes, like Swamp, do a bit of post-processing by multiplying on a
 * hard-coded color. The RGB tuple used follows the temp/rainfall.
 * 255, 255, 255 is white, which means do nothing
 *
 * keep in mind the x/y coordinate in the color tables is found *after*
 * multiplying rainfall and temperature for the second coordinate, *and* the
 * origin is in the lower-right. <3 biomes.
 */
static Biome biome_table[] = {
    /* 0 */
    {"Ocean", 0.5, 0.5, 255, 255, 255},
    {"Plains", 0.8, 0.4, 255, 255, 255},
    {"Desert", 2.0, 0.0, 255, 255, 255},
    {"Extreme Hills", 0.2, 0.3, 255, 255, 255},
    {"Forest", 0.7, 0.8, 255, 255, 255},
    /* 5 */
    {"Taiga", 0.05, 0.8, 255, 255, 255},
    {"Swampland", 0.8, 0.9, 205, 128, 255},
    {"River", 0.5, 0.5, 255, 255, 255},
    {"Hell", 2.0, 0.0, 255, 255, 255},
    {"Sky", 0.5, 0.5, 255, 255, 255},
    /* 10 */
    {"FrozenOcean", 0.0, 0.5, 255, 255, 255},
    {"FrozenRiver", 0.0, 0.5, 255, 255, 255},
    {"Ice Plains", 0.0, 0.5, 255, 255, 255},
    {"Ice Mountains", 0.0, 0.5, 255, 255, 255},
    {"MushroomIsland", 0.9, 1.0, 255, 255, 255},
    /* 15 */
    {"MushroomIslandShore", 0.9, 1.0, 255, 255, 255},
    {"Beach", 0.8, 0.4, 255, 255, 255},
    {"DesertHills", 2.0, 0.0, 255, 255, 255},
    {"ForestHills", 0.7, 0.8, 255, 255, 255},
    {"TaigaHills", 0.05, 0.8, 255, 255, 255},
    /* 20 */
    {"Extreme Hills Edge", 0.2, 0.3, 255, 255, 255},
    {"Jungle", 2.0, 0.45, 255, 255, 255}, /* <-- GUESS, but a good one */
    {"Jungle Mountains", 2.0, 0.45, 255, 255, 255}, /* <-- also a guess */
};

#define NUM_BIOMES (sizeof(biome_table) / sizeof(Biome))

