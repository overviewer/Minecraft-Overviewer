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

#include <stdint.h>

#define DEFAULT_BIOME 4 /* forest, nice and green */

typedef struct {
    const char* name;

    float temperature;
    float rainfall;

    uint32_t r, g, b;
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
    {"Mountains", 0.2, 0.3, 255, 255, 255},
    {"Forest", 0.7, 0.8, 255, 255, 255},
    /* 5 */
    {"Taiga", 0.05, 0.8, 255, 255, 255},
    {"Swamp", 0.8, 0.9, 205, 128, 255},
    {"River", 0.5, 0.5, 255, 255, 255},
    {"Nether", 2.0, 0.0, 255, 255, 255},
    {"The End", 0.5, 0.5, 255, 255, 255},
    /* 10 */
    {"Frozen Ocean", 0.0, 0.5, 255, 255, 255},
    {"Frozen River", 0.0, 0.5, 255, 255, 255},
    {"Snowy Tundra", 0.0, 0.5, 255, 255, 255},
    {"Snowy Mountains", 0.0, 0.5, 255, 255, 255},
    {"Mushroom Fields", 0.9, 1.0, 255, 255, 255},
    /* 15 */
    {"Mushroom Field Shore", 0.9, 1.0, 255, 255, 255},
    {"Beach", 0.8, 0.4, 255, 255, 255},
    {"Desert Hills", 2.0, 0.0, 255, 255, 255},
    {"Wooded Hills", 0.7, 0.8, 255, 255, 255},
    {"Taiga Hills", 0.05, 0.8, 255, 255, 255},
    /* 20 */
    {"Mountain Edge", 0.2, 0.3, 255, 255, 255},
    /* Values below are guesses */
    {"Jungle", 2.0, 0.45, 255, 255, 255},
    {"Jungle Hills", 2.0, 0.45, 255, 255, 255},
    {"Jungle Edge", 2.0, 0.45, 255, 255, 255},
    {"Deep Ocean", 0.0, 1, 255, 255, 255},
    /* 25 */
    {"Stone Shore", 0.2, 1, 255, 255, 255},
    {"Snowy Beach", 0.2, 1, 255, 255, 255},
    {"Birch Forest", 0.7, 0.8, 255, 255, 255},
    {"Birch Forest Hills", 0.7, 0.8, 255, 255, 255},
    {"Dark Forest", 2.0, 0.45, 255, 255, 255},
    /* 30 */
    {"Snowy Taiga", 0.05, 0.8, 255, 255, 255},
    {"Snowy Taiga Hills", 0.05, 0.8, 255, 255, 255},
    {"Giant Tree Taiga", 0.05, 0.8, 255, 255, 255},
    {"Giant Tree Taiga Hills", 0.05, 0.8, 255, 255, 255},
    {"Wooded Mountains", 0.2, 0.3, 255, 255, 255},
    /* 35 */
    {"Savanna", 1.0, 0.1, 255, 255, 255},
    {"Savanna Plateau", 1.0, 0.1, 255, 255, 255},
    {"Badlands", 1.8, 0.0, 255, 255, 255},
    {"Wooded Badlands Plateau", 1.8, 0.0, 255, 255, 255},
    {"Badlands Plateau", 1.8, 0.0, 255, 255, 255},
    /* 40 */
    {"Small End Islands", 0.5, 0.0, 255, 255, 255},
    {"End Midlands", 0.5, 0.0, 255, 255, 255},
    {"End Highlands", 0.5, 0.0, 255, 255, 255},
    {"End Barrens", 0.5, 0.0, 255, 255, 255},
    {"Warm Ocean", 0.5, 0.0, 255, 255, 255},
    /* 45 */
    {"Lukewarm Ocean", 0.5, 0.0, 255, 255, 255},
    {"Cold Ocean", 0.5, 0.0, 255, 255, 255},
    {"Deep Warm Ocean", 0.5, 0.0, 255, 255, 255},
    {"Deep Lukewarm Ocean", 0.5, 0.0, 255, 255, 255},
    {"Deep Cold Ocean", 0.5, 0.0, 255, 255, 255},
    //* 50 */
    {"Deep Frozen Ocean", 0.5, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 55 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 60 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 65 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 70 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 75 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 80 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 85 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 90 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 95 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 100 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 105 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 110 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 115 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 120 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 125 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"The Void", 0.5, 0.0, 255, 255, 255},
    {"Plains M", 0.8, 0.4, 255, 255, 255},
    {"Sunflower Plains", 0.8, 0.4, 255, 255, 255},
    /* 130 */
    {"Desert Lakes", 2.0, 0.0, 255, 255, 255},
    {"Gravelly Mountains", 0.2, 0.3, 255, 255, 255},
    {"Flower Forest", 0.7, 0.8, 255, 255, 255},
    {"Taiga Mountains", 0.25, 0.8, 255, 255, 255},
    {"Swamp Hills", 0.8, 0.9, 205, 128, 255},
    /* 135 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 140 */
    {"Ice Spikes", 0.12, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 145 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"Modified Jungle", 0.95, 0.8, 255, 255, 255},
    /* 150 */
    {"", 0.0, 0.0, 255, 255, 255},
    {"Modified Jungle Edge", 0.95, 0.8, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 155 */
    {"Tall Birch Forest", 0.6, 0.6, 255, 255, 255},
    {"Tall Birch Hills", 0.6, 0.6, 255, 255, 255},
    {"Dark Forest Hills", 0.7, 0.8, 255, 255, 255},
    {"Snowy Taiga Mountains", 0.0, 0.0, 255, 255, 255},
    {"", 0.0, 0.0, 255, 255, 255},
    /* 160 */
    {"Giant Spruce Taiga", 0.25, 0.8, 255, 255, 255},
    {"Giant Spruce Taiga Hills", 0.25, 0.8, 255, 255, 255},
    {"Gravelly Mountains+", 0.2, 0.3, 255, 255, 255},
    {"Shattered Savanna", 2.0, 0.0, 255, 255, 255},
    {"Shattered Savanna Plateau", 2.0, 0.0, 255, 255, 255},
    /* 165 */
    {"Eroded Badlands", 0.0, 0.0, 255, 255, 255},
    {"Modified Wooded Badlands Plateau", 0.0, 0.0, 255, 255, 255},
    {"Modified Badlands Plateau", 0.0, 0.0, 255, 255, 255},
    {"Bamboo Jungle", 0.95, 0.8, 255, 255, 255},
    {"Bamboo Jungle Hills", 0.95, 0.8, 255, 255, 255},
};

#define NUM_BIOMES (sizeof(biome_table) / sizeof(Biome))
