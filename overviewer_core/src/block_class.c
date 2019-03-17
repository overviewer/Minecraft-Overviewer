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

#include "block_class.h"
#include "utils.h"


bool block_class_is_subset(
	mc_block_t block,
	const mc_block_t block_class[],
	size_t     block_class_len
)
{
	size_t i;
	for( i = 0; i < block_class_len; ++i )
	{
		if( block == block_class[i] )
		{
			return true;
		}
	}
	return false;
}


const mc_block_t block_class_stair[] = {
	block_oak_stairs,
	block_stone_stairs,
	block_brick_stairs,
	block_stone_brick_stairs,
	block_nether_brick_stairs,
	block_sandstone_stairs,
	block_spruce_stairs,
	block_birch_stairs,
	block_jungle_stairs,
	block_quartz_stairs,
	block_acacia_stairs,
	block_dark_oak_stairs,
	block_red_sandstone_stairs,
	block_purpur_stairs
};
const size_t block_class_stair_len = count_of(block_class_stair);

const mc_block_t block_class_door[] = {
	block_wooden_door,
	block_iron_door,
	block_spruce_door,
	block_birch_door,
	block_jungle_door,
	block_acacia_door,
	block_dark_oak_door
};
const size_t block_class_door_len = count_of(block_class_door);

const mc_block_t block_class_fence[] = {
	block_fence,
	block_nether_brick_fence,
	block_spruce_fence,
	block_birch_fence,
	block_jungle_fence,
	block_dark_oak_fence,
	block_acacia_fence
};
const size_t block_class_fence_len = count_of(block_class_fence);

const mc_block_t block_class_fence_gate[] = {
	block_fence_gate,
	block_spruce_fence_gate,
	block_birch_fence_gate,
	block_jungle_fence_gate,
	block_dark_oak_fence_gate,
	block_acacia_fence_gate
};
const size_t block_class_fence_gate_len = count_of(block_class_fence_gate);

const mc_block_t block_class_ancil[] = {
	block_wooden_door,
	block_iron_door,
	block_spruce_door,
	block_birch_door,
	block_jungle_door,
	block_acacia_door,
	block_dark_oak_door,
	block_oak_stairs,
	block_stone_stairs,
	block_brick_stairs,
	block_stone_brick_stairs,
	block_nether_brick_stairs,
	block_sandstone_stairs,
	block_spruce_stairs,
	block_birch_stairs,
	block_jungle_stairs,
	block_quartz_stairs,
	block_acacia_stairs,
	block_dark_oak_stairs,
	block_red_sandstone_stairs,
	block_purpur_stairs,
	block_grass,
	block_flowing_water,
	block_water,
	block_glass,
	block_chest,
	block_redstone_wire,
	block_ice,
	block_fence,
	block_portal,
	block_iron_bars,
	block_glass_pane,
	block_waterlily,
	block_nether_brick_fence,
	block_cobblestone_wall,
	block_double_plant,
	block_stained_glass_pane,
	block_stained_glass,
	block_trapped_chest,
	block_spruce_fence,
	block_birch_fence,
	block_jungle_fence,
	block_dark_oak_fence,
	block_acacia_fence
};
const size_t block_class_ancil_len = count_of(block_class_ancil);