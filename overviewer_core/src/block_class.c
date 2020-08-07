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

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

bool block_class_is_subset(
    mc_block_t block,
    const mc_block_t block_class[],
    size_t block_class_len) {
    size_t i = 0;

#ifdef __SSE2__
    for (; i / 8 < block_class_len / 8; i += 8) {
        const __m128i block_class_vec = _mm_loadu_si128(
            (__m128i*)&block_class[i]);
        const __m128i block_vec = _mm_set1_epi16(block);
        const __m128i block_cmp = _mm_cmpeq_epi16(block_vec, block_class_vec);
        if (_mm_movemask_epi8(block_cmp)) {
            return true;
        }
    }
#endif
#ifdef __MMX__
    for (; i / 4 < block_class_len / 4; i += 4) {
        const __m64 block_class_vec = _mm_cvtsi64_m64(
            *(uint64_t*)&block_class[i]);
        const __m64 block_vec = _mm_set1_pi16(block);
        const __m64 block_cmp = _mm_cmpeq_pi16(block_vec, block_class_vec);
        if (_mm_cvtm64_si64(block_cmp)) {
            return true;
        }
    }
#endif
    for (; i < block_class_len; ++i) {
        if (block == block_class[i]) {
            return true;
        }
    }
    return false;
}

bool block_class_is_wall(mc_block_t block) {
    mc_block_t mask = 0b11111111;
    mc_block_t prefix = 0b111 << 8;     // 1792 is the starting offset
    // if the xor zeroes all bits, the prefix must've matched.
    return ((block & ~mask) ^ prefix) == 0;
}

const mc_block_t block_class_stair[] = {
    block_oak_stairs,
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
    block_smooth_red_sandstone_stairs,
    block_purpur_stairs,
    block_prismarine_stairs,
    block_dark_prismarine_stairs,
    block_prismarine_brick_stairs,
    block_mossy_cobblestone_stairs,
    block_cobblestone_stairs,
    block_smooth_quartz_stairs,
    block_polished_granite_stairs,
    block_polished_diorite_stairs,
    block_polished_andesite_stairs,
    block_stone_stairs,
    block_granite_stairs,
    block_diorite_stairs,
    block_andesite_stairs,
    block_end_stone_brick_stairs,
    block_red_nether_brick_stairs,
    block_mossy_stone_brick_stairs,
    block_smooth_sandstone_stairs};
const size_t block_class_stair_len = COUNT_OF(block_class_stair);

const mc_block_t block_class_door[] = {
    block_wooden_door,
    block_iron_door,
    block_spruce_door,
    block_birch_door,
    block_jungle_door,
    block_acacia_door,
    block_dark_oak_door};
const size_t block_class_door_len = COUNT_OF(block_class_door);

const mc_block_t block_class_fence[] = {
    block_fence,
    block_nether_brick_fence,
    block_spruce_fence,
    block_birch_fence,
    block_jungle_fence,
    block_dark_oak_fence,
    block_acacia_fence};
const size_t block_class_fence_len = COUNT_OF(block_class_fence);

const mc_block_t block_class_fence_gate[] = {
    block_fence_gate,
    block_spruce_fence_gate,
    block_birch_fence_gate,
    block_jungle_fence_gate,
    block_dark_oak_fence_gate,
    block_acacia_fence_gate};
const size_t block_class_fence_gate_len = COUNT_OF(block_class_fence_gate);

const mc_block_t block_class_ancil[] = {
    block_wooden_door,
    block_iron_door,
    block_spruce_door,
    block_birch_door,
    block_jungle_door,
    block_acacia_door,
    block_dark_oak_door,
    block_oak_stairs,
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
    block_smooth_red_sandstone_stairs,
    block_purpur_stairs,
    block_prismarine_stairs,
    block_dark_prismarine_stairs,
    block_prismarine_brick_stairs,
    block_cobblestone_stairs,
    block_mossy_cobblestone_stairs,
    block_mossy_stone_brick_stairs,
    block_smooth_quartz_stairs,
    block_polished_granite_stairs,
    block_polished_diorite_stairs,
    block_polished_andesite_stairs,
    block_stone_stairs,
    block_granite_stairs,
    block_diorite_stairs,
    block_andesite_stairs,
    block_end_stone_brick_stairs,
    block_red_nether_brick_stairs,
    block_smooth_sandstone_stairs,
    block_grass,
    block_flowing_water,
    block_water,
    block_glass,
    block_redstone_wire,
    block_ice,
    block_fence,
    block_portal,
    block_iron_bars,
    block_glass_pane,
    block_waterlily,
    block_nether_brick_fence,
    block_andesite_wall,
    block_brick_wall,
    block_cobblestone_wall,
    block_diorite_wall,
    block_end_stone_brick_wall,
    block_granite_wall,
    block_mossy_cobblestone_wall,
    block_mossy_stone_brick_wall,
    block_nether_brick_wall,
    block_prismarine_wall,
    block_red_nether_brick_wall,
    block_red_sandstone_wall,
    block_sandstone_wall,
    block_stone_brick_wall,
    block_double_plant,
    block_stained_glass_pane,
    block_stained_glass,
    block_spruce_fence,
    block_birch_fence,
    block_jungle_fence,
    block_dark_oak_fence,
    block_acacia_fence};
const size_t block_class_ancil_len = COUNT_OF(block_class_ancil);

const mc_block_t block_class_alt_height[] = {
    block_stone_slab,
    block_oak_stairs,
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
    block_smooth_red_sandstone_stairs,
    block_prismarine_stairs,
    block_dark_prismarine_stairs,
    block_prismarine_brick_stairs,
    block_cobblestone_stairs,
    block_mossy_cobblestone_stairs,
    block_mossy_stone_brick_stairs,
    block_smooth_quartz_stairs,
    block_polished_granite_stairs,
    block_polished_diorite_stairs,
    block_polished_andesite_stairs,
    block_stone_stairs,
    block_granite_stairs,
    block_diorite_stairs,
    block_andesite_stairs,
    block_end_stone_brick_stairs,
    block_red_nether_brick_stairs,
    block_smooth_sandstone_stairs,
    block_prismarine_slab,
    block_dark_prismarine_slab,
    block_prismarine_brick_slab,
    block_andesite_slab,
    block_diorite_slab,
    block_granite_slab,
    block_polished_andesite_slab,
    block_polished_diorite_slab,
    block_polished_granite_slab,
    block_red_nether_brick_slab,
    block_smooth_sandstone_slab,
    block_cut_sandstone_slab,
    block_smooth_red_sandstone_slab,
    block_cut_red_sandstone_slab,
    block_end_stone_brick_slab,
    block_mossy_cobblestone_slab,
    block_mossy_stone_brick_slab,
    block_smooth_quartz_slab,
    block_smooth_stone_slab,
    block_stone_slab2,
    block_purpur_stairs,
    block_purpur_slab,
    block_wooden_slab};
const size_t block_class_alt_height_len = COUNT_OF(block_class_alt_height);

const mc_block_t block_class_nether_roof[] = {
    block_bedrock,
    block_netherrack,
    block_quartz_ore,
    block_lava,
    block_soul_sand,
    block_basalt,
    block_blackstone,
    block_soul_soil,
    block_nether_gold_ore};
const size_t block_class_nether_roof_len = COUNT_OF(block_class_nether_roof);
