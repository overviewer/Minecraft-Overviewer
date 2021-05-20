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
#if defined(__aarch64__)
#include <arm_neon.h>
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
#if defined(__aarch64__)
    for (; i / 8 < block_class_len / 8; i += 8) {
        const uint16x8_t block_class_vec = vld1q_u16(
            (uint16_t*)&block_class[i]);
        const uint16x8_t block_vec = vmovq_n_u16(block);
        const uint16x8_t block_cmp = vceqq_u16(block_vec, (uint16x8_t) block_class_vec);
        if(vgetq_lane_s64((int64x2_t) block_cmp, 0) +
           vgetq_lane_s64((int64x2_t) block_cmp, 1)) {
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
    block_crimson_stairs,
    block_warped_stairs,
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
    block_smooth_sandstone_stairs,
    block_blackstone_stairs,
    block_polished_blackstone_stairs,
    block_polished_blackstone_brick_stairs};
const size_t block_class_stair_len = COUNT_OF(block_class_stair);

const mc_block_t block_class_door[] = {
    block_wooden_door,
    block_iron_door,
    block_spruce_door,
    block_birch_door,
    block_jungle_door,
    block_acacia_door,
    block_dark_oak_door,
    block_crimson_door,
    block_warped_door};
const size_t block_class_door_len = COUNT_OF(block_class_door);

const mc_block_t block_class_fence[] = {
    block_fence,
    block_nether_brick_fence,
    block_spruce_fence,
    block_birch_fence,
    block_jungle_fence,
    block_dark_oak_fence,
    block_crimson_fence,
    block_warped_fence,
    block_acacia_fence};
const size_t block_class_fence_len = COUNT_OF(block_class_fence);

const mc_block_t block_class_fence_gate[] = {
    block_fence_gate,
    block_spruce_fence_gate,
    block_birch_fence_gate,
    block_jungle_fence_gate,
    block_dark_oak_fence_gate,
    block_acacia_fence_gate,
    block_crimson_fence_gate,
    block_warped_fence_gate};
const size_t block_class_fence_gate_len = COUNT_OF(block_class_fence_gate);

const mc_block_t block_class_ancil[] = {
    block_wooden_door,
    block_iron_door,
    block_spruce_door,
    block_birch_door,
    block_jungle_door,
    block_acacia_door,
    block_dark_oak_door,
    block_crimson_door,
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
    block_crimson_stairs,
    block_warped_stairs,
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
    block_blackstone_stairs,
    block_polished_blackstone_stairs,
    block_polished_blackstone_brick_stairs,    
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
    block_blackstone_wall,
    block_polished_blackstone_wall,
    block_polished_blackstone_brick_wall,
    block_crying_obsidian,
    block_lodestone,
    block_respawn_anchor,
    block_double_plant,
    block_stained_glass_pane,
    block_stained_glass,
    block_spruce_fence,
    block_birch_fence,
    block_jungle_fence,
    block_dark_oak_fence,
    block_crimson_fence,
    block_warped_fence,
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
    block_crimson_stairs,
    block_warped_stairs,
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
    block_blackstone_stairs,
    block_polished_blackstone_stairs,
    block_polished_blackstone_brick_stairs,    
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
    block_blackstone_slab,
    block_polished_blackstone_slab,
    block_polished_blackstone_brick_slab,
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
    block_nether_gold_ore,
    block_ancient_debris};
const size_t block_class_nether_roof_len = COUNT_OF(block_class_nether_roof);

const mc_block_t block_connects_to_glass_pane[] = {
    block_stone,
    block_grass,
    block_dirt,
    block_cobblestone,
    block_planks,
    block_bedrock,
    block_sand,
    block_gravel,
    block_gold_ore,
    block_iron_ore,
    block_coal_ore,
    block_log,
    block_sponge,
    block_glass,
    block_lapis_ore,
    block_lapis_block,
    block_dispenser,
    block_sandstone,
    block_noteblock,
    block_sticky_piston,
    block_piston,
    block_wool,
    block_gold_block,
    block_iron_block,
    block_double_stone_slab,
    block_brick_block,
    block_tnt,
    block_bookshelf,
    block_mossy_cobblestone,
    block_obsidian,
    block_mob_spawner,
    block_diamond_ore,
    block_diamond_block,
    block_crafting_table,
    block_furnace,
    block_lit_furnace,
    block_redstone_ore,
    block_lit_redstone_ore,
    block_ice,
    block_snow,
    block_clay,
    block_jukebox,
    block_netherrack,
    block_soul_sand,
    block_glowstone,
    block_stained_glass,
    block_stonebrick,
    block_brown_mushroom_block,
    block_red_mushroom_block,
    block_iron_bars,
    block_glass_pane,
    block_mycelium,
    block_nether_brick,
    block_cauldron,
    block_end_stone,
    block_redstone_lamp,
    block_lit_redstone_lamp,
    block_double_wooden_slab,
    block_emerald_ore,
    block_emerald_block,
    block_beacon,
    block_mushroom_stem,
    block_redstone_block,
    block_quartz_ore,
    block_hopper,
    block_quartz_block,
    block_dropper,
    block_stained_hardened_clay,
    block_stained_glass_pane,
    block_prismarine,
    block_sea_lantern,
    block_hay_block,
    block_hardened_clay,
    block_coal_block,
    block_packed_ice,
    block_red_sandstone,
    block_double_stone_slab2,
    block_chorus_flower,
    block_purpur_block,
    block_purpur_pillar,
    block_purpur_double_slab,
    block_end_bricks,
    block_frosted_ice,
    block_magma,
    block_nether_wart_block,
    block_red_nether_brick,
    block_bone_block,
    block_observer,
    block_white_glazed_terracotta,
    block_orange_glazed_terracotta,
    block_magenta_glazed_terracotta,
    block_light_blue_glazed_terracotta,
    block_yellow_glazed_terracotta,
    block_lime_glazed_terracotta,
    block_pink_glazed_terracotta,
    block_gray_glazed_terracotta,
    block_light_gray_glazed_terracotta,
    block_cyan_glazed_terracotta,
    block_purple_glazed_terracotta,
    block_blue_glazed_terracotta,
    block_brown_glazed_terracotta,
    block_green_glazed_terracotta,
    block_red_glazed_terracotta,
    block_black_glazed_terracotta,
    block_concrete,
    block_concrete_powder,
    block_ancient_debris,
    block_basalt,
    block_polished_basalt,
    block_blackstone,
    block_netherite_block,
    block_warped_wart_block,
    block_shroomlight,
    block_soul_soil,
    block_nether_gold_ore,
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
    block_fletching_table,
    block_cartography_table,
    block_smithing_table,
    block_blast_furnace,
    block_smoker,
    block_loom,
    block_composter,
    block_beehive,
    block_bee_nest,
    block_honeycomb_block
};
const size_t block_connects_to_glass_pane_len = COUNT_OF(block_connects_to_glass_pane);

const mc_block_t block_class_trapdoor[] = {
    block_trapdoor,
    block_iron_trapdoor,
    block_spruce_trapdoor,
    block_birch_trapdoor,
    block_jungle_trapdoor,
    block_acacia_trapdoor,
    block_dark_oak_trapdoor
};
const size_t block_class_trapdoor_len = COUNT_OF(block_class_trapdoor);
