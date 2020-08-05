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

#ifndef __BLOCK_CLASS_H_INCLUDED__
#define __BLOCK_CLASS_H_INCLUDED__

#include <stdbool.h>
#include <stddef.h>

#include "mc_id.h"

bool block_class_is_subset(
    mc_block_t block,
    const mc_block_t block_class[],
    size_t block_class_len);

bool block_class_is_wall(mc_block_t block);

extern const mc_block_t block_class_stair[];
extern const size_t block_class_stair_len;

extern const mc_block_t block_class_door[];
extern const size_t block_class_door_len;

extern const mc_block_t block_class_fence[];
extern const size_t block_class_fence_len;

extern const mc_block_t block_class_fence_gate[];
extern const size_t block_class_fence_gate_len;

extern const mc_block_t block_class_ancil[];
extern const size_t block_class_ancil_len;

extern const mc_block_t block_class_alt_height[];
extern const size_t block_class_alt_height_len;

extern const mc_block_t block_class_nether_roof[];
extern const size_t block_class_nether_roof_len;

#endif
