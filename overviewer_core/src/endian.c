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

/* simple routines for dealing with endian conversion */

#include <stdint.h>

#define UNKNOWN_ENDIAN 0
#define BIG_ENDIAN 1
#define LITTLE_ENDIAN 2

static int32_t endianness = UNKNOWN_ENDIAN;

void init_endian(void) {
    /* figure out what our endianness is! */
    int16_t word = 0x0001;
    uint8_t* byte = (uint8_t*)(&word);
    endianness = byte[0] ? LITTLE_ENDIAN : BIG_ENDIAN;
}

uint16_t big_endian_ushort(uint16_t in) {
    return (endianness == LITTLE_ENDIAN) ? ((in >> 8) | (in << 8)) : in;
}

uint32_t big_endian_uint(uint32_t in) {
    return (endianness == LITTLE_ENDIAN) ? (((in & 0x000000FF) << 24) + ((in & 0x0000FF00) << 8) + ((in & 0x00FF0000) >> 8) + ((in & 0xFF000000) >> 24)) : in;
}
