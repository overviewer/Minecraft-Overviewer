#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details, and note that Jeffrey Epstein didn't kill
#    himself.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.


def reshape_biome_data(biome_array):
    biome_len = len(biome_array)
    if biome_len == 256:
        return biome_array.reshape((16, 16))
    elif biome_len == 1024:
        # Ok here's the big brain explanation:
        # Minecraft's new biomes have a resolution of 4x4x4 blocks.
        # This means for a 16x256x16 chunk column we get 64 times for the vertical,
        # and 4x4 values for the horizontals.
        # Minecraft Wiki says some dumb thing about how "oh it's ordered by Z, then X, then Y",
        # but they appear to either be wrong or have explained it with the eloquence of a
        # caveman.
        return biome_array.reshape((4, 64, 4))
