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

import numpy

class BiomeDispensary:
    """Turns biome arrays of either 256 or 1024 integer values into 16x16 2d arrays,
    which can then be retrieved for any Y level with get_biome.
    """
    def __init__(self, biome_array):
        self.biome_len = len(biome_array)
        if self.biome_len == 256:
            self.biomes = [biome_array.reshape((16, 16))]
        elif self.biome_len == 1024:
            self.biomes = [None] * 4
            for i in range(0, 4):
                # Map 256 values of the array to each self.biomes entry, resulting
                # in 4 entries
                self.biomes[i] = biome_array[i * 256:(i + 1) * 256].reshape((16, 16))

    def get_biome(self, y_level):
        if self.biome_len == 256 or y_level < 0:
            return self.biomes[0]
        else:
            # We clamp the value to a max of 3 here because apparently Y=16
            # also exists, and while I don't know what biome level Mojang uses for
            # that, the highest one is probably a good bet.
            return self.biomes[min(y_level // 4, 3)]
