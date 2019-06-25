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

typedef struct {
    PyObject* facemasks_py;
    PyObject* facemasks[3];

    /* light color image, loaded if color_light is True */
    PyObject* lightcolor;

    /* can be overridden in derived rendermodes to control lighting
       arguments are data, skylight, blocklight, return RGB */
    void (*calculate_light_color)(void*, uint8_t, uint8_t, uint8_t*, uint8_t*, uint8_t*);

    /* can be set to 0 in derived modes to indicate that lighting the chunk
     * sides is actually important. Right now, this is used in cave mode
     */
    bool skip_sides;

    float strength;
    int32_t color;
    int32_t night;
} RenderPrimitiveLighting;

/* exposed so that smooth-lighting can use them */
extern RenderPrimitiveInterface primitive_lighting;
bool lighting_is_face_occluded(RenderState* state, bool skip_sides, int32_t x, int32_t y, int32_t z);
void get_lighting_color(RenderPrimitiveLighting* self, RenderState* state,
                        int32_t x, int32_t y, int32_t z,
                        uint8_t* r, uint8_t* g, uint8_t* b);
