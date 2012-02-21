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
    PyObject *facemasks_py;
    PyObject *facemasks[3];
    
    /* light color image, loaded if color_light is True */
    PyObject *lightcolor;
    
    /* can be overridden in derived rendermodes to control lighting
       arguments are data, skylight, blocklight, return RGB */
    void (*calculate_light_color)(void *, unsigned char, unsigned char, unsigned char *, unsigned char *, unsigned char *);
    
    /* can be set to 0 in derived modes to indicate that lighting the chunk
     * sides is actually important. Right now, this is used in cave mode
     */
    int skip_sides;
    
    float strength;
    int color;
    int night;
} RenderPrimitiveLighting;

/* exposed so that smooth-lighting can use them */
extern RenderPrimitiveInterface primitive_lighting;
int lighting_is_face_occluded(RenderState *state, int skip_sides, int x, int y, int z);
void get_lighting_color(RenderPrimitiveLighting *self, RenderState *state,
                        int x, int y, int z,
                        unsigned char *r, unsigned char *g, unsigned char *b);
