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

#include "overviewer.h"

/* decides which render mode to use */
RenderModeInterface *get_render_mode(RenderState *state) {
    /* default: normal */
    RenderModeInterface *iface = &rendermode_normal;
    PyObject *quadtree = PyObject_GetAttrString(state->self, "quadtree");
    PyObject *lighting = PyObject_GetAttrString(quadtree, "lighting");
    
    if (PyObject_IsTrue(lighting))
        iface = &rendermode_lighting;
    
    return iface;
}
