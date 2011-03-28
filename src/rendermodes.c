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
#include <string.h>

/* decides which render mode to use */
RenderModeInterface *get_render_mode(RenderState *state) {
    /* default: normal */
    RenderModeInterface *iface = &rendermode_normal;
    PyObject *rendermode_py = PyObject_GetAttrString(state->self, "rendermode");
    const char *rendermode = PyString_AsString(rendermode_py);
    
    if (strcmp(rendermode, "lighting") == 0) {
        iface = &rendermode_lighting;
    } else if (strcmp(rendermode, "night") == 0) {
        iface = &rendermode_night;
    } else if (strcmp(rendermode, "spawn") == 0) {
        iface = &rendermode_spawn;
    }
    
    Py_DECREF(rendermode_py);
    return iface;
}
