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

PyObject *get_extension_version(PyObject *self, PyObject *args) {

    return Py_BuildValue("i", OVERVIEWER_EXTENSION_VERSION);
}

static PyMethodDef COverviewerMethods[] = {
    {"alpha_over", alpha_over_wrap, METH_VARARGS,
     "alpha over composite function"},
    
    {"init_chunk_render", init_chunk_render, METH_VARARGS,
     "Initializes the stuffs renderer."},
    {"render_loop", chunk_render, METH_VARARGS,
     "Renders stuffs"},
    
    {"get_render_modes", get_render_modes, METH_VARARGS,
     "returns available render modes"},
    {"get_render_mode_info", get_render_mode_info, METH_VARARGS,
     "returns info for a particular render mode"},
    {"get_render_mode_parent", get_render_mode_parent, METH_VARARGS,
     "returns parent for a particular render mode"},
    {"get_render_mode_inheritance", get_render_mode_inheritance, METH_VARARGS,
     "returns inheritance chain for a particular render mode"},
    {"get_render_mode_children", get_render_mode_children, METH_VARARGS,
     "returns (direct) children for a particular render mode"},
    
    {"extension_version", get_extension_version, METH_VARARGS, 
        "Returns the extension version"},
    
    {NULL, NULL, 0, NULL}       /* Sentinel */
};


PyMODINIT_FUNC
initc_overviewer(void)
{
    (void)Py_InitModule("c_overviewer", COverviewerMethods);
    /* for numpy */
    import_array();
    
    init_endian();
}
