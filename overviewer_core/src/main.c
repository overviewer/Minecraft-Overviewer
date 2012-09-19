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
    
    {"resize_half", resize_half_wrap, METH_VARARGS,
     "downscale image to half size"},
    
    {"render_loop", chunk_render, METH_VARARGS,
     "Renders stuffs"},
    
    {"extension_version", get_extension_version, METH_VARARGS, 
        "Returns the extension version"},
    
    {NULL, NULL, 0, NULL}       /* Sentinel */
};


PyMODINIT_FUNC
initc_overviewer(void)
{
    PyObject *mod, *numpy;
    mod = Py_InitModule("c_overviewer", COverviewerMethods);

    /* for numpy
       normally you should use import_array(), but that will break across
       numpy versions. This doesn't, and we don't use enough of numpy to worry
       about the API changing too much, so this is fine.
    */
    numpy = PyImport_ImportModule("numpy.core.multiarray");
    Py_XDECREF(numpy);

    /* initialize, and return if error is set */
    if (!init_chunk_render()) {
        PyErr_Print();
        exit(1);
        return;
    }

    init_endian();
}
