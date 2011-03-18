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

#include <numpy/arrayobject.h>

static PyMethodDef COverviewerMethods[] = {
    {"alpha_over", alpha_over_wrap, METH_VARARGS,
     "alpha over composite function"},
    {"render_loop", chunk_render, METH_VARARGS,
     "Renders stuffs"},
    {NULL, NULL, 0, NULL}       /* Sentinel */
};

PyMODINIT_FUNC
initc_overviewer(void)
{
    (void)Py_InitModule("c_overviewer", COverviewerMethods);
    /* for numpy */
    import_array();
}
