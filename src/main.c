#include "overviewer.h"

#include <numpy/arrayobject.h>

static PyMethodDef COverviewerMethods[] = {
	{"alpha_over", alpha_over_wrap, METH_VARARGS,
        "alpha over composite function"},
    {"render_loop", chunk_render, METH_VARARGS,
        "Renders stuffs"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initc_overviewer(void)
{
        (void) Py_InitModule("c_overviewer", COverviewerMethods);
        import_array();  // for numpy
}
