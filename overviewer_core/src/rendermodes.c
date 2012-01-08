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
#include <stdarg.h>

/* list of all render modes, ending in NULL
   all of these will be available to the user, so DON'T include modes
   that are only useful as a base for other modes. */
static RenderModeInterface *render_modes[] = {
    &rendermode_normal,
    &rendermode_lighting,
    &rendermode_smooth_lighting,
    &rendermode_cave,
    
    &rendermode_spawn,
    &rendermode_mineral,
    NULL
};

/* rendermode encapsulation */

RenderMode *render_mode_create(const char *mode, RenderState *state) {
    PyObject *options;
    RenderMode *ret = NULL;
    RenderModeInterface *iface = NULL;
    unsigned int i;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, mode) == 0) {
            iface = render_modes[i];
            break;
        }
    }

    if (iface == NULL)
        return NULL;
    
    options = PyDict_New();
    
    ret = calloc(1, sizeof(RenderMode));
    if (ret == NULL) {
        Py_DECREF(options);
        return PyErr_Format(PyExc_RuntimeError, "Failed to alloc a rendermode");
    }
    
    ret->mode = calloc(1, iface->data_size);
    if (ret->mode == NULL) {
        Py_DECREF(options);
        free(ret);
        return PyErr_Format(PyExc_RuntimeError, "Failed to alloc rendermode data");
    }
    
    ret->iface = iface;
    ret->state = state;
    
    if (iface->start(ret->mode, state, options)) {
        Py_DECREF(options);
        free(ret->mode);
        free(ret);
        return NULL;
    }
    
    Py_DECREF(options);
    return ret;
}

void render_mode_destroy(RenderMode *self) {
    self->iface->finish(self->mode, self->state);
    free(self->mode);
    free(self);
}

int render_mode_occluded(RenderMode *self, int x, int y, int z) {
    return self->iface->occluded(self->mode, self->state, x, y, z);
}

int render_mode_hidden(RenderMode *self, int x, int y, int z) {
    return self->iface->hidden(self->mode, self->state, x, y, z);
}

void render_mode_draw(RenderMode *self, PyObject *img, PyObject *mask, PyObject *mask_light) {
    self->iface->draw(self->mode, self->state, img, mask, mask_light);
}

/* options parse helper */
int render_mode_parse_option(PyObject *dict, const char *name, const char *format, ...) {
    va_list ap;
    PyObject *item;
    int ret;
    
    if (dict == NULL || name == NULL)
        return 1;
    
    item = PyDict_GetItemString(dict, name);
    if (item == NULL)
        return 1;
    
    /* make sure the item we're parsing is a tuple
       for VaParse to work correctly */
    if (!PyTuple_Check(item)) {
        item = PyTuple_Pack(1, item);
    } else {
        Py_INCREF(item);
    }
    
    va_start(ap, format);
    ret = PyArg_VaParse(item, format, ap);
    va_end(ap);
    
    Py_DECREF(item);
    
    if (!ret) {
        PyObject *errtype, *errvalue, *errtraceback;
        const char *errstring;
        
        PyErr_Fetch(&errtype, &errvalue, &errtraceback);
        errstring = PyString_AsString(errvalue);
        
        PyErr_Format(PyExc_TypeError, "rendermode option \"%s\" has incorrect type (%s)", name, errstring);
        
        Py_DECREF(errtype);
        Py_DECREF(errvalue);
        Py_XDECREF(errtraceback);
    }
    
    return ret;
}
