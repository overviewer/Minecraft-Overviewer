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

/* list of all render modes, ending in NULL
   all of these will be available to the user, so DON'T include modes
   that are only useful as a base for other modes. */
static RenderModeInterface *render_modes[] = {
    &rendermode_normal,
    &rendermode_lighting,
    &rendermode_night,
    &rendermode_spawn,
    &rendermode_cave,
    NULL
};

static PyObject *render_mode_options = NULL;

/* rendermode encapsulation */

/* helper to recursively find options for a given mode */
static inline PyObject *
render_mode_create_options(RenderModeInterface *iface) {
    PyObject *base_options, *ret, *parent_options;
    if (render_mode_options == NULL)
        return PyDict_New();
    
    base_options = PyDict_GetItemString(render_mode_options, iface->name);
    if (base_options) {
        ret = PyDict_Copy(base_options);
    } else {
        ret = PyDict_New();
    }
    
    if (iface->parent) {
        parent_options = render_mode_create_options(iface->parent);
        if (parent_options) {
            if (PyDict_Merge(ret, parent_options, 0) == -1) {
                Py_DECREF(parent_options);
                return NULL;
            }
            Py_DECREF(parent_options);
        }
    }
    
    return ret;
}

RenderMode *render_mode_create(const char *mode, RenderState *state) {
    unsigned int i;
    PyObject *options;
    RenderMode *ret = NULL;
    RenderModeInterface *iface = NULL;
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, mode) == 0) {
            iface = render_modes[i];
            break;
        }
    }
    
    if (iface == NULL)
        return NULL;
    
    options = render_mode_create_options(iface);
    if (options == NULL)
        return NULL;
    
    ret = malloc(sizeof(RenderMode));
    if (ret == NULL) {
        Py_DECREF(options);
        return NULL;
    }
    
    ret->mode = malloc(iface->data_size);
    if (ret->mode == NULL) {
        Py_DECREF(options);
        free(ret);
        return NULL;
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

int render_mode_occluded(RenderMode *self) {
    return self->iface->occluded(self->mode, self->state);
}

void render_mode_draw(RenderMode *self, PyObject *img, PyObject *mask, PyObject *mask_light) {
    self->iface->draw(self->mode, self->state, img, mask, mask_light);
}

/* bindings for python -- get all the rendermode names */
PyObject *get_render_modes(PyObject *self, PyObject *args) {
    PyObject *modes;
    unsigned int i;
    if (!PyArg_ParseTuple(args, ""))
        return NULL;
    
    modes = PyList_New(0);
    if (modes == NULL)
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        PyObject *name = PyString_FromString(render_modes[i]->name);
        PyList_Append(modes, name);
        Py_DECREF(name);
    }
    
    return modes;
}

/* more bindings -- return info for a given rendermode name */
PyObject *get_render_mode_info(PyObject *self, PyObject *args) {
    const char* rendermode;
    PyObject *info;
    unsigned int i;
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    info = PyDict_New();
    if (info == NULL)
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, rendermode) == 0) {
            PyObject *tmp;
            
            tmp = PyString_FromString(render_modes[i]->name);
            PyDict_SetItemString(info, "name", tmp);
            Py_DECREF(tmp);
            
            tmp = PyString_FromString(render_modes[i]->description);
            PyDict_SetItemString(info, "description", tmp);
            Py_DECREF(tmp);
            
            return info;
        }
    }
    
    Py_DECREF(info);
    return PyErr_Format(PyExc_ValueError, "invalid rendermode: \"%s\"", rendermode);
}

/* bindings -- get parent's name */
PyObject *get_render_mode_parent(PyObject *self, PyObject *args) {
    const char *rendermode;
    unsigned int i;
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, rendermode) == 0) {
            if (render_modes[i]->parent) {
                /* has parent */
                return PyString_FromString(render_modes[i]->parent->name);
            } else {
                /* no parent */
                Py_RETURN_NONE;
            }
        }
    }
    
    return PyErr_Format(PyExc_ValueError, "invalid rendermode: \"%s\"", rendermode);
}

/* bindings -- get list of inherited parents */
PyObject *get_render_mode_inheritance(PyObject *self, PyObject *args) {
    const char *rendermode;
    PyObject *parents;
    unsigned int i;
    RenderModeInterface *iface = NULL;
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    parents = PyList_New(0);
    if (!parents)
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, rendermode) == 0) {
            iface = render_modes[i];
            break;
        }
    }
    
    if (!iface) {
        Py_DECREF(parents);
        return PyErr_Format(PyExc_ValueError, "invalid rendermode: \"%s\"", rendermode);
    }
    
    while (iface) {
        PyObject *name = PyString_FromString(iface->name);
        PyList_Append(parents, name);
        Py_DECREF(name);
        
        iface = iface->parent;
    }
    
    PyList_Reverse(parents);
    return parents;
}

/* bindings -- get list of (direct) children */
PyObject *get_render_mode_children(PyObject *self, PyObject *args) {
    const char *rendermode;
    PyObject *children;
    unsigned int i;
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    children = PyList_New(0);
    if (!children)
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (render_modes[i]->parent && strcmp(render_modes[i]->parent->name, rendermode) == 0) {
            PyObject *child_name = PyString_FromString(render_modes[i]->name);
            PyList_Append(children, child_name);
            Py_DECREF(child_name);
        }
    }
    
    return children;
}

/* bindings -- get list of options */
PyObject *get_render_mode_options(PyObject *self, PyObject *args)
{
    const char *rendermode;
    PyObject *options;
    unsigned int i, j;
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    options = PyList_New(0);
    if (!options)
        return NULL;
    
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, rendermode) == 0) {
            if (render_modes[i]->options == NULL)
                break;
            
            for (j = 0; render_modes[i]->options[j].name != NULL; j++) {
                RenderModeOption opt = render_modes[i]->options[j];
                PyObject *name = PyString_FromString(opt.name);
                PyObject *description = PyString_FromString(opt.description);
                PyObject *option = PyDict_New();
                if (!name || !description || !option) {
                    Py_XDECREF(name);
                    Py_XDECREF(description);
                    Py_XDECREF(option);
                    Py_DECREF(options);
                    return NULL;
                }
                
                PyDict_SetItemString(option, "name", name);
                PyDict_SetItemString(option, "description", description);
                PyList_Append(options, option);
                Py_DECREF(name);
                Py_DECREF(description);
                Py_DECREF(option);
            }
            break;
        }
    }
    
    return options;
}

/* python rendermode options bindings */
PyObject *set_render_mode_options(PyObject *self, PyObject *args) {
    const char *rendermode;
    PyObject *opts;
    if (!PyArg_ParseTuple(args, "sO!", &rendermode, &PyDict_Type, &opts))
        return NULL;
    
    /* check options here */
    
    if (render_mode_options == NULL)
        render_mode_options = PyDict_New();
    
    PyDict_SetItemString(render_mode_options, rendermode, opts);
    Py_RETURN_NONE;
}
