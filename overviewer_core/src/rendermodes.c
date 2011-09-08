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
    &rendermode_night,
    &rendermode_spawn,
    &rendermode_cave,
    &rendermode_mineral,
    NULL
};

PyObject *render_mode_options = NULL;
PyObject *custom_render_modes = NULL;

/* rendermode encapsulation */

/* helper to recursively find options for a given mode */
static inline PyObject *
render_mode_create_options(const char *mode) {
    const char *parent = NULL;
    PyObject *base_options, *ret, *parent_options;
    unsigned int i, found_concrete;
    
    base_options = PyDict_GetItemString(render_mode_options, mode);
    if (base_options) {
        ret = PyDict_Copy(base_options);
    } else {
        ret = PyDict_New();
    }
    
    /* figure out the parent mode name */
    found_concrete = 0;
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, mode) == 0) {
            found_concrete = 1;
            if (render_modes[i]->parent) {
                parent = render_modes[i]->parent->name;
            }
            break;
        }
    }
    
    /* check custom mode info if needed */
    if (found_concrete == 0) {
        PyObject *custom = PyDict_GetItemString(custom_render_modes, mode);
        if (custom) {
            custom = PyDict_GetItemString(custom, "parent");
            if (custom) {
                parent = PyString_AsString(custom);
            }
        }
    }
    
    /* merge parent options, if the parent was found */
    if (parent) {
        parent_options = render_mode_create_options(parent);
        if (parent_options) {
            if (PyDict_Merge(ret, parent_options, 0) == -1) {
                Py_DECREF(ret);
                Py_DECREF(parent_options);
                return NULL;
            }
            Py_DECREF(parent_options);
        }
    }
    
    return ret;
}

/* helper to find the first concrete, C interface for a given mode */
inline static RenderModeInterface *
render_mode_find_interface(const char *mode) {
    PyObject *custom;
    const char *custom_parent;
    unsigned int i;
    
    /* if it is *itself* concrete, we're done */
    for (i = 0; render_modes[i] != NULL; i++) {
        if (strcmp(render_modes[i]->name, mode) == 0)
            return render_modes[i];
    }
    
    /* check for custom modes */
    custom = PyDict_GetItemString(custom_render_modes, mode);
    if (custom == NULL)
        return NULL;
    custom = PyDict_GetItemString(custom, "parent");
    if (custom == NULL)
        return NULL;
    custom_parent = PyString_AsString(custom);
    if (custom_parent == NULL)
        return NULL;
    
    return render_mode_find_interface(custom_parent);
}

RenderMode *render_mode_create(const char *mode, RenderState *state) {
    PyObject *options;
    RenderMode *ret = NULL;
    RenderModeInterface *iface = NULL;
    
    iface = render_mode_find_interface(mode);
    if (iface == NULL)
        return NULL;
    
    options = render_mode_create_options(mode);
    if (options == NULL)
        return NULL;
    
    ret = calloc(1, sizeof(RenderMode));
    if (ret == NULL) {
        Py_DECREF(options);
        return NULL;
    }
    
    ret->mode = calloc(1, iface->data_size);
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

/* bindings for python -- get all the rendermode names */
PyObject *get_render_modes(PyObject *self, PyObject *args) {
    PyObject *modes;
    unsigned int i;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

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
    
    
    while (PyDict_Next(custom_render_modes, &pos, &key, &value)) {
        PyList_Append(modes, key);
    }
    
    return modes;
}

/* helper, get the list of options for a render mode */
static inline PyObject *
get_render_mode_options(const char *rendermode)
{
    PyObject *options;
    unsigned int i, j;
    
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

/* more bindings -- return info for a given rendermode name */
PyObject *get_render_mode_info(PyObject *self, PyObject *args) {
    const char* rendermode;
    PyObject *info;
    unsigned int i;
    PyObject *custom;

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
            
            tmp = PyString_FromString(render_modes[i]->label);
            PyDict_SetItemString(info, "label", tmp);
            Py_DECREF(tmp);
            
            tmp = PyString_FromString(render_modes[i]->description);
            PyDict_SetItemString(info, "description", tmp);
            Py_DECREF(tmp);
            
            tmp = get_render_mode_options(rendermode);
            PyDict_SetItemString(info, "options", tmp);
            Py_DECREF(tmp);
            
            if (render_modes[i]->parent != NULL) {
                tmp = PyString_FromString(render_modes[i]->parent->name);
                PyDict_SetItemString(info, "parent", tmp);
                Py_DECREF(tmp);
            }
            
            return info;
        }
    }
    
    custom = PyDict_GetItemString(custom_render_modes, rendermode);
    if (custom) {
        PyObject *tmp, *copy = PyDict_Copy(custom);
        Py_DECREF(info);
        
        tmp = PyString_FromString(rendermode);
        PyDict_SetItemString(copy, "name", tmp);
        Py_DECREF(tmp);
        
        tmp = PyList_New(0);
        PyDict_SetItemString(copy, "options", tmp);
        Py_DECREF(tmp);
        
        return copy;
    }
    
    Py_DECREF(info);
    return PyErr_Format(PyExc_ValueError, "invalid rendermode: \"%s\"", rendermode);
}

/* bindings -- get list of inherited parents */
PyObject *get_render_mode_inheritance(PyObject *self, PyObject *args) {
    const char *rendermode;
    PyObject *parents;
    unsigned int i;
    RenderModeInterface *iface = NULL;
    PyObject *custom;
    
    if (!PyArg_ParseTuple(args, "s", &rendermode))
        return NULL;
    
    parents = PyList_New(0);
    if (!parents)
        return NULL;
    
    /* take care of the chain of custom modes, if there are any */
    custom = PyDict_GetItemString(custom_render_modes, rendermode);
    while (custom != NULL) {
        PyObject *name = PyString_FromString(rendermode);
        PyList_Append(parents, name);
        Py_DECREF(name);
        
        custom = PyDict_GetItemString(custom, "parent");
        rendermode = PyString_AsString(custom);
        custom = PyDict_GetItem(custom_render_modes, custom);
    }
    
    /* now handle concrete modes */
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
    PyObject *key, *value;
    Py_ssize_t pos = 0;

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
    
    
    while (PyDict_Next(custom_render_modes, &pos, &key, &value)) {
        PyObject *pyparent = PyDict_GetItemString(value, "parent");
        const char *parent = PyString_AsString(pyparent);
        
        if (strcmp(parent, rendermode) == 0) {
            PyList_Append(children, key);
        }
    }
    
    return children;
}

/* helper to decide if a rendermode supports a given option */
static inline int
render_mode_supports_option(RenderModeInterface *iface, const char *name) {
    unsigned int i;
    
    if (iface->options != NULL) {
        for (i = 0; iface->options[i].name != NULL; i++) {
            if (strcmp(iface->options[i].name, name) == 0) {
                return 1;
            }
        }
    }
    
    if (iface->parent != NULL)
        return render_mode_supports_option(iface->parent, name);
    
    return 0;
}

/* python rendermode options bindings */
PyObject *set_render_mode_options(PyObject *self, PyObject *args) {
    const char *rendermode;
    PyObject *opts, *key, *value;
    Py_ssize_t pos = 0;
    RenderModeInterface *iface = NULL;
    if (!PyArg_ParseTuple(args, "sO!", &rendermode, &PyDict_Type, &opts))
        return NULL;
    
    iface = render_mode_find_interface(rendermode);
    
    if (iface == NULL) {
        return PyErr_Format(PyExc_ValueError, "'%s' is not a valid rendermode name", rendermode);
    }
    
    /* check options to make sure they're available */
    while (PyDict_Next(opts, &pos, &key, &value)) {
        const char *name = PyString_AsString(key);
        if (name == NULL)
            return NULL;
        
        if (!render_mode_supports_option(iface, name)) {
            return PyErr_Format(PyExc_ValueError, "'%s' is not a valid option for rendermode '%s'", name, rendermode);
        }
    }
    
    PyDict_SetItemString(render_mode_options, rendermode, opts);
    Py_RETURN_NONE;
}

PyObject *add_custom_render_mode(PyObject *self, PyObject *args) {
    const char *rendermode, *parentmode;
    PyObject *opts, *options, *pyparent;
    if (!PyArg_ParseTuple(args, "sO!", &rendermode, &PyDict_Type, &opts))
        return NULL;

    /* first, make sure the parent is set correctly */
    pyparent = PyDict_GetItemString(opts, "parent");
    if (pyparent == NULL)
        return PyErr_Format(PyExc_ValueError, "'%s' does not have a parent mode", rendermode);
    parentmode = PyString_AsString(pyparent);
    if (parentmode == NULL)
        return PyErr_Format(PyExc_ValueError, "'%s' does not have a valid parent", rendermode);
    
    /* check that parentmode exists */
    if (PyDict_GetItemString(custom_render_modes, parentmode) == NULL) {
        unsigned int parent_valid = 0, i;
        for (i = 0; render_modes[i] != NULL; i++) {
            if (strcmp(render_modes[i]->name, parentmode) == 0) {
                parent_valid = 1;
            }
        }
        
        if (parent_valid == 0)
            return PyErr_Format(PyExc_ValueError, "'%s' parent '%s' is not valid", rendermode, parentmode);
    }
    
    /* remove and handle options seperately, if needed */
    options = PyDict_GetItemString(opts, "options");
    if (options != NULL) {
        PyObject *opts_copy, *set_opts_args;
        
        opts_copy = PyDict_Copy(opts);
        if (opts_copy == NULL)
            return NULL;
        
        PyDict_DelItemString(opts_copy, "options");
        PyDict_SetItemString(custom_render_modes, rendermode, opts_copy);
        Py_DECREF(opts_copy);

        /* call set_render_mode_options */
        set_opts_args = Py_BuildValue("sO", rendermode, options);
        if (set_opts_args == NULL)
            return NULL;
        if (set_render_mode_options(NULL, set_opts_args) == NULL) {
            Py_DECREF(set_opts_args);
            return NULL;
        }
        Py_DECREF(set_opts_args);
    } else {
        PyDict_SetItemString(custom_render_modes, rendermode, opts);
    }
    Py_RETURN_NONE;
}
