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

struct HideRule {
    unsigned short blockid;
    unsigned char has_data;
    unsigned char data;
};

typedef struct {
    struct HideRule* rules;
} RenderPrimitiveHide;

static int
hide_start(void *data, RenderState *state, PyObject *support) {
    PyObject *opt;
    RenderPrimitiveHide* self = (RenderPrimitiveHide *)data;
    self->rules = NULL;
    
    if (!render_mode_parse_option(support, "blocks", "O", &(opt)))
        return 1;
    if (opt && opt != Py_None) {
        Py_ssize_t blocks_size = 0, i;
        
        if (!PyList_Check(opt)) {
            PyErr_SetString(PyExc_TypeError, "'blocks' must be a list");
            return 1;
        }
        
        blocks_size = PyList_GET_SIZE(opt);
        self->rules = calloc(blocks_size + 1, sizeof(struct HideRule));
        if (self->rules == NULL) {
            return 1;
        }

        for (i = 0; i < blocks_size; i++) {
            PyObject *block = PyList_GET_ITEM(opt, i);
            
            if (PyInt_Check(block)) {
                /* format 1: just a block id */
                self->rules[i].blockid = PyInt_AsLong(block);
                self->rules[i].has_data = 0;
            } else if (PyArg_ParseTuple(block, "Hb", &(self->rules[i].blockid), &(self->rules[i].data))) {
                /* format 2: (blockid, data) */
                self->rules[i].has_data = 1;
            } else {
                /* format not recognized */
                free(self->rules);
                self->rules = NULL;
                return 1;
            }
        }
    }
    
    return 0;
}

static void
hide_finish(void *data, RenderState *state) {
    RenderPrimitiveHide *self = (RenderPrimitiveHide *)data;
    
    if (self->rules) {
        free(self->rules);
    }
}

static int
hide_hidden(void *data, RenderState *state, int x, int y, int z) {
    RenderPrimitiveHide *self = (RenderPrimitiveHide *)data;
    unsigned int i;
    unsigned short block;
    
    if (self->rules == NULL)
        return 0;
    
    block = get_data(state, BLOCKS, x, y, z);
    for (i = 0; self->rules[i].blockid != 0; i++) {
        if (block == self->rules[i].blockid) {
            unsigned char data;
            
            if (!(self->rules[i].has_data))
                return 1;
            
            data = get_data(state, DATA, x, y, z);
            if (data == self->rules[i].data)
                return 1;
        }
    }
    
    return 0;
}

RenderPrimitiveInterface primitive_hide = {
    "hide",
    sizeof(RenderPrimitiveHide),
    hide_start,
    hide_finish,
    NULL,
    hide_hidden,
    NULL,
};
