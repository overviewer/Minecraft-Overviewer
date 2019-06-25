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

#include "../mc_id.h"
#include "../overviewer.h"

struct HideRule {
    mc_block_t blockid;
    bool has_data;
    uint8_t data;
};

typedef struct {
    struct HideRule* rules;
} RenderPrimitiveHide;

static bool
hide_start(void* data, RenderState* state, PyObject* support) {
    PyObject* opt;
    RenderPrimitiveHide* self = (RenderPrimitiveHide*)data;
    self->rules = NULL;

    if (!render_mode_parse_option(support, "blocks", "O", &(opt)))
        return true;
    if (opt && opt != Py_None) {
        Py_ssize_t blocks_size = 0, i;

        if (!PyList_Check(opt)) {
            PyErr_SetString(PyExc_TypeError, "'blocks' must be a list");
            return true;
        }

        blocks_size = PyList_GET_SIZE(opt);
        self->rules = calloc(blocks_size + 1, sizeof(struct HideRule));
        if (self->rules == NULL) {
            return true;
        }

        for (i = 0; i < blocks_size; i++) {
            PyObject* block = PyList_GET_ITEM(opt, i);

            if (PyLong_Check(block)) {
                /* format 1: just a block id */
                self->rules[i].blockid = PyLong_AsLong(block);
                self->rules[i].has_data = false;
            } else if (PyArg_ParseTuple(block, "Hb", &(self->rules[i].blockid), &(self->rules[i].data))) {
                /* format 2: (blockid, data) */
                self->rules[i].has_data = true;
            } else {
                /* format not recognized */
                free(self->rules);
                self->rules = NULL;
                return true;
            }
        }
    }

    return false;
}

static void
hide_finish(void* data, RenderState* state) {
    RenderPrimitiveHide* self = (RenderPrimitiveHide*)data;

    if (self->rules) {
        free(self->rules);
    }
}

static bool
hide_hidden(void* data, RenderState* state, int32_t x, int32_t y, int32_t z) {
    RenderPrimitiveHide* self = (RenderPrimitiveHide*)data;
    uint32_t i;
    mc_block_t block;

    if (self->rules == NULL)
        return false;

    block = get_data(state, BLOCKS, x, y, z);
    for (i = 0; self->rules[i].blockid != block_air; i++) {
        if (block == self->rules[i].blockid) {
            uint8_t data;

            if (!(self->rules[i].has_data))
                return true;

            data = get_data(state, DATA, x, y, z);
            if (data == self->rules[i].data)
                return true;
        }
    }

    return false;
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
