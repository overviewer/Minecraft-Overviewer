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
#include "overlay.h"

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    void* structures;
    int32_t numcolors;
} RenderPrimitiveStructure;

struct Condition {
    int32_t relx, rely, relz;
    mc_block_t block;
    signed short data;
};

struct Color {
    int32_t numconds;
    struct Condition* conditions;
    uint8_t r, g, b, a;
};

static void get_color(void* data,
                      RenderState* state,
                      uint8_t* r,
                      uint8_t* g,
                      uint8_t* b,
                      uint8_t* a) {
    /**
     * Calculate the color at the current position and store the values to r,g,b,a.
     **/
    RenderPrimitiveStructure* self = (RenderPrimitiveStructure*)data;
    int32_t x = state->x, z = state->z, y_max, y, col, cond;
    struct Color* structures = (struct Color*)(self->structures);
    struct Condition* c = NULL;
    bool all = true;
    y_max = state->y + 1;

    /**
     * Check for every color in every y level if all its Conditions are met.
     * If all conditions are met for one y level set r,b,g,a accordingly.
     **/
    // iterate over all the colors
    for (col = 0; col < self->numcolors; col++) {
        // iterate over all y levels
        for (y = state->chunky * -16; y <= y_max; y++) {
            // iterate over all the conditions
            for (cond = 0; cond < structures[col].numconds; cond++) {
                all = true;
                c = (struct Condition*)&structures[col].conditions[cond];
                // check if the condition does apply and break from the conditions loop if not.
                if (!(
                        c->block == get_data(state, BLOCKS, x + c->relx, y + c->rely, z + c->relz) &&
                        (c->data == -1 || (uint32_t)c->data == get_data(state, DATA, x + c->relx, y + c->rely, z + c->relz))
                )) {
                    all = false;
                    break;
                }
            }
            if (all) {
                // set the color
                *r = structures[col].r;
                *g = structures[col].g;
                *b = structures[col].b;
                *a = structures[col].a;
                return;
            }
        }
    }
    return;
}

static bool overlay_structure_start(void* data, RenderState* state, PyObject* support) {
    /**
     * Initializing the search for structures by parsing the arguments and storing them into
     * appropriate structures. If no arguments are passed create and use default values.
     **/
    PyObject* opt;
    RenderPrimitiveStructure* self;

    /* first, chain up */
    bool ret = primitive_overlay.start(data, state, support);
    if (ret != false)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveStructure*)data;

    // opt is a borrowed reference.  do not deref
    // store the structures python object into opt.
    if (!render_mode_parse_option(support, "structures", "O", &(opt)))
        return true;

    /**
     * Check if a sane option was passed.
     **/
    if (opt && opt != Py_None) {
        struct Color* structures = NULL;
        struct Condition* cond = NULL;
        Py_ssize_t structures_size = 0, i, cond_size = 0, n = 0;
        bool cont = true;

        opt = PySequence_Fast(opt, "expected a sequence");
        if (!opt) {
            PyErr_SetString(PyExc_TypeError, "'structures' must be a a sequence");
            return true;
        }

        structures_size = PySequence_Fast_GET_SIZE(opt);
        // Getting space on the heap and do not forget to set self->numcolors.
        structures = self->structures = calloc(structures_size, sizeof(struct Color));
        self->numcolors = structures_size;
        if (structures == NULL) {
            PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
            return true;
        }

        /**
         * Try to parse the definitions of conditions and colors.
         **/
        if (cont) {
            for (i = 0; i < structures_size; i++) {
                PyObject* structure = PyList_GET_ITEM(opt, i);
                // condspy holding the conditions tuple of variable length (python object)
                PyObject* condspy;
                // colorpy holding the 4 tuple with r g b a values of the color
                PyObject* colorpy;

                // getting the condspy and colorpy out of the structures.
                if (!PyArg_ParseTuple(structure, "OO", &condspy, &colorpy)) {
                    // Exception set automatically
                    free(structures);
                    self->structures = NULL;
                    return true;
                }

                structures[i].a = 255;
                // Parse colorpy into a c-struct.
                if (!PyArg_ParseTuple(colorpy, "bbb|b",
                                      &structures[i].r,
                                      &structures[i].g,
                                      &structures[i].b,
                                      &structures[i].a)) {
                    free(structures);
                    self->structures = NULL;
                    return true;
                }

                // Convert condspy to a fast sequence
                condspy = PySequence_Fast(condspy, "Failed to parse conditions");
                if (condspy == NULL) {
                    free(structures);
                    self->structures = NULL;
                    return true;
                }

                // get the number of conditions.
                structures[i].numconds = PySequence_Fast_GET_SIZE(condspy);
                // reserve enough memory for the conditions.
                cond = calloc(structures[i].numconds, sizeof(struct Condition));
                structures[i].conditions = cond;

                if (structures[i].conditions == NULL) {
                    PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
                    free(structures);
                    self->structures = NULL;
                    return true;
                }

                // iterate over all the conditions and read them.
                for (n = 0; n < structures[i].numconds; n++) {
                    PyObject* ccond = PySequence_Fast_GET_ITEM(condspy, n);
                    cond[n].data = -1;     // If no data value is given the data value stays at -1
                    if (!PyArg_ParseTuple(ccond, "iiih|h",
                                          &cond[n].relx,
                                          &cond[n].rely,
                                          &cond[n].relz,
                                          &cond[n].block,
                                          &cond[n].data)) {
                        int32_t x = 0;
                        for (x = 0; x < structures_size; x++) {
                            free(structures[x].conditions);
                        }
                        free(structures);
                        self->structures = NULL;
                        return true;
                    }
                }
            }
        }
    }

    /* setup custom color */
    self->parent.get_color = get_color;

    return false;
}

static void overlay_structure_finish(void* data, RenderState* state) {
    /* first free all *our* stuff */
    RenderPrimitiveStructure* self = (RenderPrimitiveStructure*)data;
    int32_t i = 0;

    if (self->structures) {
        // freeing the nested structure
        struct Color* m = self->structures;
        for (i = 0; i < self->numcolors; i++) {
            if (m[i].conditions)
                free(m[i].conditions);
        }
    }

    if (self->structures) {
        free(self->structures);
        self->structures = NULL;
    }

    /* now, chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_structure = {
    "overlay-structure",
    sizeof(RenderPrimitiveStructure),
    overlay_structure_start,
    overlay_structure_finish,
    NULL,
    NULL,
    overlay_draw,
};
