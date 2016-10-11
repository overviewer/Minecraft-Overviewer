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

#include "overlay.h"

typedef enum { false, true } bool;

typedef struct {
    /* inherits from overlay */
    RenderPrimitiveOverlay parent;
    void *bounds;
    int numcolors;
} RenderPrimitiveStructure;

struct Condition{
    int minx, minz, maxx, maxz;
};


struct Color {
    int numconds;
    struct Condition *conditions;
    unsigned char r, g, b, a;
};

/*
* Determines if point x,z is inside bounds minx, minz, maxx, maxz
*/
static int is_inside(int x, int z, int minx, int minz, int maxx, int maxz) {
	return (x >= minx && x <= maxx && z >= minz && z <= maxz);
}

static void get_color(void *data,
                      RenderState *state,
                      unsigned char *r,
                      unsigned char *g,
                      unsigned char *b,
                      unsigned char *a) {
    /**
     * Calculate the color at the current position and store the values to r,g,b,a.
     **/
    RenderPrimitiveStructure *self = (RenderPrimitiveStructure *)data;
    int x = state->x, z = state->z, y_max, y, col, cond;
    struct Color *bounds = (struct Color *)(self->bounds);
    struct Condition * c = NULL;
    bool any = true;
    y_max = state->y + 1;

    /**
     * Check for every color in the current point is in the given bounds,
	 * and color appropriately
     **/
    // iterate over all the colors
    for ( col = 0; col < self->numcolors; col++) {
		any = false;
        // iterate over all conditions
        for (cond = 0; cond < bounds[col].numconds; cond++) {
            c = (struct Condition *)&bounds[col].conditions[cond];
            // check current point is in the condition
            if(is_inside(x, z, c->minx, c->minz, c->maxx, c->maxz)) {
				any = true;
            }
        }

		//if current point is in any of the conditions, draw it this color
		if (any) {
			// set the color
			*r = bounds[col].r;
			*g = bounds[col].g;
			*b = bounds[col].b;
			*a = bounds[col].a;
			return;
		}      
    }
    return;
}

static int overlay_bounds_start(void *data, RenderState *state, PyObject *support) {
    /**
     * Initializing the search for bounds by parsing the arguments and storing them into
     * appropriate bounds. If no arguments are passed create and use default values.
     **/
    PyObject *opt;
    RenderPrimitiveStructure* self;

    /* first, chain up */
    int ret = primitive_overlay.start(data, state, support);
    if (ret != 0)
        return ret;

    /* now do custom initializations */
    self = (RenderPrimitiveStructure *)data;

    // opt is a borrowed reference.  do not deref
    // store the bounds python object into opt.
    if (!render_mode_parse_option(support, "bounds", "O", &(opt)))
        return 1;

    /**
     * Check if a sane option was passed.
     **/
    if (opt && opt != Py_None) {
        struct Color *bounds = NULL;
        struct Condition *cond = NULL;
        Py_ssize_t bounds_size = 0, i, cond_size = 0, n = 0;
        bool cont = true;

        opt = PySequence_Fast(opt, "expected a sequence");
        if (!opt) {
            PyErr_SetString(PyExc_TypeError, "'bounds' must be a a sequence");
            return 1;
        }

        bounds_size = PySequence_Fast_GET_SIZE(opt);
        // Getting space on the heap and do not forget to set self->numcolors.
        bounds = self->bounds = calloc(bounds_size, sizeof(struct Color));
        self->numcolors = bounds_size;
        if (bounds == NULL) {
            PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
            return 1;
        }

        /**
         * Try to parse the definitions of conditions and colors.
         **/
        if (cont) {
            for (i = 0; i < bounds_size; i++) {
                PyObject *structure = PyList_GET_ITEM(opt, i);
                // condspy holding the conditions tuple of variable length (python object)
                PyObject *condspy;
                // colorpy holding the 4 tuple with r g b a values of the color
                PyObject *colorpy;

                // getting the condspy and colorpy out of the bounds.
                if (!PyArg_ParseTuple(structure, "OO", &condspy, &colorpy)) {
                    // Exception set automatically
                    free(bounds);
                    self->bounds = NULL;
                    return 1;
                }

                // Parse colorpy into a c-struct.
                if (!PyArg_ParseTuple( colorpy, "bbbb",
                                       &bounds[i].r,
                                       &bounds[i].g,
                                       &bounds[i].b,
                                       &bounds[i].a)) {
                    free(bounds);
                    self->bounds = NULL;
                    return 1;
                }

                // Convert condspy to a fast sequence
                condspy = PySequence_Fast(condspy, "Failed to parse conditions");
                if(condspy == NULL) {
                    free(bounds);
                    self->bounds = NULL;
                    return 1;
                }

                // get the number of conditions.
                bounds[i].numconds = PySequence_Fast_GET_SIZE(condspy);
                // reserve enough memory for the conditions.
                cond = calloc(bounds[i].numconds, sizeof(struct Condition));
                bounds[i].conditions = cond;

                if (bounds[i].conditions == NULL) {
                    PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
                    free(bounds);
                    self->bounds = NULL;
                    return 1;
                }

                // iterate over all the conditions and read them.
                for (n = 0; n < bounds[i].numconds; n++) {
                    PyObject *ccond = PySequence_Fast_GET_ITEM(condspy, n);
                    if(!PyArg_ParseTuple( ccond, "iiii",
                                          &cond[n].minx,
                                          &cond[n].minz,
                                          &cond[n].maxx,
                                          &cond[n].maxz)){
                        int x = 0;
                        for(x = 0; x < bounds_size; x++){
                            free(bounds[x].conditions);
                        }
                        free(bounds);
                        self->bounds = NULL;
                        return 1;
                    }
                }
            }
        }
    }

    /* setup custom color */
    self->parent.get_color = get_color;

    return 0;
}

static void overlay_bounds_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderPrimitiveStructure* self = (RenderPrimitiveStructure *)data;
    int i = 0;

    if(self->bounds) {
        // freeing the nested structure
        struct Color * m = self->bounds;
        for(i = 0; i < self->numcolors; i++){
            if(m[i].conditions)
                free(m[i].conditions);
        }
    }

    if (self->bounds) {
        free(self->bounds);
        self->bounds = NULL;
    }

    /* now, chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_bounds = {
    "overlay-bounds",
    sizeof(RenderPrimitiveStructure),
    overlay_bounds_start,
    overlay_bounds_finish,
    NULL,
    NULL,
    overlay_draw,
};

