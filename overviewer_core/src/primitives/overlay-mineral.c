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
    void *minerals;
    int numcolors;
} RenderPrimitiveMineral;

struct Condition{
    int relx, rely, relz;
    unsigned char block;
};

struct Color {
    int numconds;
    struct Condition *conditions;
    unsigned char r, g, b, a;
};

static struct Color default_colors[8] = { 
    {1,NULL, 31 , 153, 9  , 255}, /* Mossy Stone  */
    {1,NULL, 32 , 230, 220, 255}, /* Diamond Ore  */
    {1,NULL, 0  , 23 , 176, 255}, /* Lapis Lazuli */
    {1,NULL, 255, 234, 0  , 255}, /* Gold Ore     */
    {1,NULL, 204, 204, 204, 255}, /* Iron Ore     */
    {1,NULL, 186, 0  , 0  , 255}, /* Redstone     */
    {1,NULL, 186, 0  , 0  , 255}, /* Lit Redstone */
    {1,NULL, 54 , 54 , 54 , 255}  /* Coal Ore     */
};

static inline void setv(int i, int m){
    /**
     * allocate and set default values into the cond list.
     **/
    default_colors[i].conditions = calloc(1, sizeof(struct Condition));
    // verify that memory allocations succeeded
    if (default_colors[i].conditions == NULL){
        int n = 0;
        for (n = 0; n < i; i++){
            free(default_colors[n].conditions);
        }
        PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
        return 1;
    }
    // set the blockid to m
    default_colors[i].conditions[0].block = m;
}

static inline void init_default() {
    /**
     * Set default values for all the default minerals
     **/
    setv(0, 48); /* Mossy Stone  */
    setv(1, 56); /* Diamond Ore  */
    setv(2, 21); /* Lapis Lazuli */
    setv(3, 14); /* Gold Ore     */
    setv(4, 15); /* Iron Ore     */
    setv(5, 73); /* Redstone     */
    setv(6, 74); /* Lit Redstone */
    setv(7, 16); /* Coal Ore     */
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
    RenderPrimitiveMineral *self = (RenderPrimitiveMineral *)data;
    int x = state->x, z = state->z, y_max, y, col, cond;
    struct Color *minerals = (struct Color *)(self->minerals);
    struct Condition * c = NULL;
    bool all = true;
    y_max = state->y + 1;

    /**
     * Check for every color in every y level if all its Conditions are met.
     * If all conditions are met for one y level set r,b,g,a accordingly.
     **/
    // iterate over all the colors
    for ( col = 0; col < self->numcolors; col++) {
        // iterate over all y levels
        for (y = state->chunky * -16; y <= y_max; y++) {
            // iterate over all the conditions
            for (cond = 0; cond < minerals[col].numconds; cond++) {
                all = true;
                c = (struct Condition *)&minerals[col].conditions[cond];
                // check if the condition does apply and break from the conditions loop if not.
                if(!(c->block == get_data(state, BLOCKS, x+c->relx, y+c->rely, z+c->relz))) {
                    all = false;
                    break;
                }
            }
            if (all){
                // Add the color to the overlay color (this can result in mixed colors).
                *r = minerals[col].r;
                *g = minerals[col].g;
                *b = minerals[col].b;
                *a = minerals[col].a;
                return;
            }
        }
    }
    return;
}

static int
overlay_mineral_start(void *data, RenderState *state, PyObject *support) {
    /**
     * Initializing the search for minerals by parsing the arguments and storing them into
     * appropriate structures. If no arguments are passed create and use default values.
     **/
    PyObject *opt;
    RenderPrimitiveMineral* self;

    /* first, chain up */
    int ret = primitive_overlay.start(data, state, support);
    if (ret != 0)
        return ret;
    
    /* now do custom initializations */
    self = (RenderPrimitiveMineral *)data;
    
    // opt is a borrowed reference.  do not deref
    // store the minerals python object into opt.
    if (!render_mode_parse_option(support, "minerals", "O", &(opt)))
        return 1;

    /**
     * Check if a sane option was passed. If so parse it else setup default values.
     **/
    if (opt && opt != Py_None) {
        struct Color *minerals = NULL;
        struct Condition *cond = NULL;
        Py_ssize_t minerals_size = 0, i, cond_size = 0, n = 0;
        bool cont = true;

        opt = PySequence_Fast(opt, "expected a sequence");
        if (!opt) {
            PyErr_SetString(PyExc_TypeError, "'minerals' must be a a sequence");
            return 1;
        }
        
        minerals_size = PySequence_Fast_GET_SIZE(opt);
        // Getting space on the heap and do not forget to set self->numcolors.
        minerals = self->minerals = calloc(minerals_size, sizeof(struct Color));
        self->numcolors = minerals_size;
        if (minerals == NULL) {
            PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
            return 1;
        }

        /**
         * Try to parse the old style mineral definitions for compatibility.
         * It is not allowed to mix old style and new style conditions.
         **/
        for (i = 0; i < minerals_size; i++) {
            PyObject *mineral = PyList_GET_ITEM(opt, i);
            minerals[i].numconds = 1;
            cond = calloc(minerals[i].numconds, sizeof(struct Condition));
            minerals[i].conditions = cond;

            if((cond == NULL) || !PyArg_ParseTuple(mineral, "b(bbb)",
                                                         &cond->block,
                                                         &minerals[i].r,
                                                         &minerals[i].g,
                                                         &minerals[i].b)) {
                // If some conditions succeeded but now failed. Go into error state.
                if( i > 0) {
                    int x = 0;
                    for(x = 0; x < minerals_size; x++){
                        free(minerals[x].conditions);
                    }
                    free(minerals);
                    return 1;
                }
                // If parsing failed continue with new style parsing.
                else {   
                    free(cond);
                    minerals[i].conditions = NULL;
                    PyErr_Clear();
                    cont = true;
                    break;
                }
            }
            // If Parsing succeeded store the values.
            else {
                cont = false;
                cond->relx = 0;
                cond->rely = 0;
                cond->relz = 0;
                minerals[i].a = 255;
            }
        }

        /**
         * Try to parse the new style definitions of colors.
         **/
        if (cont) {
            for (i = 0; i < minerals_size; i++) {
                PyObject *mineral = PyList_GET_ITEM(opt, i);
                // condspy holding the conditions tuple of variable length (python object)
                PyObject *condspy;
                // colorpy holding the 4 tuple with r g b a values of the color
                PyObject *colorpy;

                // getting the condspy and colorpy out of the minerals.
                if (!PyArg_ParseTuple(mineral, "OO", &condspy, &colorpy)) {
                    // Exception set automatically
                    free(minerals);
                    self->minerals = NULL;
                    return 1;
                }

                // Parse colorpy into a c-struct.
                if (!PyArg_ParseTuple( colorpy, "bbbb",
                                       &minerals[i].r,
                                       &minerals[i].g,
                                       &minerals[i].b,
                                       &minerals[i].a)) {
                    free(minerals);
                    self->minerals = NULL;
                    return 1;
                }

                // Convert condspy to a fast sequence
                condspy = PySequence_Fast(condspy, "Failed to parse conditions");
                if(condspy == NULL) {
                    free(minerals);
                    self->minerals = NULL;
                    return 1;
                }

                // get the number of conditions.
                minerals[i].numconds = PySequence_Fast_GET_SIZE(condspy);
                // reserve enough memory for the conditions.
                cond = calloc(minerals[i].numconds, sizeof(struct Condition));
                minerals[i].conditions = cond;

                if (minerals[i].conditions == NULL) {
                    PyErr_SetString(PyExc_MemoryError, "failed to allocate memory");
                    free(minerals);
                    self->minerals = NULL;
                    return 1;
                }

                // iterate over all the conditions and read them.
                for (n = 0; n < minerals[i].numconds; n++) {
                    PyObject *ccond = PySequence_Fast_GET_ITEM(condspy, n);
                    if(!PyArg_ParseTuple( ccond, "iiib",
                                          &cond[n].relx,
                                          &cond[n].rely,
                                          &cond[n].relz,
                                          &cond[n].block)){
                        int x = 0;
                        for(x = 0; x < minerals_size; x++){
                            free(minerals[x].conditions);
                        }
                        free(minerals);
                        self->minerals = NULL;
                        return 1;
                    }
                }
            }
        }
    }
    /**
     * No minerals specified so use some sane defaults.
     **/
    else {
        self->minerals = default_colors;
        init_default();
        self->numcolors = 8;
    }
    
    /* setup custom color */
    self->parent.get_color = get_color;
    
    return 0;
}

static void
overlay_mineral_finish(void *data, RenderState *state) {
    /* first free all *our* stuff */
    RenderPrimitiveMineral* self = (RenderPrimitiveMineral *)data;
    int i = 0;

    if(self->minerals) {
        struct Color * m = self->minerals;
        for(i = 0; i < self->numcolors; i++){
            free(m[i].conditions);
        }
    }

    if (self->minerals && self->minerals != default_colors) {
        free(self->minerals);
        self->minerals = NULL;
    }
    
    /* now, chain up */
    primitive_overlay.finish(data, state);
}

RenderPrimitiveInterface primitive_overlay_mineral = {
    "overlay-mineral",
    sizeof(RenderPrimitiveMineral),
    overlay_mineral_start,
    overlay_mineral_finish,
    NULL,
    NULL,
    overlay_draw,
};
