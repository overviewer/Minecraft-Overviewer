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

/* shades the drawn block with the given facemask/black_color, based on the
   lighting results from (x, y, z) */
static inline void
do_shading_for_face(PyObject *chunk, int x, int y, int z, PyObject *facemask, PyObject *black_color,
                    PyObject *img, int imgx, int imgy) {
    // returns new references
    PyObject* light_tup = PyObject_CallMethod(chunk, "get_lighting_coefficient", "iii", x, y, z);
    PyObject *black_coeff_py = PySequence_GetItem(light_tup, 0);
    double black_coeff = PyFloat_AsDouble(black_coeff_py);
    Py_DECREF(black_coeff_py);
    
    PyObject *face_occlude_py = PySequence_GetItem(light_tup, 1);
    int face_occlude = PyInt_AsLong(face_occlude_py);
    Py_DECREF(face_occlude_py);

    Py_DECREF(light_tup);
    
    
    if (!face_occlude) {
        //#composite.alpha_over(img, over_color, (imgx, imgy), ImageEnhance.Brightness(facemasks[0]).enhance(black_coeff))
        
        PyObject *mask = PyObject_CallMethod(facemask, "copy", NULL); // new ref
        //printf("black_coeff: %f\n", black_coeff);
        brightness(mask, black_coeff);
        //printf("done with brightness\n");
        alpha_over(img, black_color, mask, imgx, imgy, 0, 0);
        //printf("done with alpha_over\n");
        Py_DECREF(mask);
        
    }
}

static int
rendermode_lighting_start(void *data, RenderState *state) {
    /* first, chain up */
    int ret = rendermode_normal.start(data, state);
    if (ret != 0)
        return ret;
    
    RenderModeLighting* self = (RenderModeLighting *)data;
    
    self->black_color = PyObject_GetAttrString(state->chunk, "black_color");
    self->facemasks_py = PyObject_GetAttrString(state->chunk, "facemasks");
    // borrowed references, don't need to be decref'd
    self->facemasks[0] = PyTuple_GetItem(self->facemasks_py, 0);
    self->facemasks[1] = PyTuple_GetItem(self->facemasks_py, 1);
    self->facemasks[2] = PyTuple_GetItem(self->facemasks_py, 2);
    
    return 0;
}

static void
rendermode_lighting_finish(void *data, RenderState *state) {
    RenderModeLighting *self = (RenderModeLighting *)data;
    
    Py_DECREF(self->black_color);
    Py_DECREF(self->facemasks_py);
    
    /* now chain up */
    rendermode_normal.finish(data, state);
}

static int
rendermode_lighting_occluded(void *data, RenderState *state) {
    /* no special occlusion here */
    return rendermode_normal.occluded(data, state);
}

static void
rendermode_lighting_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    /* first, chain up */
    rendermode_normal.draw(data, state, src, mask);
    
    RenderModeLighting* self = (RenderModeLighting *)data;
    
    PyObject *chunk = state->self;
    int x = state->x, y = state->y, z = state->z;
    PyObject **facemasks = self->facemasks;
    PyObject *black_color = self->black_color, *img = state->img;
    int imgx = state->imgx, imgy = state->imgy;
    
    // FIXME whole-block shading for transparent blocks
    do_shading_for_face(chunk, x, y, z+1, facemasks[0], black_color,
                        img, imgx, imgy);
    do_shading_for_face(chunk, x-1, y, z, facemasks[1], black_color,
                        img, imgx, imgy);
    do_shading_for_face(chunk, x, y+1, z, facemasks[2], black_color,
                        img, imgx, imgy);
}

RenderModeInterface rendermode_lighting = {
    sizeof(RenderModeLighting),
    rendermode_lighting_start,
    rendermode_lighting_finish,
    rendermode_lighting_occluded,
    rendermode_lighting_draw,
};
