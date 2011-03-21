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
                    PyObject *img, int imgx, int imgy)
{
    // returns new references
    PyObject* light_tup = PyObject_CallMethod(chunk, "get_lighting_coefficient", "iii", x, y, z);
    PyObject *black_coeff_py = PySequence_GetItem(light_tup, 0);
    double black_coeff = PyFloat_AsDouble(black_coeff_py);
    Py_DECREF(black_coeff_py);
    
    PyObject *face_occlude_py = PySequence_GetItem(light_tup, 1);
    int face_occlude = PyInt_AsLong(face_occlude_py);
    Py_DECREF(face_occlude_py);
    
    
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

typedef struct {
    int lighting;
    PyObject *black_color, *facemasks_py;
    PyObject *facemasks[3];
} RenderModeNormal;

static int
rendermode_normal_start(void *data, RenderState *state) {
    RenderModeNormal* self = (RenderModeNormal *)data;
    
    PyObject *quadtree = PyObject_GetAttrString(state->self, "quadtree");
    PyObject *lighting_py = PyObject_GetAttrString(quadtree, "lighting");
    self->lighting = PyObject_IsTrue(lighting_py);
    Py_DECREF(lighting_py);
    Py_DECREF(quadtree);
    
    self->black_color = PyObject_GetAttrString(state->chunk, "black_color");
    self->facemasks_py = PyObject_GetAttrString(state->chunk, "facemasks");
    // borrowed references, don't need to be decref'd
    self->facemasks[0] = PyTuple_GetItem(self->facemasks_py, 0);
    self->facemasks[1] = PyTuple_GetItem(self->facemasks_py, 1);
    self->facemasks[2] = PyTuple_GetItem(self->facemasks_py, 2);
    
    return 0;
}

static void
rendermode_normal_finish(void *data, RenderState *state) {
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    Py_DECREF(self->black_color);
    Py_DECREF(self->facemasks_py);
}

static int
rendermode_normal_occluded(void *data, RenderState *state) {
    int x = state->x, y = state->y, z = state->z;
    
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }

    return 0;
}

static void
rendermode_normal_draw(void *data, RenderState *state, PyObject *src, PyObject *mask)
{
    RenderModeNormal* self = (RenderModeNormal *)data;
    
    PyObject *chunk = state->self;
    int x = state->x, y = state->y, z = state->z;
    PyObject **facemasks = self->facemasks;
    PyObject *black_color = self->black_color, *img = state->img;
    int imgx = state->imgx, imgy = state->imgy;
    
    alpha_over(img, src, mask, imgx, imgy, 0, 0);
    
    if (self->lighting) {
        // FIXME whole-block shading for transparent blocks
        do_shading_for_face(chunk, x, y, z+1, facemasks[0], black_color,
                            img, imgx, imgy);
        do_shading_for_face(chunk, x-1, y, z, facemasks[1], black_color,
                            img, imgx, imgy);
        do_shading_for_face(chunk, x, y+1, z, facemasks[2], black_color,
                            img, imgx, imgy);
    }

}

RenderModeInterface rendermode_normal = {
    sizeof(RenderModeNormal),
    rendermode_normal_start,
    rendermode_normal_finish,
    rendermode_normal_occluded,
    rendermode_normal_draw,
};
