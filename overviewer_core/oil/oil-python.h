#ifndef __OIL_PYTHON_H_INCLUDED__
#define __OIL_PYTHON_H_INCLUDED__

#include <Python.h>
#include "oil.h"

/* helper to get an OIL module type */
static PyTypeObject *py_oil_get_type(const char *typ) {
    PyTypeObject *typeobj;
    PyObject *oilmod = PyImport_ImportModule("overviewer_core.oil");
    if (!oilmod)
        return NULL;
    typeobj = (PyTypeObject *)PyObject_GetAttrString(oilmod, typ);
    Py_DECREF(oilmod);
    return typeobj;
}

/* the Matrix type instance */
typedef struct {
    PyObject_HEAD
    OILMatrix matrix;
} PyOILMatrix;

/* helper to get the matrix type object */
static PyTypeObject *py_oil_get_matrix_type(void) {
    static PyTypeObject *matrix_type = NULL;
    if (matrix_type) {
        Py_INCREF(matrix_type);
        return matrix_type;
    }
    return (matrix_type = py_oil_get_type("Matrix"));
}

/* the Image type instance */
typedef struct {
    PyObject_HEAD
    OILImage *im;
} PyOILImage;

/* helper to get the image type object */
static PyTypeObject *py_oil_get_image_type(void) {
    static PyTypeObject *image_type = NULL;
    if (image_type) {
        Py_INCREF(image_type);
        return image_type;
    }
    return (image_type = py_oil_get_type("Image"));
}

#endif /* __OIL_PYTHON_H_INCLUDED__ */
