#ifndef __OIL_PYTHON_H_INCLUDED__
#define __OIL_PYTHON_H_INCLUDED__

#include <Python.h>
#include "oil.h"

/* the Matrix type instance */
typedef struct {
    PyObject_HEAD
    OILMatrix matrix;
} PyOILMatrix;

/* forward declaration for matrix type object */
extern PyTypeObject PyOILMatrixType;

/* the Image type instance */
typedef struct {
    PyObject_HEAD
    OILImage *im;
} PyOILImage;

/* forward delcaration for image type object */
extern PyTypeObject PyOILImageType;

#endif /* __OIL_PYTHON_H_INCLUDED__ */
