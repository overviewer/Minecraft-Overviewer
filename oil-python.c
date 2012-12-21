#include "oil-python.h"

/*
 * file IO implementation for python file-like objects
 */

static size_t oil_python_read(void *file, void *data, size_t length) {
    void *buffer = NULL;
    Py_ssize_t buflength = 0;
    PyObject *result = PyObject_CallMethod(file, "read", "i", length);
    
    if (!result)
        return 0;
    if (!PyString_Check(result)) {
        Py_DECREF(result);
        return 0;
    }
    
    PyString_AsStringAndSize(result, (char **)&buffer, &buflength);
    if (!buffer || !buflength || buflength > length) {
        Py_DECREF(result);
        return 0;
    }
    
    memcpy(data, buffer, buflength);
    Py_DECREF(result);
    return buflength;
}

static size_t oil_python_write(void *file, void *data, size_t length) {
    PyObject *result = PyObject_CallMethod(file, "write", "s#", data, length);
    if (!result)
        return 0;
    Py_DECREF(result);
    return length;
}

static void oil_python_flush(void *file) {
    PyObject *result = PyObject_CallMethod(file, "flush", "");
    if (result) {
        Py_DECREF(result);
    }
}

/* init and dealloc for the Matrix type */

static int PyOILMatrix_set_data(PyOILMatrix *self, PyObject *arg, void *unused);

static PyOILMatrix *PyOILMatrix_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs) {
    PyObject *data = NULL;
    PyOILMatrix *self;
    
    /* for now, do not allow initialization */
    if (!PyArg_ParseTuple(args, "|O", &data)) {
        return NULL;
    }
    
    self = (PyOILMatrix *)(subtype->tp_alloc(subtype, 0));
    if (!self)
        return NULL;
    
    if (data) {
        if (PyOILMatrix_set_data(self, data, NULL)) {
            Py_DECREF(self);
            return NULL;
        }
    } else {
        oil_matrix_set_identity(&(self->matrix));
    }
    
    return self;
}

static void PyOILMatrix_dealloc(PyOILMatrix *self) {
    self->ob_type->tp_free((PyObject *)self);
}

/* methods for Matrix type */

static PyObject *PyOILMatrix_get_data(PyOILMatrix *self, PyObject *args) {
    int i;
    PyObject *tuples[4];
    PyObject *full_tuple;
    
    /* args == NULL means we're called as an attribute-getter */
    if (args != NULL) {
        if (!PyArg_ParseTuple(args, "")) {
            return NULL;
        }
    }

    for (i = 0; i < 4; i++) {
        PyObject *t0 = PyFloat_FromDouble(self->matrix.data[i][0]);
        PyObject *t1 = PyFloat_FromDouble(self->matrix.data[i][1]);
        PyObject *t2 = PyFloat_FromDouble(self->matrix.data[i][2]);
        PyObject *t3 = PyFloat_FromDouble(self->matrix.data[i][3]);
        tuples[i] = PyTuple_Pack(4, t0, t1, t2, t3);
        Py_DECREF(t0);
        Py_DECREF(t1);
        Py_DECREF(t2);
        Py_DECREF(t3);
    }
    
    full_tuple = PyTuple_Pack(4, tuples[0], tuples[1], tuples[2], tuples[3]);
    Py_DECREF(tuples[0]);
    Py_DECREF(tuples[1]);
    Py_DECREF(tuples[2]);
    Py_DECREF(tuples[3]);
    return full_tuple;
}

static PyObject *PyOILMatrix_transform(PyOILMatrix *self, PyObject *args) {
    float x, y, z;
    
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z)) {
        return NULL;
    }
    
    oil_matrix_transform(&(self->matrix), &x, &y, &z);
    return Py_BuildValue("fff", x, y, z);
}

static int PyOILMatrix_set_data(PyOILMatrix *self, PyObject *arg, void *unused) {
    int i;
    float data[4][4];
    PyObject *argfast = PySequence_Fast(arg, "matrix data is not a 4x4 sequence of sequences");
    if (!argfast)
        return 1;
    
    if (PySequence_Fast_GET_SIZE(argfast) != 4) {
        Py_DECREF(argfast);
        PyErr_SetString(PyExc_ValueError, "matrix data is not a 4x4 sequence of sequences");
        return 1;
    }
    
    for (i = 0; i < 4; i++) {
        int j;
        PyObject *argi = PySequence_Fast_GET_ITEM(argfast, i);
        PyObject *argifast = PySequence_Fast(argi, "matrix data is not a 4x4 sequence of sequences");
        if (!argifast) {
            Py_DECREF(argfast);
            return 1;
        }
        
        if (PySequence_Fast_GET_SIZE(argifast) != 4) {
            Py_DECREF(argfast);
            Py_DECREF(argifast);
            PyErr_SetString(PyExc_ValueError, "matrix data is not a 4x4 sequence of sequences");
            return 1;
        }
        
        for (j = 0; j < 4; j++) {
            PyObject *val = PySequence_Fast_GET_ITEM(argifast, j);
            PyObject *fval = PyNumber_Float(val);
            if (!fval) {
                Py_DECREF(argfast);
                Py_DECREF(argifast);
                PyErr_SetString(PyExc_ValueError, "matrix data is not composed of float-compatible numbers");
                return 1;
            }
            
            data[i][j] = PyFloat_AsDouble(fval);
        }
        
        Py_DECREF(argifast);
    }
    
    Py_DECREF(argfast);
    oil_matrix_set_data(&(self->matrix), (float *)data);
    return 0;
}

static PyObject *PyOILMatrix_str(PyOILMatrix *self) {
    PyObject *full_tuple = PyOILMatrix_get_data(self, NULL);
    PyObject *repr;
    
    repr = PyObject_Repr(full_tuple);
    Py_DECREF(full_tuple);
    return repr;
}

static PyObject *PyOILMatrix_repr(PyOILMatrix *self) {
    PyObject *repr = PyOILMatrix_str(self);
    PyObject *full_repr = PyString_FromFormat("OIL.Matrix(%s)", PyString_AsString(repr));
    Py_DECREF(repr);
    return full_repr;
}

static inline PyObject *PyOILMatrix_binop(PyOILMatrix *a, PyOILMatrix *b, void (*op)(OILMatrix *, const OILMatrix *, const OILMatrix *), int inplace) {
    if (!PyObject_TypeCheck((PyObject *)a, &PyOILMatrixType) ||
        !PyObject_TypeCheck((PyObject *)b, &PyOILMatrixType)) {
        Py_INCREF(Py_NotImplemented);
        return Py_NotImplemented;
    }
    
    if (!inplace) {
        PyOILMatrix *result;
        result = (PyOILMatrix *)(PyOILMatrixType.tp_alloc(&PyOILMatrixType, 0));
        if (!result)
            return NULL;
        
        op(&(result->matrix), &(a->matrix), &(b->matrix));
        return (PyObject *)result;
    }
    
    op(&(a->matrix), &(a->matrix), &(b->matrix));
    Py_INCREF(a);
    return (PyObject *)a;
}

static PyObject *PyOILMatrix_add(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_add, 0);
}

static PyObject *PyOILMatrix_inplace_add(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_add, 1);
}

static PyObject *PyOILMatrix_subtract(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_subtract, 0);
}

static PyObject *PyOILMatrix_inplace_subtract(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_subtract, 1);
}

static PyObject *PyOILMatrix_multiply(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_multiply, 0);
}

static PyObject *PyOILMatrix_inplace_multiply(PyOILMatrix *a, PyOILMatrix *b) {
    return PyOILMatrix_binop(a, b, oil_matrix_multiply, 1);
}

static PyObject *PyOILMatrix_negative(PyOILMatrix *mat) {
    PyOILMatrix *result;
    result = (PyOILMatrix *)(PyOILMatrixType.tp_alloc(&PyOILMatrixType, 0));
    if (!result)
        return NULL;
    oil_matrix_negate(&(result->matrix), &(mat->matrix));
    return (PyObject *)result;
}

static PyObject *PyOILMatrix_positive(PyOILMatrix *mat) {
    Py_INCREF(mat);
    return (PyObject *)mat;
}

static int PyOILMatrix_nonzero(PyOILMatrix *mat) {
    return !oil_matrix_is_zero(&(mat->matrix));
}

static PyObject *PyOILMatrix_get_inverse(PyOILMatrix *self, PyObject *args) {
    PyOILMatrix *other;
    
    /* args == NULL means we're called as an attribute-getter */
    if (args != NULL) {
        if (!PyArg_ParseTuple(args, "")) {
            return NULL;
        }
    }
    
    other = (PyOILMatrix *)(PyOILMatrixType.tp_alloc(&PyOILMatrixType, 0));
    if (!other)
        return NULL;

    if (!oil_matrix_invert(&(other->matrix), &(self->matrix))) {
        Py_DECREF(other);
        PyErr_SetString(PyExc_ValueError, "cannot invert matrix");
        return NULL;
    }
    return (PyObject *)other;
}

static PyObject *PyOILMatrix_invert(PyOILMatrix *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "")) {
        return NULL;
    }
    
    if (!oil_matrix_invert(&(self->matrix), &(self->matrix))) {
        PyErr_SetString(PyExc_ValueError, "cannot invert matrix");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyOILMatrix_translate(PyOILMatrix *self, PyObject *args) {
    float x, y, z;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z))
        return NULL;
    
    oil_matrix_translate(&(self->matrix), x, y, z);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *PyOILMatrix_scale(PyOILMatrix *self, PyObject *args) {
    float x, y, z;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z))
        return NULL;
    
    oil_matrix_scale(&(self->matrix), x, y, z);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *PyOILMatrix_rotate(PyOILMatrix *self, PyObject *args) {
    float x, y, z;
    if (!PyArg_ParseTuple(args, "fff", &x, &y, &z))
        return NULL;
    
    oil_matrix_rotate(&(self->matrix), x, y, z);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *PyOILMatrix_orthographic(PyOILMatrix *self, PyObject *args) {
    float x1, x2, y1, y2, z1, z2;
    if (!PyArg_ParseTuple(args, "ffffff", &x1, &x2, &y1, &y2, &z1, &z2))
        return NULL;
    
    oil_matrix_orthographic(&(self->matrix), x1, x2, y1, y2, z1, z2);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyMethodDef PyOILMatrix_methods[] = {
    {"get_data", (PyCFunction)PyOILMatrix_get_data, METH_VARARGS,
     "Returns a nested tuple of the matrix data."},
    {"transform", (PyCFunction)PyOILMatrix_transform, METH_VARARGS,
     "Transform 3 coordinates."},
    {"get_inverse", (PyCFunction)PyOILMatrix_get_inverse, METH_VARARGS,
     "Returns the inverse of the matrix."},
    {"invert", (PyCFunction)PyOILMatrix_invert, METH_VARARGS,
     "Invert the matrix in-place."},
    {"translate", (PyCFunction)PyOILMatrix_translate, METH_VARARGS,
     "Multiply on a translation matrix."},
    {"scale", (PyCFunction)PyOILMatrix_scale, METH_VARARGS,
     "Multiply on a scaling matrix."},
    {"rotate", (PyCFunction)PyOILMatrix_rotate, METH_VARARGS,
     "Multiply on a rotation matrix."},
    {"orthographic", (PyCFunction)PyOILMatrix_orthographic, METH_VARARGS,
     "Multiply on an orthographic matrix."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyOILMatrix_getset[] = {
    {"data", (getter)PyOILMatrix_get_data, (setter)PyOILMatrix_set_data,
     "A nested tuple of matrix data.", NULL},
    {"inverse", (getter)PyOILMatrix_get_inverse, NULL,
     "Return the inverse of this matrix.", NULL},
    {NULL, NULL, 0, NULL}
};

/* the Matrix math ops */
static PyNumberMethods PyOILMatrixNumberMethods = {
    (binaryfunc)PyOILMatrix_add, /* nb_add */
    (binaryfunc)PyOILMatrix_subtract, /* nb_subtract */
    (binaryfunc)PyOILMatrix_multiply, /* nb_multiply */
    NULL,                      /* nb_divide */
    NULL,                      /* nb_remainder */
    NULL,                      /* nb_divmod */
    NULL,                      /* nb_power */
    (unaryfunc)PyOILMatrix_negative, /* nb_negative */
    (unaryfunc)PyOILMatrix_positive, /* nb_positive */
    NULL,                      /* nb_absolute */
    (inquiry)PyOILMatrix_nonzero, /* nb_nonzero */
    NULL,                      /* nb_invert */
    NULL,                      /* nb_lshift */
    NULL,                      /* nb_rshift */
    NULL,                      /* nb_and */
    NULL,                      /* nb_xor */
    NULL,                      /* nb_or */
    NULL,                      /* nb_coerce */
    NULL,                      /* nb_int */
    NULL,                      /* nb_long */
    NULL,                      /* nb_float */
    NULL,                      /* nb_oct */
    NULL,                      /* nb_hex */

    (binaryfunc)PyOILMatrix_inplace_add, /* nb_inplace_add */
    (binaryfunc)PyOILMatrix_inplace_subtract, /* nb_inplace_subtract */
    (binaryfunc)PyOILMatrix_inplace_multiply, /* nb_inplace_multiply */
    NULL,                      /* nb_inplace_divide */
    NULL,                      /* nb_inplace_remainder */
    NULL,                      /* nb_inplace_power */
    NULL,                      /* nb_inplace_lshift */
    NULL,                      /* nb_inplace_rshift */
    NULL,                      /* nb_inplace_and */
    NULL,                      /* nb_inplace_xor */
    NULL,                      /* nb_inplace_or */
};    

/* the Matrix type */
PyTypeObject PyOILMatrixType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
    "Matrix",                  /* tp_name */
    sizeof(PyOILMatrix),       /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyOILMatrix_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)PyOILMatrix_repr, /* tp_repr */
    &(PyOILMatrixNumberMethods), /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    (reprfunc)PyOILMatrix_str, /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
                               /* tp_doc */
    "Encapsulates matrix data and operations.",
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyOILMatrix_methods,       /* tp_methods */
    NULL,                      /* tp_members */
    PyOILMatrix_getset,        /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)PyOILMatrix_new,  /* tp_new */
};

/* init and dealloc for the Image type */

static PyOILImage *PyOILImage_new(PyTypeObject *subtype, PyObject *args, PyObject *kwargs) {
    PyOILImage *self;
    unsigned int width = 0, height = 0;
    
    if (!PyArg_ParseTuple(args, "II", &width, &height)) {
        return NULL;
    }
    
    if (width == 0 || height == 0) {
        PyErr_SetString(PyExc_ValueError, "cannot create image with 0 size");
        return NULL;
    }
    
    self = (PyOILImage *)(subtype->tp_alloc(subtype, 0));
    if (!self)
        return NULL;
    
    self->im = oil_image_new(width, height);
    if (self->im == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cannot create image");
        Py_DECREF(self);
        return NULL;
    }
    
    return self;
}

static void PyOILImage_dealloc(PyOILImage *self) {
    if (self->im) {
        oil_image_free(self->im);
        self->im = NULL;
    }
    self->ob_type->tp_free((PyObject *)self);
}

/* methods for Image type */

static PyObject *PyOILImage_load(PyTypeObject *type, PyObject *args) {
    PyObject *src = NULL;
    const char *path = NULL;
    PyOILImage *self;
    
    if (!PyArg_ParseTuple(args, "O", &src)) {
        return NULL;
    }
    
    if (PyString_Check(src)) {
        path = PyString_AsString(src);
    }

    self = (PyOILImage *)(type->tp_alloc(type, 0));
    if (!self)
        return NULL;
    
    if (path) {
        self->im = oil_image_load(path);
    } else {
        OILFile file;
        file.file = src;
        file.read = oil_python_read;
        file.write = oil_python_write;
        file.flush = oil_python_flush;
    
        self->im = oil_image_load_ex(&file);
    }
    
    if (self->im == NULL) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_IOError, "cannot load image");
        Py_DECREF(self);
        return NULL;
    }
    
    return (PyObject *)self;
}

static PyObject *PyOILImage_save(PyOILImage *self, PyObject *args, PyObject *kwargs) {
    OILFormatOptions opts = {0, 0};
    static char *argnames[] = {"dest", "indexed", "palette_size", NULL};
    PyObject *dest = NULL;
    const char *path = NULL;
    int save_success = 0;
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iI", argnames,
                                     &dest, &opts.indexed, &opts.palette_size)) {
        return NULL;
    }
    
    if (PyString_Check(dest)) {
        path = PyString_AsString(dest);
    }
    
    if (path) {
        save_success = oil_image_save(self->im, path, &opts);
    } else {
        OILFile file;
        file.file = dest;
        file.read = oil_python_read;
        file.write = oil_python_write;
        file.flush = oil_python_flush;
        
        save_success = oil_image_save_ex(self->im, &file, &opts);
    }
    
    if (!save_success) {
        if (!PyErr_Occurred())
            PyErr_SetString(PyExc_IOError, "cannot save image");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject *PyOILImage_get_size(PyOILImage *self, PyObject *args) {
    unsigned int width = 0, height = 0;
    
    /* args == NULL means we're called as an attribute-getter */
    if (args != NULL) {
        if (!PyArg_ParseTuple(args, "")) {
            return NULL;
        }
    }
    
    oil_image_get_size(self->im, &width, &height);
    return Py_BuildValue("II", width, height);
}

static PyObject *PyOILImage_composite(PyOILImage *self, PyObject *args) {
    PyOILImage *src;
    unsigned char alpha = 255;
    int dx = 0, dy = 0;
    unsigned int sx = 0, sy = 0, xsize = 0, ysize = 0;

    if (!PyArg_ParseTuple(args, "O!|biiIIII", &PyOILImageType, &src, &alpha, &dx, &dy, &sx, &sy, &xsize, &ysize)) {
        return NULL;
    }
    
    if (!oil_image_composite(self->im, src->im, alpha, dx, dy, sx, sy, xsize, ysize)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot composite image");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject *PyOILImage_draw_triangles(PyOILImage *self, PyObject *args) {
    unsigned int i;
    PyOILMatrix *matrix = NULL;
    PyOILImage *tex = NULL;
    PyObject *pyvertices = NULL;
    PyObject *pyindices = NULL;
    unsigned int flags = 0;
    OILVertex *vertices = NULL;
    unsigned int vertices_length = 0;
    unsigned int *indices = NULL;
    unsigned int indices_length = 0;
    
    if (!PyArg_ParseTuple(args, "O!O!OO|I", &PyOILMatrixType, &matrix, &PyOILImageType, &tex, &pyvertices, &pyindices, &flags)) {
        return NULL;
    }
    
    pyvertices = PySequence_Fast(pyvertices, "vertices are not a sequence");
    if (pyvertices)
        pyindices = PySequence_Fast(pyindices, "indices are not a sequence");
    if (!pyvertices || !pyindices) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pyindices);
        return NULL;
    }
    
    vertices_length = PySequence_Fast_GET_SIZE(pyvertices);
    vertices = malloc(sizeof(OILVertex) * vertices_length);
    indices_length = PySequence_Fast_GET_SIZE(pyindices);
    indices = malloc(sizeof(unsigned int) * indices_length);
    if (!vertices || !indices) {
        if (vertices)
            free(vertices);
        if (indices)
            free(indices);
        Py_DECREF(pyvertices);
        Py_DECREF(pyindices);
        PyErr_SetString(PyExc_RuntimeError, "out of memory");
        return NULL;
    }
    
    for (i = 0; i < vertices_length; i++) {
        PyObject *vert = PySequence_Fast_GET_ITEM(pyvertices, i);
        if (!PyArg_ParseTuple(vert, "(fff)(ff)(bbbb)", &(vertices[i].x), &(vertices[i].y), &(vertices[i].z), &(vertices[i].s), &(vertices[i].t), &(vertices[i].color.r), &(vertices[i].color.g), &(vertices[i].color.b), &(vertices[i].color.a))) {
            free(vertices);
            free(indices);
            Py_DECREF(pyvertices);
            Py_DECREF(pyindices);
            PyErr_SetString(PyExc_ValueError, "vertex has invalid form");
            return NULL;
        }
    }
    
    for (i = 0; i < indices_length; i++) {
        PyObject *pyindex = PySequence_Fast_GET_ITEM(pyindices, i);
        pyindex = PyNumber_Index(pyindex);
        if (!pyindex) {
            free(vertices);
            free(indices);
            Py_DECREF(pyvertices);
            Py_DECREF(pyindices);
            PyErr_SetString(PyExc_ValueError, "index is not valid");
            return NULL;
        }
        
        indices[i] = PyInt_AsLong(pyindex);
        Py_DECREF(pyindex);
    }
    
    Py_DECREF(pyvertices);
    Py_DECREF(pyindices);
    
    oil_image_draw_triangles(self->im, &(matrix->matrix), tex->im, vertices, indices, indices_length, flags);
    
    free(vertices);
    free(indices);
    Py_RETURN_NONE;
}

static PyMethodDef PyOILImage_methods[] = {
    {"load", (PyCFunction)PyOILImage_load, METH_VARARGS | METH_CLASS,
     "Load the given path name into an Image object."},
    {"save", (PyCFunction)PyOILImage_save, METH_KEYWORDS,
     "Save the Image object to a file."},
    {"get_size", (PyCFunction)PyOILImage_get_size, METH_VARARGS,
     "Return a (width, height) tuple."},
    {"composite", (PyCFunction)PyOILImage_composite, METH_VARARGS,
     "Composite another image on top of this one."},
    {"draw_triangles", (PyCFunction)PyOILImage_draw_triangles, METH_VARARGS,
     "Draw 3D triangles on top of the image."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyOILImage_getset[] = {
    {"size", (getter)PyOILImage_get_size, NULL,
     "Return a (width, height) tuple.", NULL},
    {NULL, NULL, 0, NULL}
};

/* the Image type */
PyTypeObject PyOILImageType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /* ob_size */
    "Image",                   /* tp_name */
    sizeof(PyOILImage),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)PyOILImage_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
                               /* tp_doc */
    "Encapsulates image data and image operations.",
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    PyOILImage_methods,        /* tp_methods */
    NULL,                      /* tp_members */
    PyOILImage_getset,         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)PyOILImage_new,   /* tp_new */
};

static PyObject *OIL_backend_set(PyObject *self, PyObject *args) {
    unsigned int backend = OIL_BACKEND_CPU;
    if (!PyArg_ParseTuple(args, "I", &backend)) {
        return NULL;
    }
    
    if (backend >= OIL_BACKEND_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid backend");
        return NULL;
    }
    
    oil_backend_set(backend);
    Py_RETURN_NONE;
}

static PyMethodDef OIL_methods[] = {
    {"backend_set", OIL_backend_set, METH_VARARGS,
     "Set the OIL backend to use"},
    {NULL, NULL, 0, NULL}
};

/* helper to add a type to the module */
#define ADD_TYPE(type) do {                                         \
        if (PyType_Ready(&type) < 0)                                \
            return;                                                 \
        PyModule_AddObject(mod, type.tp_name, (PyObject *)&type);   \
    } while (0)

PyMODINIT_FUNC initOIL(void) {
    PyObject *mod;
    
    mod = Py_InitModule3("OIL", OIL_methods,
                         "OIL is an image handling library for Python.");
    if (mod == NULL)
        return;
    
    ADD_TYPE(PyOILMatrixType);
    ADD_TYPE(PyOILImageType);
    
    /* add in the flag enums */
    PyModule_AddIntConstant(mod, "DEPTH_TEST", OIL_DEPTH_TEST);
    
    /* add in the backend enums */
    PyModule_AddIntConstant(mod, "BACKEND_CPU", OIL_BACKEND_CPU);
    PyModule_AddIntConstant(mod, "BACKEND_DEBUG", OIL_BACKEND_DEBUG);
}

