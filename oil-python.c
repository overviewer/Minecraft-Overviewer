#include <Python.h>

#include "oil-private.h"

/* the Image type instance */
typedef struct {
    PyObject_HEAD
    OILImage *im;
} PyOILImage;

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
    const char *path = NULL;
    PyOILImage *self;
    
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }

    self = (PyOILImage *)(type->tp_alloc(type, 0));
    if (!self)
        return NULL;
    
    self->im = oil_image_load(path);
    if (self->im == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "cannot load image");
        Py_DECREF(self);
        return NULL;
    }
    
    return (PyObject *)self;
}

static PyObject *PyOILImage_save(PyOILImage *self, PyObject *args) {
    const char *path = NULL;
    
    if (!PyArg_ParseTuple(args, "s", &path)) {
        return NULL;
    }
    
    if (!oil_image_save(self->im, path, NULL)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot save image");
        return NULL;
    }
    
    Py_RETURN_NONE;
}

static PyObject *PyOILImage_get_size(PyOILImage *self, PyObject *args) {
    unsigned int width = 0, height = 0;
    
    oil_image_get_size(self->im, &width, &height);
    return Py_BuildValue("II", width, height);
}

static PyMethodDef PyOILImage_methods[] = {
    {"load", (PyCFunction)PyOILImage_load, METH_VARARGS | METH_CLASS,
     "Load the given path name into an Image object."},
    {"save", (PyCFunction)PyOILImage_save, METH_VARARGS,
     "Save the Image object to a file."},
    {"get_size", (PyCFunction)PyOILImage_get_size, METH_VARARGS,
     "Return a (width, height) tuple."},
    {NULL, NULL, 0, NULL}
};

/* the Image type */
static PyTypeObject PyOILImageType = {
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
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    0,                         /* tp_init */
    0,                         /* tp_alloc */
    (newfunc)PyOILImage_new,   /* tp_new */
};

static PyMethodDef OIL_methods[] = {
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initOIL(void) {
    PyObject *mod;
    
    mod = Py_InitModule3("OIL", OIL_methods,
                         "OIL is an image handling library for Python.");
    if (mod == NULL)
        return;
    
    if (PyType_Ready(&PyOILImageType) < 0)
        return;
    
    Py_INCREF(&PyOILImageType);
    PyModule_AddObject(mod, "Image", (PyObject *)&PyOILImageType);
}

