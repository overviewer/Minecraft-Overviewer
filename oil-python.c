#include <Python.h>

#include "oil.h"

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
            PyErr_SetString(PyExc_RuntimeError, "cannot load image");
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
    {"save", (PyCFunction)PyOILImage_save, METH_KEYWORDS,
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

PyMODINIT_FUNC initOIL(void) {
    PyObject *mod;
    
    mod = Py_InitModule3("OIL", OIL_methods,
                         "OIL is an image handling library for Python.");
    if (mod == NULL)
        return;
    
    if (PyType_Ready(&PyOILImageType) < 0)
        return;
    
    PyModule_AddObject(mod, "Image", (PyObject *)&PyOILImageType);
    
    /* add in the backend enums */
    PyModule_AddIntConstant(mod, "BACKEND_CPU", OIL_BACKEND_CPU);
    PyModule_AddIntConstant(mod, "BACKEND_DEBUG", OIL_BACKEND_DEBUG);
}

