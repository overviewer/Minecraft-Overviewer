from overviewer.oil.cdefs cimport *

cdef class Matrix:
    cdef OILMatrix _m

cdef class Image:
    cdef OILImage *_im
