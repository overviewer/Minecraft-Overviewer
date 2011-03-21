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

/*
 * This is a general include file for the Overviewer C extension. It
 * lists useful, defined functions as well as those that are exported
 * to python, so all files can use them.
 */

#ifndef __OVERVIEWER_H_INCLUDED__
#define __OVERVIEWER_H_INCLUDED__

/* Python and PIL headers */
#include <Python.h>
#include <Imaging.h>

/* in composite.c */
Imaging imaging_python_to_c(PyObject *obj);
PyObject *alpha_over(PyObject *dest, PyObject *src, PyObject *mask, int dx,
                     int dy, int xsize, int ysize);
PyObject *alpha_over_wrap(PyObject *self, PyObject *args);
PyObject *brightness(PyObject *img, double factor);

/* in iterate.c */
int is_transparent(PyObject* tup, unsigned char b);
PyObject *chunk_render(PyObject *self, PyObject *args);
int init_chunk_render(void);

#endif /* __OVERVIEWER_H_INCLUDED__ */
