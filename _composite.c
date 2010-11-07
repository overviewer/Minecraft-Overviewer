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
 * This file implements a custom alpha_over function for (some) PIL
 * images. It's designed to be used through composite.py, which
 * includes a proxy alpha_over function that falls back to the default
 * PIL paste if this extension is not found.
 */

#include <Python.h>
#include <Imaging.h>

/* like (a * b + 127) / 255), but much faster on most platforms
   from PIL's _imaging.c */
#define MULDIV255(a, b, tmp)								\
	(tmp = (a) * (b) + 128, ((((tmp) >> 8) + (tmp)) >> 8))

typedef struct
{
	PyObject_HEAD
	Imaging image;
} ImagingObject;

static Imaging imaging_python_to_c(PyObject* obj)
{
	PyObject* im;
	Imaging image;

	/* first, get the 'im' attribute */
	im = PyObject_GetAttrString(obj, "im");
	if (!im)
		return NULL;
	
	/* make sure 'im' is the right type */
	if (strcmp(im->ob_type->tp_name, "ImagingCore") != 0)
	{
		/* it's not -- raise an error and exit */
		PyErr_SetString(PyExc_TypeError, "image attribute 'im' is not a core Imaging type");
		return NULL;
	}
	
	image = ((ImagingObject*)im)->image;
	Py_DECREF(im);
	return image;
}

static PyObject* _composite_alpha_over(PyObject* self, PyObject* args)
{
	/* raw input python variables */
	PyObject* dest, * src, * pos, * mask;
	/* libImaging handles */
	Imaging imDest, imSrc, imMask;
	/* cached blend properties */
	int src_has_alpha, mask_offset, mask_stride;
	/* destination position and size */
	int dx, dy, xsize, ysize;
	/* source position */
	int sx, sy;
	/* iteration variables */
	unsigned int x, y, i;
	/* temporary calculation variables */
	int tmp1, tmp2, tmp3;
	
	if (!PyArg_ParseTuple(args, "OOOO", &dest, &src, &pos, &mask))
		return NULL;
	
	imDest = imaging_python_to_c(dest);
	imSrc = imaging_python_to_c(src);
	imMask = imaging_python_to_c(mask);
	
	if (!imDest || !imSrc || !imMask)
		return NULL;
	
	/* check the various image modes, make sure they make sense */
	if (strcmp(imDest->mode, "RGBA") != 0)
	{
		PyErr_SetString(PyExc_ValueError, "given destination image does not have mode \"RGBA\"");
		return NULL;
	}
	
	if (strcmp(imSrc->mode, "RGBA") != 0 && strcmp(imSrc->mode, "RGB") != 0)
	{
		PyErr_SetString(PyExc_ValueError, "given source image does not have mode \"RGBA\" or \"RGB\"");
		return NULL;
	}
	
	if (strcmp(imMask->mode, "RGBA") != 0 && strcmp(imMask->mode, "L") != 0)
	{
		PyErr_SetString(PyExc_ValueError, "given mask image does not have mode \"RGBA\" or \"L\"");
		return NULL;
	}
	
	/* make sure mask size matches src size */
	if (imSrc->xsize != imMask->xsize || imSrc->ysize != imMask->ysize)
	{
		PyErr_SetString(PyExc_ValueError, "mask and source image sizes do not match");
		return NULL;
	}
	
	/* set up flags for the src/mask type */
    src_has_alpha = (imSrc->pixelsize == 4 ? 1 : 0);
	/* how far into image the first alpha byte resides */
	mask_offset = (imMask->pixelsize == 4 ? 3 : 0);
	/* how many bytes to skip to get to the next alpha byte */
	mask_stride = imMask->pixelsize;
	
	/* destination position read */
	if (!PyArg_ParseTuple(pos, "iiii", &dx, &dy, &xsize, &ysize))
	{
		PyErr_SetString(PyExc_TypeError, "given blend destination rect is not valid");
		return NULL;
	}
	
	/* set up the source position, size and destination position */
	/* handle negative dest pos */
	if (dx < 0)
	{
		sx = -dx;
		dx = 0;
	} else {
		sx = 0;
	}
	
	if (dy < 0)
	{
		sy = -dy;
		dy = 0;
	} else {
		sy = 0;
	}
	
	/* set up source dimensions */
	xsize -= sx;
	ysize -= sy;
	
	/* clip dimensions, if needed */
	if (dx + xsize > imDest->xsize)
		xsize = imDest->xsize - dx;
	if (dy + ysize > imDest->ysize)
		ysize = imDest->ysize - dy;
	
	/* check that there remains any blending to be done */
	if (xsize <= 0 || ysize <= 0)
	{
		/* nothing to do, return */
		Py_INCREF(dest);
		return dest;
	}
	
	for (y = 0; y < ysize; y++)
	{
		UINT8* out = (UINT8*) imDest->image[dy + y] + dx*4;
		UINT8* outmask = (UINT8*) imDest->image[dy + y] + dx*4 + 3;
		UINT8* in = (UINT8*) imSrc->image[sy + y] + sx*(imSrc->pixelsize);
		UINT8* inmask = (UINT8*) imMask->image[sy + y] + sx*mask_stride + mask_offset;
		
		for (x = 0; x < xsize; x++)
		{
			/* special cases */
			if (*inmask == 255 || *outmask == 0)
			{
				*outmask = *inmask;
				
				*out = *in;
				out++, in++;
				*out = *in;
				out++, in++;
				*out = *in;
				out++, in++;
			} else if (*inmask == 0) {
				/* do nothing -- source is fully transparent */
				out += 3;
				in += 3;
			} else {
				/* general case */
				int alpha = *inmask + MULDIV255(*outmask, 255 - *inmask, tmp1);
				for (i = 0; i < 3; i++)
				{
					/* general case */
					*out = MULDIV255(*in, *inmask, tmp1) + MULDIV255(MULDIV255(*out, *outmask, tmp2), 255 - *inmask, tmp3);
					*out = (*out * 255) / alpha;
					out++, in++;
				}
				
				*outmask = alpha;
			}
			
			out++;
			if (src_has_alpha)
				in++;
			outmask += 4;
			inmask += mask_stride;
		}
	}
	
	Py_INCREF(dest);
	return dest;
}

static PyMethodDef _CompositeMethods[] =
{
	{"alpha_over", _composite_alpha_over, METH_VARARGS, "alpha over composite function"},
	{NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_composite(void)
{
	(void) Py_InitModule("_composite", _CompositeMethods);
}
