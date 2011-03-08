#include <Python.h>

#include <numpy/arrayobject.h>
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

// macro for getting blockID from a chunk of memory
#define getBlock(blockThing, x,y,z) blockThing[ y + ( z * 128 + ( x * 128 * 16) ) ]

static inline int isTransparent(unsigned char b) {
    // TODO expand this to include all transparent blocks
    return b == 0;

}


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



// TODO refact iterate.c and _composite.c so share implementations
static PyObject* alpha_over(PyObject* dest, PyObject* t, int imgx, int imgy)
{
	/* raw input python variables */
	PyObject * src, * mask;
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
	
    src = PyTuple_GET_ITEM(t, 0);
    mask = PyTuple_GET_ITEM(t, 1);
    if (mask == Py_None) {
        printf("mask is none\n");
        Py_INCREF(mask);
        mask = src;
    }
	//if (!PyArg_ParseTuple(args, "OOOO", &dest, &src, &pos, &mask))
	//	return NULL;
	
	imDest = imaging_python_to_c(dest);
	imSrc = imaging_python_to_c(src);
	imMask = imaging_python_to_c(mask);

    //printf("alpha1\n");    
	if (!imDest || !imSrc || !imMask) {
		PyErr_SetString(PyExc_ValueError, "dest, src, or mask is missing");
		return NULL;
    }
    //printf("alpha2\n");    
	
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
	
    //printf("alpha3\n");    
	/* set up flags for the src/mask type */
    src_has_alpha = (imSrc->pixelsize == 4 ? 1 : 0);
	/* how far into image the first alpha byte resides */
	mask_offset = (imMask->pixelsize == 4 ? 3 : 0);
	/* how many bytes to skip to get to the next alpha byte */
	mask_stride = imMask->pixelsize;
	
    //printf("alpha4\n");    
	/* destination position read */
	//if (!PyArg_ParseTuple(pos, "iiii", &dx, &dy, &xsize, &ysize))
	//{
	//	PyErr_SetString(PyExc_TypeError, "given blend destination rect is not valid");
	//	return NULL;
	//}
    dx = imgx;
    dy = imgy;

    xsize = imSrc->xsize;
    ysize = imSrc->ysize;
    //printf("xsize/ysize %d/%d\n", xsize, ysize);

    //printf("alpha5\n");    
	
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
   
    //printf("imDest->xsize=%d imDest->yize=%d\n", imDest->xsize, imDest->ysize); 
	
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



// TODO triple check this to make sure reference counting is correct
static PyObject*
chunk_render(PyObject *self, PyObject *args) {

    PyObject *chunk;
    PyObject *blockdata_expanded; 
    int xoff, yoff;
    PyObject *img;

    if (!PyArg_ParseTuple(args, "OOiiO",  &chunk, &img, &xoff, &yoff, &blockdata_expanded))
        return Py_BuildValue("i", "-1");

    // tuple
    PyObject *imgsize = PyObject_GetAttrString(img, "size");

    PyObject *imgsize0_py = PySequence_GetItem(imgsize, 0);
    PyObject *imgsize1_py = PySequence_GetItem(imgsize, 1);
    Py_DECREF(imgsize);

    int imgsize0 = PyInt_AsLong(imgsize0_py);
    int imgsize1 = PyInt_AsLong(imgsize1_py);
    Py_DECREF(imgsize0_py);
    Py_DECREF(imgsize1_py);


    // get the block data directly from numpy:
    PyObject *blocks_py = PyObject_GetAttrString(chunk, "blocks");
    char *blocks = PyArray_BYTES(blocks_py);
    Py_DECREF(blocks_py);

    //PyObject *left_blocks = PyObject_GetAttrString(chunk, "left_blocks");
    //PyObject *right_blocks = PyObject_GetAttrString(chunk, "right_blocks");
    //PyObject *transparent_blocks = PyObject_GetAttrString(chunk, "transparent_blocks");
    
    PyObject *textures = PyImport_ImportModule("textures");

    // TODO can these be global static?  these don't change during program execution
    PyObject *blockmap = PyObject_GetAttrString(textures, "blockmap");
    PyObject *special_blocks = PyObject_GetAttrString(textures, "special_blocks");
    PyObject *specialblockmap = PyObject_GetAttrString(textures, "specialblockmap");

    Py_DECREF(textures);
    
    //printf("render_loop\n");

    int imgx, imgy;
    int x, y, z;
    for (x = 15; x > -1; x--) {
        for (y = 0; y < 16; y++) {
            imgx = xoff + x*12 + y*12;
            imgy = yoff - x*6 + y*6 + 1632; // 1632 == 128*12 + 16*12//2
            for (z = 0; z < 128; z++) {
                //printf("c/imgx/%d\n", imgx);
                //printf("c/imgy/%d\n", imgy);
                if ((imgx >= imgsize0 + 24) || (imgx <= -24)) {
                    imgy -= 12; // NOTE: we need to do this at every continue
                    continue;
                }
                if ((imgy >= imgsize1 + 24) || (imgy <= -24)) {
                    imgy -= 12;
                    continue;
                }

                // get blockid
                unsigned char block = getBlock(blocks, x, z, y);  // Note the order: x,z,y
                if (block == 0) {
                    imgy -= 12;
                    continue;
                }
                //printf("checking blockid %hhu\n", block);
                PyObject *blockid = PyInt_FromLong(block); // TODO figure out how to DECREF this easily, instead at every 'continue'. 


                if ( (x != 0) && (y != 15) && (z != 127) &&
                        !isTransparent(getBlock(blocks, x-1, z, y)) &&
                        !isTransparent(getBlock(blocks, x, z+1, y)) &&
                        !isTransparent(getBlock(blocks, x, z, y+1))    ) {
                    imgy -= 12;
                    continue;
                }


                if (!PySequence_Contains(special_blocks, blockid)) {
                    //t = textures.blockmap[blockid]
                    PyObject *t = PyList_GetItem(blockmap, block);
                    // PyList_GetItem returns borrowed ref
                    if (t == Py_None) {
                        printf("t == Py_None.  blockid=%d\n", block);
                        imgy -= 12;
                        continue;

                    }

                    // note that this version of alpha_over has a different signature than the 
                    // version in _composite.c
                    alpha_over(img, t, imgx, imgy );

                } else {
                    // this should be a pointer to a unsigned char
                    void* ancilData_p = PyArray_GETPTR3(blockdata_expanded, x, y, z); 
                    unsigned char ancilData = *((unsigned char*)ancilData_p);
                    if (block == 85) { // fence.  skip the generate_pseudo_ancildata for now
                        imgy -= 12;
                        continue;
                    }
                    PyObject *tmp = PyTuple_New(2);

                    Py_INCREF(blockid); // because SetItem steals
                    PyTuple_SetItem(tmp, 0, blockid);
                    PyTuple_SetItem(tmp, 1, PyInt_FromLong(ancilData));
                    PyObject *t = PyDict_GetItem(specialblockmap, tmp);  // this is a borrowed reference.  no need to decref
                    Py_DECREF(tmp);
                    if (t != NULL) 
                        alpha_over(img, t, imgx, imgy );
                    imgy -= 12;
                    continue;
                }
                imgy -= 12;
            }
        }
    } 

    Py_DECREF(blockmap);
    Py_DECREF(special_blocks);
    Py_DECREF(specialblockmap);

    return Py_BuildValue("i",2);
}

static PyMethodDef IterateMethods[] = {
    {"render_loop", chunk_render, METH_VARARGS,
        "Renders stuffs"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
init_iterate(void)
{
        (void) Py_InitModule("_iterate", IterateMethods);
        import_array();  // for numpy
}
