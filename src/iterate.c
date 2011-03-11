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

#include <numpy/arrayobject.h>

// macro for getting blockID from a chunk of memory
#define getBlock(blockThing, x,y,z) blockThing[ y + ( z * 128 + ( x * 128 * 16) ) ]

static inline int isTransparent(unsigned char b) {
    // TODO expand this to include all transparent blocks
    return b == 0;

}

// helper to handle alpha_over calls involving a texture tuple
static inline PyObject *
texture_alpha_over(PyObject *dest, PyObject *t, int imgx, int imgy)
{
    PyObject* src, * mask;

    src = PyTuple_GET_ITEM(t, 0);
    mask = PyTuple_GET_ITEM(t, 1);
    if (mask == Py_None) {
        mask = src;
    }

    return alpha_over(dest, src, mask, imgx, imgy, 0, 0);
}

// TODO triple check this to make sure reference counting is correct
PyObject*
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
            /* 128*12 -- offset for z direction, 15*6 -- offset for x */
            imgy = yoff - x*6 + y*6 + 128*12 + 15*6;
            for (z = 0; z < 128; z++) {
                imgy -= 12;
                
                if ((imgx >= imgsize0 + 24) || (imgx <= -24)) {
                    continue;
                }
                if ((imgy >= imgsize1 + 24) || (imgy <= -24)) {
                    continue;
                }

                // get blockid
                unsigned char block = getBlock(blocks, x, z, y);  // Note the order: x,z,y
                if (block == 0) {
                    continue;
                }
                //printf("checking blockid %hhu\n", block);
                PyObject *blockid = PyInt_FromLong(block); // TODO figure out how to DECREF this easily, instead at every 'continue'. 


                if ( (x != 0) && (y != 15) && (z != 127) &&
                        !isTransparent(getBlock(blocks, x-1, z, y)) &&
                        !isTransparent(getBlock(blocks, x, z+1, y)) &&
                        !isTransparent(getBlock(blocks, x, z, y+1))    ) {
                    continue;
                }


                if (!PySequence_Contains(special_blocks, blockid)) {
                    //t = textures.blockmap[blockid]
                    PyObject *t = PyList_GetItem(blockmap, block);
                    // PyList_GetItem returns borrowed ref
                    if (t == Py_None) {
                        printf("t == Py_None.  blockid=%d\n", block);
                        continue;

                    }

                    // note that this version of alpha_over has a different signature than the 
                    // version in _composite.c
                    texture_alpha_over(img, t, imgx, imgy );

                } else {
                    // this should be a pointer to a unsigned char
                    void* ancilData_p = PyArray_GETPTR3(blockdata_expanded, x, y, z); 
                    unsigned char ancilData = *((unsigned char*)ancilData_p);
                    if (block == 85) { // fence.  skip the generate_pseudo_ancildata for now
                        continue;
                    }
                    PyObject *tmp = PyTuple_New(2);

                    Py_INCREF(blockid); // because SetItem steals
                    PyTuple_SetItem(tmp, 0, blockid);
                    PyTuple_SetItem(tmp, 1, PyInt_FromLong(ancilData));
                    PyObject *t = PyDict_GetItem(specialblockmap, tmp);  // this is a borrowed reference.  no need to decref
                    Py_DECREF(tmp);
                    if (t != NULL) 
                        texture_alpha_over(img, t, imgx, imgy );
                    continue;
                }
            }
        }
    } 

    Py_DECREF(blockmap);
    Py_DECREF(special_blocks);
    Py_DECREF(specialblockmap);

    return Py_BuildValue("i",2);
}
