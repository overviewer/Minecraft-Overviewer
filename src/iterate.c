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

static PyObject *textures = NULL;
static PyObject *chunk_mod = NULL;
static PyObject *blockmap = NULL;
static PyObject *special_blocks = NULL;
static PyObject *specialblockmap = NULL;
static PyObject *transparent_blocks = NULL;

int init_chunk_render(void) {
   
    /* if blockmap (or any of these) is not NULL, then that means that we've 
     * somehow called this function twice.  error out so we can notice this
     * */
    if (blockmap) return 1;

    textures = PyImport_ImportModule("textures");
    chunk_mod = PyImport_ImportModule("chunk");

    blockmap = PyObject_GetAttrString(textures, "blockmap");
    special_blocks = PyObject_GetAttrString(textures, "special_blocks");
    specialblockmap = PyObject_GetAttrString(textures, "specialblockmap");
    transparent_blocks = PyObject_GetAttrString(chunk_mod, "transparent_blocks");
    
    /* ensure none of these pointers are NULL */    
    if ((!transparent_blocks) || (!blockmap) || (!special_blocks) || (!specialblockmap)) {
        fprintf(stderr, "\ninit_chunk_render failed\n");
        return 1;
    }

    return 0;

}

int
is_transparent(unsigned char b) {
    PyObject *block = PyInt_FromLong(b);
    int ret = PySequence_Contains(transparent_blocks, block);
    Py_DECREF(block);
    return ret;

}

/* TODO triple check this to make sure reference counting is correct */
PyObject*
chunk_render(PyObject *self, PyObject *args) {
    RenderState state;

    PyObject *blockdata_expanded; 
    int xoff, yoff;
    
    PyObject *imgsize, *imgsize0_py, *imgsize1_py;
    int imgsize0, imgsize1;
    
    PyObject *blocks_py;
    PyObject *left_blocks_py;
    PyObject *right_blocks_py;
    PyObject *up_left_blocks_py;
    PyObject *up_right_blocks_py;

    RenderModeInterface *rendermode;

    void *rm_data;
                
    PyObject *t = NULL;
    
    if (!PyArg_ParseTuple(args, "OOiiO",  &state.self, &state.img, &xoff, &yoff, &blockdata_expanded))
        return Py_BuildValue("i", "-1");
    
    /* fill in important modules */
    state.textures = textures;
    state.chunk = chunk_mod;
    
    /* set up the render mode */
    rendermode = get_render_mode(&state);
    rm_data = malloc(rendermode->data_size);
    if (rendermode->start(rm_data, &state)) {
        free(rm_data);
        return Py_BuildValue("i", "-1");
    }

    /* get the image size */
    imgsize = PyObject_GetAttrString(state.img, "size");

    imgsize0_py = PySequence_GetItem(imgsize, 0);
    imgsize1_py = PySequence_GetItem(imgsize, 1);
    Py_DECREF(imgsize);

    imgsize0 = PyInt_AsLong(imgsize0_py);
    imgsize1 = PyInt_AsLong(imgsize1_py);
    Py_DECREF(imgsize0_py);
    Py_DECREF(imgsize1_py);


    /* get the block data directly from numpy: */
    blocks_py = PyObject_GetAttrString(state.self, "blocks");
    state.blocks = blocks_py;

    left_blocks_py = PyObject_GetAttrString(state.self, "left_blocks");
    state.left_blocks = left_blocks_py;

    right_blocks_py = PyObject_GetAttrString(state.self, "right_blocks");
    state.right_blocks = right_blocks_py;

    up_left_blocks_py = PyObject_GetAttrString(state.self, "up_left_blocks");
    state.up_left_blocks = up_left_blocks_py;

    up_right_blocks_py = PyObject_GetAttrString(state.self, "up_right_blocks");
    state.up_right_blocks = up_right_blocks_py;
    
    for (state.x = 15; state.x > -1; state.x--) {
        for (state.y = 0; state.y < 16; state.y++) {
            PyObject *blockid = NULL;
            
            /* set up the render coordinates */
            state.imgx = xoff + state.x*12 + state.y*12;
            /* 128*12 -- offset for z direction, 15*6 -- offset for x */
            state.imgy = yoff - state.x*6 + state.y*6 + 128*12 + 15*6;
            
            for (state.z = 0; state.z < 128; state.z++) {
                state.imgy -= 12;
                
                /* make sure we're rendering inside the image boundaries */
                if ((state.imgx >= imgsize0 + 24) || (state.imgx <= -24)) {
                    continue;
                }
                if ((state.imgy >= imgsize1 + 24) || (state.imgy <= -24)) {
                    continue;
                }

                /* get blockid */
                state.block = getArrayByte3D(blocks_py, state.x, state.y, state.z);
                if (state.block == 0) {
                    continue;
                }
                
                /* decref'd on replacement *and* at the end of the z for block */
                if (blockid) {
                    Py_DECREF(blockid);
                }
                blockid = PyInt_FromLong(state.block);

                // check for occlusion
                if (rendermode->occluded(rm_data, &state)) {
                    continue;
                }
                
                // everything stored here will be a borrowed ref

                /* get the texture and mask from block type / ancil. data */
                if (!PySequence_Contains(special_blocks, blockid)) {
                    /* t = textures.blockmap[blockid] */
                    t = PyList_GetItem(blockmap, state.block);
                } else {
                    PyObject *tmp;
                    
                    unsigned char ancilData = getArrayByte3D(blockdata_expanded, state.x, state.y, state.z);
                    if (state.block == 85) {
                        /* fence.  skip the generate_pseudo_ancildata for now */
                        continue;
                    }
                    
                    tmp = PyTuple_New(2);

                    Py_INCREF(blockid); /* because SetItem steals */
                    PyTuple_SetItem(tmp, 0, blockid);
                    PyTuple_SetItem(tmp, 1, PyInt_FromLong(ancilData));
                    
                    /* this is a borrowed reference. no need to decref */
                    t = PyDict_GetItem(specialblockmap, tmp);
                    Py_DECREF(tmp);
                }
                
                /* if we found a proper texture, render it! */
                if (t != NULL && t != Py_None)
                {
                    PyObject *src, *mask;
                    src = PyTuple_GetItem(t, 0);
                    mask = PyTuple_GetItem(t, 1);
                    
                    if (mask == Py_None)
                        mask = src;
                    
                    rendermode->draw(rm_data, &state, src, mask);
                }               
            }
            
            if (blockid) {
                Py_DECREF(blockid);
                blockid = NULL;
            }
        }
    } 

    /* free up the rendermode info */
    rendermode->finish(rm_data, &state);
    free(rm_data);
    
    Py_DECREF(blocks_py);
    Py_XDECREF(left_blocks_py);
    Py_XDECREF(right_blocks_py);
    Py_XDECREF(up_left_blocks_py);
    Py_XDECREF(up_right_blocks_py);

    return Py_BuildValue("i",2);
}
