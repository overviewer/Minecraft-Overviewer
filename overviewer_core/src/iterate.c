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

PyObject *init_chunk_render(PyObject *self, PyObject *args) {
   
    /* this function only needs to be called once, anything more should be
     * ignored */
    if (blockmap) {
        Py_RETURN_NONE;
    }

    textures = PyImport_ImportModule("overviewer_core.textures");
    /* ensure none of these pointers are NULL */    
    if ((!textures)) {
        return NULL;
    }

    chunk_mod = PyImport_ImportModule("overviewer_core.chunk");
    /* ensure none of these pointers are NULL */    
    if ((!chunk_mod)) {
        return NULL;
    }
    
    blockmap = PyObject_GetAttrString(textures, "blockmap");
    if (!blockmap)
        return NULL;
    special_blocks = PyObject_GetAttrString(textures, "special_blocks");
    if (!special_blocks)
        return NULL;
    specialblockmap = PyObject_GetAttrString(textures, "specialblockmap");
    if (!specialblockmap)
        return NULL;
    transparent_blocks = PyObject_GetAttrString(chunk_mod, "transparent_blocks");
    if (!transparent_blocks)
        return NULL;
    
    Py_RETURN_NONE;
}

int
is_transparent(unsigned char b) {
    PyObject *block = PyInt_FromLong(b);
    int ret = PySequence_Contains(transparent_blocks, block);
    Py_DECREF(block);
    return ret;

}


unsigned char
    check_adjacent_blocks(RenderState *state, int x,int y,int z, unsigned char blockid) {
        /*
         * Generates a pseudo ancillary data for blocks that depend of 
         * what are surrounded and don't have ancillary data. This 
         * function is through generate_pseudo_data.
         *
         * This uses a binary number of 4 digits to encode the info. 
         * The encode is:
         *
         * 0b1234:
         * Bit:   1   2   3   4
         * Side: +x  +y  -x  -y
         * Values: bit = 0 -> The corresponding side block has different blockid
         *         bit = 1 -> The corresponding side block has same blockid
         * Example: if the bit1 is 1 that means that there is a block with 
         * blockid in the side of the +x direction.
         */
        
        unsigned char pdata=0;
        
        if (state->x == 15) { /* +x direction */
            if (state->up_right_blocks != Py_None) { /* just in case we are in the end of the world */
                if (getArrayByte3D(state->up_right_blocks, 0, y, z) == blockid) {
                    pdata = pdata|(1 << 3);
                }
            }
        } else {
            if (getArrayByte3D(state->blocks, x + 1, y, z) == blockid) {
                pdata = pdata|(1 << 3);
            }
        }
        
        if (state->y == 15) { /* +y direction*/
            if (state->right_blocks != Py_None) {
                if (getArrayByte3D(state->right_blocks, x, 0, z) == blockid) {
                    pdata = pdata|(1 << 2);
                }
            }
        } else {
            if (getArrayByte3D(state->blocks, x, y + 1, z) == blockid) {
                pdata = pdata|(1 << 2);
            }
        }
        
        if (state->x == 0) { /* -x direction*/
            if (state->left_blocks != Py_None) {
                if (getArrayByte3D(state->left_blocks, 15, y, z) == blockid) {
                    pdata = pdata|(1 << 1);
                }
            }
        } else {
            if (getArrayByte3D(state->blocks, x - 1, y, z) == blockid) {
                pdata = pdata|(1 << 1);
            }
        }
        
        if (state->y == 0) { /* -y direction */
            if (state->up_left_blocks != Py_None) {
                if (getArrayByte3D(state->up_left_blocks, x, 15, z) == blockid) {
                    pdata = pdata|(1 << 0);
                }
            }
        } else {
            if (getArrayByte3D(state->blocks, x, y - 1, z) == blockid) {
                pdata = pdata|(1 << 0);
            }
        }

        return pdata;
}


unsigned char
generate_pseudo_data(RenderState *state, unsigned char ancilData) {
    /*
     * Generates a fake ancillary data for blocks that are drawn 
     * depending on what are surrounded.
     */
    int x = state->x, y = state->y, z = state->z;
    unsigned char data = 0;
    
    if (state->block == 2) { /* grass */
        /* return 0x10 if grass is covered in snow */
        if (z < 127 && getArrayByte3D(state->blocks, x, y, z+1) == 78)
            return 0x10;
        return ancilData;
    } else if (state->block == 9) { /* water */
        /* an aditional bit for top is added to the 4 bits of check_adjacent_blocks */
        if (ancilData == 0) { /* static water */
            if ((z != 127) && (getArrayByte3D(state->blocks, x, y, z+1) == 9)) {
                data = 0;
            } else { 
                data = 16;
            }
            return data; /* = 0b10000 */
        } else if ((ancilData > 0) && (ancilData < 8)) { /* flowing water */
            data = (check_adjacent_blocks(state, x, y, z, state->block) ^ 0x0f) | 0x10;
            return data;
        } else if (ancilData >= 8) { /* falling water */
            data = (check_adjacent_blocks(state, x, y, z, state->block) ^ 0x0f);
            return data;
        }
    } else if ((state->block == 20) || (state->block == 79)) { /* glass and ice */
        /* an aditional bit for top is added to the 4 bits of check_adjacent_blocks */
        if ((z != 127) && (getArrayByte3D(state->blocks, x, y, z+1) == 20)) {
            data = 0;
        } else { 
            data = 16;
        }
        data = (check_adjacent_blocks(state, x, y, z, state->block) ^ 0x0f) | data;
        return data;
    } else if (state->block == 85) { /* fences */
        return check_adjacent_blocks(state, x, y, z, state->block);

    } else if (state->block == 55) { /* redstone */
        /* three addiotional bit are added, one for on/off state, and
         * another two for going-up redstone wire in the same block
         * (connection with the level z+1) */
        unsigned char above_level_data = 0, same_level_data = 0, below_level_data = 0, possibly_connected = 0, final_data = 0;

        /* check for air in z+1, no air = no connection with upper level */        
        if ((z != 127) && (getArrayByte3D(state->left_blocks, x, y, z) == 0)) { 
            above_level_data = check_adjacent_blocks(state, x, y, z + 1, state->block);
        }   /* else above_level_data = 0 */
        
        /* check connection with same level */
        same_level_data = check_adjacent_blocks(state, x, y, z, 55);
        
        /* check the posibility of connection with z-1 level, check for air */
        possibly_connected = check_adjacent_blocks(state, x, y, z, 0);
        
        /* check connection with z-1 level */
        if (z != 0) {
            below_level_data = check_adjacent_blocks(state, x, y, z - 1, state->block);
        } /* else below_level_data = 0 */
        
        final_data = above_level_data | same_level_data | (below_level_data & possibly_connected);
        
        /* add the three bits */
        if (ancilData > 0) { /* powered redstone wire */
            final_data = final_data | 0x40;
        }
        if ((above_level_data & 0x01)) { /* draw top left going up redstonewire */
            final_data = final_data | 0x20;
        }
        if ((above_level_data & 0x08)) { /* draw top right going up redstonewire */
            final_data = final_data | 0x10;
        }
        return final_data;

    } else if (state-> block == 54) { /* chests */
        /* the top 2 bits are used to store the type of chest
         * (single or double), the 2 bottom bits are used for 
         * orientation, look textures.py for more information. */

        /* if placed alone chests always face west, return 0 to make a 
         * chest facing west */
        unsigned char chest_data = 0, air_data = 0, final_data = 0;

        /* search for chests */
        chest_data = check_adjacent_blocks(state, x, y, z, 54);

        /* search for air */
        air_data = check_adjacent_blocks(state, x, y, z, 0);

        if (chest_data == 1) { /* another chest in the east */
            final_data = final_data | 0x8; /* only can face to north or south */
            if ( (air_data & 0x2) == 2 ) {
                final_data = final_data | 0x1; /* facing north */
            } else {
                final_data = final_data | 0x3; /* facing south */
            }

        } else if (chest_data == 2) { /* in the north */
            final_data = final_data | 0x4; /* only can face to east or west */
            if ( !((air_data & 0x4) == 4) ) { /* 0 = west */
                final_data = final_data | 0x2; /* facing east */
            }

        } else if (chest_data == 4) { /*in the west */
            final_data = final_data | 0x4;
            if ( (air_data & 0x2) == 2 ) {
                final_data = final_data | 0x1; /* facing north */
            } else {
                final_data = final_data | 0x3; /* facing south */
            }

        } else if (chest_data == 8) { /*in the south */
            final_data = final_data | 0x8;
            if ( !((air_data & 0x4) == 4) ) {
                final_data = final_data | 0x2; /* facing east */
            }

        } else if (chest_data == 0) {
            /* Single chest, determine the orientation */
            if ( ((air_data & 0x8) == 0) && ((air_data & 0x2) == 2)  ) { /* block in +x and no block in -x */
                final_data = final_data | 0x1; /* facing north */

            } else if ( ((air_data & 0x2) == 0) && ((air_data & 0x8) == 8)) {
                final_data = final_data | 0x3;

            } else if ( ((air_data & 0x4) == 0) && ((air_data & 0x1) == 1)) {
                final_data = final_data | 0x2;
            } /* else, facing west, value = 0 */

        } else {
            /* more than one adjacent chests! render as normal chest */
            return 0;
        }

        return final_data;

    /* fences, iron bars and glass panes */
    } else if ((state->block == 90) || (state->block == 101) ||
               (state->block == 102)) {
        return check_adjacent_blocks(state, x, y, z, state->block);
    }


    return 0;

}


/* TODO triple check this to make sure reference counting is correct */
PyObject*
chunk_render(PyObject *self, PyObject *args) {
    RenderState state;
    PyObject *rendermode_py;

    int xoff, yoff;
    
    PyObject *imgsize, *imgsize0_py, *imgsize1_py;
    int imgsize0, imgsize1;
    
    PyObject *blocks_py;
    PyObject *left_blocks_py;
    PyObject *right_blocks_py;
    PyObject *up_left_blocks_py;
    PyObject *up_right_blocks_py;

    RenderMode *rendermode;

    PyObject *t = NULL;
    
    if (!PyArg_ParseTuple(args, "OOiiO",  &state.self, &state.img, &xoff, &yoff, &state.blockdata_expanded))
        return NULL;
    
    /* fill in important modules */
    state.textures = textures;
    state.chunk = chunk_mod;
    
    /* set up the render mode */
    rendermode_py = PyObject_GetAttrString(state.self, "rendermode");
    state.rendermode = rendermode = render_mode_create(PyString_AsString(rendermode_py), &state);
    Py_DECREF(rendermode_py);
    if (rendermode == NULL) {
        return NULL;
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
    
    /* set up the random number generator again for each chunk
       so tallgrass is in the same place, no matter what mode is used */
    srand(1);
    
    for (state.x = 15; state.x > -1; state.x--) {
        for (state.y = 0; state.y < 16; state.y++) {
            PyObject *blockid = NULL;
            
            /* set up the render coordinates */
            state.imgx = xoff + state.x*12 + state.y*12;
            /* 128*12 -- offset for z direction, 15*6 -- offset for x */
            state.imgy = yoff - state.x*6 + state.y*6 + 128*12 + 15*6;
            
            for (state.z = 0; state.z < 128; state.z++) {
                state.imgy -= 12;
		
                /* get blockid */
                state.block = getArrayByte3D(blocks_py, state.x, state.y, state.z);
                if (state.block == 0 || render_mode_hidden(rendermode, state.x, state.y, state.z)) {
                    continue;
                }
                
                /* make sure we're rendering inside the image boundaries */
                if ((state.imgx >= imgsize0 + 24) || (state.imgx <= -24)) {
                    continue;
                }
                if ((state.imgy >= imgsize1 + 24) || (state.imgy <= -24)) {
                    continue;
                }

                
                /* decref'd on replacement *and* at the end of the z for block */
                if (blockid) {
                    Py_DECREF(blockid);
                }
                blockid = PyInt_FromLong(state.block);

                // check for occlusion
                if (render_mode_occluded(rendermode, state.x, state.y, state.z)) {
                    continue;
                }
                
                // everything stored here will be a borrowed ref

                /* get the texture and mask from block type / ancil. data */
                if (!PySequence_Contains(special_blocks, blockid)) {
                    /* t = textures.blockmap[blockid] */
                    t = PyList_GetItem(blockmap, state.block);
                } else {
                    PyObject *tmp;
                    
                    unsigned char ancilData = getArrayByte3D(state.blockdata_expanded, state.x, state.y, state.z);
                    state.block_data = ancilData;
                    /* block that need pseudo ancildata:
                     * grass, water, glass, chest, restone wire,
                     * ice, fence, portal, iron bars, glass panes */
                    if ((state.block ==  2) || (state.block ==  9) || 
                        (state.block == 20) || (state.block == 54) || 
                        (state.block == 55) || (state.block == 79) ||
                        (state.block == 85) || (state.block == 90) ||
                        (state.block == 101) || (state.block == 102)) {
                        ancilData = generate_pseudo_data(&state, ancilData);
                        state.block_pdata = ancilData;
                    } else {
                        state.block_pdata = 0;
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
                    PyObject *src, *mask, *mask_light;
                    int randx = 0, randy = 0;
                    src = PyTuple_GetItem(t, 0);
                    mask = PyTuple_GetItem(t, 1);
                    mask_light = PyTuple_GetItem(t, 2);

                    if (mask == Py_None)
                        mask = src;

                    if (state.block == 31) {
                        /* add a random offset to the postion of the tall grass to make it more wild */
                        randx = rand() % 6 + 1 - 3;
                        randy = rand() % 6 + 1 - 3;
                        state.imgx += randx;
                        state.imgy += randy;
                    }
                    
                    render_mode_draw(rendermode, src, mask, mask_light);
                    
                    if (state.block == 31) {
                        /* undo the random offsets */
                        state.imgx -= randx;
                        state.imgy -= randy;
                    }
                }               
            }
            
            if (blockid) {
                Py_DECREF(blockid);
                blockid = NULL;
            }
        }
    }

    /* free up the rendermode info */
    render_mode_destroy(rendermode);
    
    Py_DECREF(blocks_py);
    Py_XDECREF(left_blocks_py);
    Py_XDECREF(right_blocks_py);
    Py_XDECREF(up_left_blocks_py);
    Py_XDECREF(up_right_blocks_py);

    return Py_BuildValue("i",2);
}
