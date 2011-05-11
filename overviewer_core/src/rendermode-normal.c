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

static int
rendermode_normal_start(void *data, RenderState *state) {
    PyObject *chunk_x_py, *chunk_y_py, *world, *use_biomes, *worlddir;
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    chunk_x_py = PyObject_GetAttrString(state->self, "chunkX");
    chunk_y_py = PyObject_GetAttrString(state->self, "chunkY");
    
    /* careful now -- C's % operator works differently from python's
       we can't just do x % 32 like we did before */
    self->chunk_x = PyInt_AsLong(chunk_x_py);
    self->chunk_y = PyInt_AsLong(chunk_y_py);
    
    while (self->chunk_x < 0)
        self->chunk_x += 32;
    while (self->chunk_y < 0)
        self->chunk_y += 32;
    
    self->chunk_x %= 32;
    self->chunk_y %= 32;
    
    /* fetch the biome data from textures.py, if needed */
    world = PyObject_GetAttrString(state->self, "world");
    worlddir = PyObject_GetAttrString(world, "worlddir");
    use_biomes = PyObject_GetAttrString(world, "useBiomeData");
    Py_DECREF(world);
    
    if (PyObject_IsTrue(use_biomes)) {
        PyObject *facemasks_py;
        
        self->biome_data = PyObject_CallMethod(state->textures, "getBiomeData", "OOO",
                                               worlddir, chunk_x_py, chunk_y_py);
        if (self->biome_data == Py_None) {
            self->biome_data = NULL;
            self->foliagecolor = NULL;
            self->grasscolor = NULL;

            self->leaf_texture = NULL;
            self->grass_texture = NULL;
            self->facemask_top = NULL;
        } else {

            self->foliagecolor = PyObject_GetAttrString(state->textures, "foliagecolor");
            self->grasscolor = PyObject_GetAttrString(state->textures, "grasscolor");

            self->leaf_texture = PyObject_GetAttrString(state->textures, "biome_leaf_texture");
            self->grass_texture = PyObject_GetAttrString(state->textures, "biome_grass_texture");

            facemasks_py = PyObject_GetAttrString(state->chunk, "facemasks");
            /* borrowed reference, needs to be incref'd if we keep it */
            self->facemask_top = PyTuple_GetItem(facemasks_py, 0);
            Py_INCREF(self->facemask_top);
            Py_DECREF(facemasks_py);
        }
    } else {
        self->biome_data = NULL;
        self->foliagecolor = NULL;
        self->grasscolor = NULL;
        
        self->leaf_texture = NULL;
        self->grass_texture = NULL;
        self->facemask_top = NULL;
    }
    
    Py_DECREF(use_biomes);
    Py_DECREF(worlddir);
    Py_DECREF(chunk_x_py);
    Py_DECREF(chunk_y_py);
    
    return 0;
}

static void
rendermode_normal_finish(void *data, RenderState *state) {
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    Py_XDECREF(self->biome_data);
    Py_XDECREF(self->foliagecolor);
    Py_XDECREF(self->grasscolor);
    Py_XDECREF(self->leaf_texture);
    Py_XDECREF(self->grass_texture);
    Py_XDECREF(self->facemask_top);
}

static int
rendermode_normal_occluded(void *data, RenderState *state) {
    int x = state->x, y = state->y, z = state->z;
    
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }

    return 0;
}

static void
rendermode_normal_draw(void *data, RenderState *state, PyObject *src, PyObject *mask) {
    RenderModeNormal *self = (RenderModeNormal *)data;
    
    /* first, check to see if we should use biome-compatible src, mask */
    if (self->biome_data && state->block == 18) {
        src = mask = self->leaf_texture;
    }
    
    /* draw the block! */
    alpha_over(state->img, src, mask, state->imgx, state->imgy, 0, 0);
    
    if (self->biome_data) {
        /* do the biome stuff! */
        unsigned int index;
        PyObject *color = NULL, *facemask = NULL;
        unsigned char r, g, b;
        
        index = ((self->chunk_y * 16) + state->y) * 16 * 32 + (self->chunk_x * 16) + state->x;
        index = big_endian_ushort(getArrayShort1D(self->biome_data, index));
        
        switch (state->block) {
        case 2:
            /* grass -- skip for snowgrass */
            if (state->z < 127 && getArrayByte3D(state->blocks, state->x, state->y, state->z+1) == 78)
                break;
            color = PySequence_GetItem(self->grasscolor, index);
            facemask = self->grass_texture;
            alpha_over(state->img, self->grass_texture, self->grass_texture, state->imgx, state->imgy, 0, 0);
            break;
        case 18:
            /* leaves */
            color = PySequence_GetItem(self->foliagecolor, index);
            facemask = mask;
            break;
        default:
            break;
        };
        
        if (color)
        {
            /* we've got work to do */
            
            r = PyInt_AsLong(PyTuple_GET_ITEM(color, 0));
            g = PyInt_AsLong(PyTuple_GET_ITEM(color, 1));
            b = PyInt_AsLong(PyTuple_GET_ITEM(color, 2));
            Py_DECREF(color);

            tint_with_mask(state->img, r, g, b, 255, facemask, state->imgx, state->imgy, 0, 0);
        }
    }


    /* Draw some edge lines! */
    // draw.line(((imgx+12,imgy+increment), (imgx+22,imgy+5+increment)), fill=(0,0,0), width=1)
    if (state->block == 44 || state->block == 78 || !is_transparent(state->block)) {
        Imaging img_i = imaging_python_to_c(state->img);
        unsigned char ink[] = {0,0,0,40};

        int increment=0;
        if (state->block == 44)  // half-step
            increment=6;
        else if ((state->block == 78) || (state->block == 93) || (state->block == 94)) // snow, redstone repeaters (on and off)
            increment=9;

        if ((state->x == 15) && (state->up_right_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_right_blocks, 0, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        } else if (state->x != 15) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x+1, state->y, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+1+increment, state->imgx+22+1, state->imgy+5+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx+12, state->imgy+increment, state->imgx+22+1, state->imgy+5+increment, &ink, 1);
            }
        }
        // if y != 0 and blocks[x,y-1,z] == 0

        // chunk boundries are annoying
        if ((state->y == 0) && (state->up_left_blocks != Py_None)) {
            unsigned char side_block = getArrayByte3D(state->up_left_blocks, state->x, 15, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        } else if (state->y != 0) {
            unsigned char side_block = getArrayByte3D(state->blocks, state->x, state->y-1, state->z);
            if (side_block != state->block && is_transparent(side_block)) {
                // draw.line(((imgx,imgy+6+increment), (imgx+12,imgy+increment)), fill=(0,0,0), width=1)
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+1+increment, state->imgx+12+1, state->imgy+1+increment, &ink, 1);
                ImagingDrawLine(img_i, state->imgx, state->imgy+6+increment, state->imgx+12+1, state->imgy+increment, &ink, 1);
            }
        }
    }
}

RenderModeInterface rendermode_normal = {
    "normal", "nothing special, just render the blocks",
    NULL,
    sizeof(RenderModeNormal),
    rendermode_normal_start,
    rendermode_normal_finish,
    rendermode_normal_occluded,
    rendermode_normal_draw,
};
