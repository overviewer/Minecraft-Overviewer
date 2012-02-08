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

#include "../overviewer.h"

typedef struct {
    /* biome data for the chunk */
    PyObject *biome_data;
    /* grasscolor and foliagecolor lookup tables */
    PyObject *grasscolor, *foliagecolor, *watercolor;
    /* biome-compatible grass/leaf textures */
    PyObject *grass_texture;
} PrimitiveBase;

static int
base_start(void *data, RenderState *state, PyObject *support) {
    PrimitiveBase *self = (PrimitiveBase *)data;
    
    /* biome-compliant grass mask (includes sides!) */
    self->grass_texture = PyObject_GetAttrString(state->textures, "biome_grass_texture");

    self->biome_data = PyObject_CallMethod(state->regionset, "get_biome_data", "ii", state->chunkx, state->chunkz);
    if (self->biome_data == NULL) {
        /* error while loading biome info, or no biomes at all */
        PyErr_Clear();
    } else {
        self->foliagecolor = PyObject_CallMethod(state->textures, "load_foliage_color", "");
        self->grasscolor = PyObject_CallMethod(state->textures, "load_grass_color", "");
        self->watercolor = PyObject_CallMethod(state->textures, "load_water_color", "");
    }
    
    return 0;
}

static void
base_finish(void *data, RenderState *state) {
    PrimitiveBase *self = (PrimitiveBase *)data;
    
    Py_XDECREF(self->biome_data);
    Py_XDECREF(self->foliagecolor);
    Py_XDECREF(self->grasscolor);
    Py_XDECREF(self->watercolor);
    Py_XDECREF(self->grass_texture);
}

static int
base_occluded(void *data, RenderState *state, int x, int y, int z) {
    if ( (x != 0) && (y != 15) && (z != 127) &&
         !render_mode_hidden(state->rendermode, x-1, y, z) &&
         !render_mode_hidden(state->rendermode, x, y, z+1) &&
         !render_mode_hidden(state->rendermode, x, y+1, z) &&
         !is_transparent(getArrayByte3D(state->blocks, x-1, y, z)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y, z+1)) &&
         !is_transparent(getArrayByte3D(state->blocks, x, y+1, z))) {
        return 1;
    }

    return 0;
}

static int
base_hidden(void *data, RenderState *state, int x, int y, int z) {
    PrimitiveBase *self = (PrimitiveBase *)data;
    
    return 0;
}

static void
base_draw(void *data, RenderState *state, PyObject *src, PyObject *mask, PyObject *mask_light) {
    PrimitiveBase *self = (PrimitiveBase *)data;

    /* draw the block! */
    alpha_over(state->img, src, mask, state->imgx, state->imgy, 0, 0);
    
    /* check for biome-compatible blocks
     *
     * NOTES for maintainers:
     *
     * To add a biome-compatible block, add an OR'd condition to this
     * following if block, a case to the first switch statement to handle when
     * biome info IS available, and another case to the second switch
     * statement for when biome info ISN'T available.
     *
     * Make sure that in textures.py, the generated textures are the
     * biome-compliant ones! The tinting is now all done here.
     */
    if (/* grass, but not snowgrass */
        (state->block == 2 && !(state->z < 127 && getArrayByte3D(state->blocks, state->x, state->y, state->z+1) == 78)) ||
        /* water */
        state->block == 8 || state->block == 9 ||
        /* leaves */
        state->block == 18 ||
        /* tallgrass, but not dead shrubs */
        (state->block == 31 && state->block_data != 0) ||
        /* pumpkin/melon stem, not fully grown. Fully grown stems
         * get constant brown color (see textures.py) */
        (((state->block == 104) || (state->block == 105)) && (state->block_data != 7)) ||
        /* vines */
        state->block == 106 ||
        /* lily pads */
        state->block == 111)
    {
        /* do the biome stuff! */
        PyObject *facemask = mask;
        unsigned char r, g, b;
        
        if (state->block == 2) {
            /* grass needs a special facemask */
            facemask = self->grass_texture;
        }
        
        if (self->biome_data) {
            /* we have data, so use it! */
            unsigned int index;
            PyObject *color = NULL;
            
            index = big_endian_ushort(getArrayShort2D(self->biome_data, state->x, state->y));
            
            switch (state->block) {
            case 2:
                /* grass */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 8:
            case 9:
                /* water */
                if (self->watercolor)
                {
                    color = PySequence_GetItem(self->watercolor, index);
                } else {
                    color = NULL;
                    facemask = NULL;
                }
                break;
            case 18:
                /* leaves */
                if (state->block_data != 2)
                {
                    /* not birch! */
                    color = PySequence_GetItem(self->foliagecolor, index);
                } else {
                    /* birch!
                       birch foliage color is flipped XY-ways */
                    unsigned int index_x = 255 - (index % 256);
                    unsigned int index_y = 255 - (index / 256);
                    index = index_y * 256 + index_x;
                    
                    color = PySequence_GetItem(self->foliagecolor, index);
                }
                break;
            case 31:
                /* tall grass */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 104:
                /* pumpkin stem */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 105:
                /* melon stem */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 106:
                /* vines */
                color = PySequence_GetItem(self->grasscolor, index);
                break;
            case 111:
                /* lily pads */
                color = PySequence_GetItem(self->grasscolor, index);
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
            }
        } else {
            if (state->block == 2 || state->block == 31 ||
                state->block == 104 || state->block == 105)
                /* grass and pumpkin/melon stems */
            {
                r = 115;
                g = 175;
                b = 71;
            }
            
            if (state->block == 8 || state->block == 9)
                /* water */
            {
                /* by default water is fine with nothing */
                facemask = NULL;
            }
            
            if (state->block == 18 || state->block == 106 || state->block == 111)
                /* leaves, vines and lyli pads */
            {
                r = 37;
                g = 118;
                b = 25;
            }
        }
        
        if (facemask)
            tint_with_mask(state->img, r, g, b, 255, facemask, state->imgx, state->imgy, 0, 0);
    }
}

RenderPrimitiveInterface primitive_base = {
    "base", sizeof(PrimitiveBase),
    base_start,
    base_finish,
    base_occluded,
    base_hidden,
    base_draw,
};
