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

unsigned int max_blockid = 0;
unsigned int max_data = 0;
unsigned char *block_properties = NULL;

static PyObject *known_blocks = NULL;
static PyObject *transparent_blocks = NULL;
static PyObject *solid_blocks = NULL;
static PyObject *fluid_blocks = NULL;
static PyObject *nospawn_blocks = NULL;
static PyObject *nodata_blocks = NULL;

PyObject *init_chunk_render(void) {
   
    PyObject *tmp = NULL;
    unsigned int i;
    
    /* this function only needs to be called once, anything more should be
     * ignored */
    if (textures) {
        Py_RETURN_NONE;
    }

    textures = PyImport_ImportModule("overviewer_core.textures");
    /* ensure none of these pointers are NULL */    
    if ((!textures)) {
        return NULL;
    }
    
    tmp = PyObject_GetAttrString(textures, "max_blockid");
    if (!tmp)
        return NULL;
    max_blockid = PyInt_AsLong(tmp);
    Py_DECREF(tmp);

    tmp = PyObject_GetAttrString(textures, "max_data");
    if (!tmp)
        return NULL;
    max_data = PyInt_AsLong(tmp);
    Py_DECREF(tmp);

    /* assemble the property table */
    known_blocks = PyObject_GetAttrString(textures, "known_blocks");
    if (!known_blocks)
        return NULL;
    transparent_blocks = PyObject_GetAttrString(textures, "transparent_blocks");
    if (!transparent_blocks)
        return NULL;
    solid_blocks = PyObject_GetAttrString(textures, "solid_blocks");
    if (!solid_blocks)
        return NULL;
    fluid_blocks = PyObject_GetAttrString(textures, "fluid_blocks");
    if (!fluid_blocks)
        return NULL;
    nospawn_blocks = PyObject_GetAttrString(textures, "nospawn_blocks");
    if (!nospawn_blocks)
        return NULL;
    nodata_blocks = PyObject_GetAttrString(textures, "nodata_blocks");
    if (!nodata_blocks)
        return NULL;
    
    block_properties = calloc(max_blockid, sizeof(unsigned char));
    for (i = 0; i < max_blockid; i++) {
        PyObject *block = PyInt_FromLong(i);
        
        if (PySequence_Contains(known_blocks, block))
            block_properties[i] |= 1 << KNOWN;
        if (PySequence_Contains(transparent_blocks, block))
            block_properties[i] |= 1 << TRANSPARENT;
        if (PySequence_Contains(solid_blocks, block))
            block_properties[i] |= 1 << SOLID;
        if (PySequence_Contains(fluid_blocks, block))
            block_properties[i] |= 1 << FLUID;
        if (PySequence_Contains(nospawn_blocks, block))
            block_properties[i] |= 1 << NOSPAWN;
        if (PySequence_Contains(nodata_blocks, block))
            block_properties[i] |= 1 << NODATA;
        
        Py_DECREF(block);
    }
    
    Py_RETURN_NONE;
}

/* helper for load_chunk, loads a section into a chunk */
static inline void load_chunk_section(ChunkData *dest, int i, PyObject *section) {
    dest->sections[i].blocks = (PyArrayObject*) PyDict_GetItemString(section, "Blocks");
    dest->sections[i].data = (PyArrayObject*) PyDict_GetItemString(section, "Data");
    dest->sections[i].skylight = (PyArrayObject*) PyDict_GetItemString(section, "SkyLight");
    dest->sections[i].blocklight = (PyArrayObject*) PyDict_GetItemString(section, "BlockLight");
    Py_INCREF(dest->sections[i].blocks);
    Py_INCREF(dest->sections[i].data);
    Py_INCREF(dest->sections[i].skylight);
    Py_INCREF(dest->sections[i].blocklight);
}

/* loads the given chunk into the chunks[] array in the state
 * returns true on error
 *
 * if required is true, failure to load the chunk will raise a python
 * exception and return true.
 */
int load_chunk(RenderState* state, int x, int z, unsigned char required) {
    ChunkData *dest = &(state->chunks[1 + x][1 + z]);
    int i;
    PyObject *chunk = NULL;
    PyObject *sections = NULL;
    
    if (dest->loaded)
        return 0;
    
    /* set up reasonable defaults */
    dest->biomes = NULL;
    for (i = 0; i < SECTIONS_PER_CHUNK; i++)
    {
        dest->sections[i].blocks = NULL;
        dest->sections[i].data = NULL;
        dest->sections[i].skylight = NULL;
        dest->sections[i].blocklight = NULL;
    }
    dest->loaded = 1;
    
    x += state->chunkx;
    z += state->chunkz;

    chunk = PyObject_CallMethod(state->regionset, "get_chunk", "ii", x, z);
    if (chunk == NULL) {
        // An exception is already set. RegionSet.get_chunk sets
        // ChunkDoesntExist
        if (!required) {
            PyErr_Clear();
        }
        return 1;
    }

    sections = PyDict_GetItemString(chunk, "Sections");
    if (sections) {
        sections = PySequence_Fast(sections, "Sections tag was not a list!");
    }
    if (sections == NULL) {
        // exception set, again
        Py_DECREF(chunk);
        if (!required) {
            PyErr_Clear();
        }
        return 1;
    }
    
    dest->biomes = (PyArrayObject*) PyDict_GetItemString(chunk, "Biomes");
    Py_INCREF(dest->biomes);
    
    for (i = 0; i < PySequence_Fast_GET_SIZE(sections); i++) {
        PyObject *ycoord = NULL;
        int sectiony = 0;
        PyObject *section = PySequence_Fast_GET_ITEM(sections, i);
        ycoord = PyDict_GetItemString(section, "Y");
        if (!ycoord)
            continue;
        
        sectiony = PyInt_AsLong(ycoord);
        if (sectiony >= 0 && sectiony < SECTIONS_PER_CHUNK)
            load_chunk_section(dest, sectiony, section);
    }
    Py_DECREF(sections);
    Py_DECREF(chunk);
    
    return 0;
}

/* helper to unload all loaded chunks */
static void
unload_all_chunks(RenderState *state) {
    unsigned int i, j, k;
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            if (state->chunks[i][j].loaded) {
                Py_XDECREF(state->chunks[i][j].biomes);
                for (k = 0; k < SECTIONS_PER_CHUNK; k++) {
                    Py_XDECREF(state->chunks[i][j].sections[k].blocks);
                    Py_XDECREF(state->chunks[i][j].sections[k].data);
                    Py_XDECREF(state->chunks[i][j].sections[k].skylight);
                    Py_XDECREF(state->chunks[i][j].sections[k].blocklight);
                }
                state->chunks[i][j].loaded = 0;
            }
        }
    }
}

unsigned short
check_adjacent_blocks(RenderState *state, int x,int y,int z, unsigned short blockid) {
    /*
     * Generates a pseudo ancillary data for blocks that depend of 
     * what are surrounded and don't have ancillary data. This 
     * function is used through generate_pseudo_data.
     *
     * This uses a binary number of 4 digits to encode the info:
     *
     * 0b1234:
     * Bit:   1   2   3   4
     * Side: +x  +z  -x  -z
     * Values: bit = 0 -> The corresponding side block has different blockid
     *         bit = 1 -> The corresponding side block has same blockid
     * Example: if the bit1 is 1 that means that there is a block with 
     * blockid in the side of the +x direction.
     */
        
    unsigned char pdata=0;
        
    if (get_data(state, BLOCKS, x + 1, y, z) == blockid) {
        pdata = pdata|(1 << 3);
    }        
    if (get_data(state, BLOCKS, x, y, z + 1) == blockid) {
        pdata = pdata|(1 << 2);
    }
    if (get_data(state, BLOCKS, x - 1, y, z) == blockid) {
        pdata = pdata|(1 << 1);
    }
    if (get_data(state, BLOCKS, x, y, z - 1) == blockid) {
        pdata = pdata|(1 << 0);
    }
    
    return pdata;
}


static int
is_stairs(int block) {
    /*
     * Determines if a block is stairs of any material
     */
    switch (block) {
        case 53: /* oak wood stairs */
        case 67: /* cobblestone stairs */
        case 108: /* brick stairs */
        case 109: /* stone brick stairs */
        case 114: /* nether brick stairs */
        case 128: /* sandstone stairs */
        case 134: /* spruce wood stairs */
        case 135: /* birch wood stairs */
        case 136: /* jungle wood stairs */
        case 156: /* quartz stairs */
        case 163: /* acacia wood stairs */
        case 164: /* dark wood stairs */
        case 180: /* red sandstone stairs */
        case 203: /* purpur stairs */
            return 1;
    }
    return 0;
}

unsigned short
generate_pseudo_data(RenderState *state, unsigned short ancilData) {
    /*
     * Generates a fake ancillary data for blocks that are drawn 
     * depending on what are surrounded.
     */
    int x = state->x, y = state->y, z = state->z;
    unsigned short data = 0;

    if (state->block == 2) { /* grass */
        /* return 0x10 if grass is covered in snow */
        if (get_data(state, BLOCKS, x, y+1, z) == 78)
            return 0x10;
        return ancilData;
    } else if (state->block == 9) { /* water */
        /* an aditional bit for top is added to the 4 bits of check_adjacent_blocks */
        if (ancilData == 0) { /* static water */
            if (get_data(state, BLOCKS, x, y+1, z) == 9) {
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
    } else if ((state->block == 20) || (state->block == 79) || (state->block == 95)) { /* glass and ice and stained glass*/
        /* an aditional bit for top is added to the 4 bits of check_adjacent_blocks
         * Note that stained glass encodes 16 colors using 4 bits.  this pushes us over the 8-bits of an unsigned char, 
         * forcing us to use an unsigned short to hold 16 bits of pseudo ancil data
         * */
        if ((get_data(state, BLOCKS, x, y+1, z) == 20) || (get_data(state, BLOCKS, x, y+1, z) == 95)) {
            data = 0;
        } else { 
            data = 16;
        }
        data = (check_adjacent_blocks(state, x, y, z, state->block) ^ 0x0f) | data;
        return (data << 4) | (ancilData & 0x0f);
    } else if ((state->block == 85) || (state->block == 188) || (state->block == 189) ||
            (state->block == 190) || (state->block == 191) || (state->block == 192)) { /* fences */
        /* check for fences AND fence gates */
        return check_adjacent_blocks(state, x, y, z, state->block) | check_adjacent_blocks(state, x, y, z, 107) |
                check_adjacent_blocks(state, x, y, z, 183) | check_adjacent_blocks(state, x, y, z, 184) | check_adjacent_blocks(state, x, y, z, 185) |
                check_adjacent_blocks(state, x, y, z, 186) | check_adjacent_blocks(state, x, y, z, 187);

    } else if (state->block == 55) { /* redstone */
        /* three addiotional bit are added, one for on/off state, and
         * another two for going-up redstone wire in the same block
         * (connection with the level y+1) */
        unsigned char above_level_data = 0, same_level_data = 0, below_level_data = 0, possibly_connected = 0, final_data = 0;

        /* check for air in y+1, no air = no connection with upper level */        
        if (get_data(state, BLOCKS, x, y+1, z) == 0) { 
            above_level_data = check_adjacent_blocks(state, x, y+1, z, state->block);
        }   /* else above_level_data = 0 */
        
        /* check connection with same level (other redstone and trapped chests */
        same_level_data = check_adjacent_blocks(state, x, y, z, 55) | check_adjacent_blocks(state, x, y, z, 146);
        
        /* check the posibility of connection with y-1 level, check for air */
        possibly_connected = check_adjacent_blocks(state, x, y, z, 0);
        
        /* check connection with y-1 level */
        below_level_data = check_adjacent_blocks(state, x, y-1, z, state->block);
        
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

    } else if (state->block == 54 || state->block == 146) { /* normal chests and trapped chests */
        /* Orientation is given by ancilData, pseudo data needed to 
         * choose from single or double chest and the correct half of
         * the chest. */
         
         /* Add two bits to ancilData to store single or double chest 
          * and which half of the chest it is: bit 0x10 = second half
          *                                    bit 0x8 = first half */

        unsigned char chest_data = 0, final_data = 0;

        /* search for adjacent chests of the same type */
        chest_data = check_adjacent_blocks(state, x, y, z, state->block);

        if (chest_data == 1) { /* another chest in the upper-left */
            final_data = final_data | 0x10 | ancilData;

        } else if (chest_data == 2) { /* in the bottom-left */
            final_data = final_data | 0x8 | ancilData;

        } else if (chest_data == 4) { /*in the bottom-right */
            final_data = final_data | 0x8 | ancilData;

        } else if (chest_data == 8) { /*in the upper-right */
            final_data = final_data | 0x10 | ancilData;

        } else if (chest_data == 0) {
            /* Single chest, determine the orientation */
            final_data = ancilData;

        } else {
            /* more than one adjacent chests! That shouldn't be 
             * possible! render as normal chest */
            return 0;
        }
        return final_data;

    } else if ((state->block == 101) || (state->block == 102) || (state->block == 160)) {
        /* iron bars and glass panes:
         * they seem to stick to almost everything but air,
         * not sure yet! Still a TODO! */
        /* return check adjacent blocks with air, bit inverted */
        // shift up 4 bits because the lower 4 bits encode color
        data = (check_adjacent_blocks(state, x, y, z, 0) ^ 0x0f);
        return (data << 4) | (ancilData & 0xf);

    } else if ((state->block == 90) || (state->block == 113)) {
        /* portal and nether brick fences */
        return check_adjacent_blocks(state, x, y, z, state->block);

    } else if ((state->block == 64) || (state->block == 71) || (state->block == 193) ||
            (state->block == 194) || (state->block == 195) || (state->block == 196) ||
            (state->block ==197)) {
        /* use bottom block data format plus one bit for top/down
         * block (0x8) and one bit for hinge position (0x10)
         */
        unsigned char data = 0;
        if ((ancilData & 0x8) == 0x8) {
            /* top door block */
            unsigned char b_data = get_data(state, DATA, x, y-1, z);
            if ((ancilData & 0x1) == 0x1) {
                /* hinge on the left */
                data = b_data | 0x8 | 0x10;
            } else {
                data = b_data | 0x8;
            }
        } else {
            /* bottom door block */
            unsigned char t_data = get_data(state, DATA, x, y+1, z);
            if ((t_data & 0x1) == 0x1) {
                /* hinge on the left */
                data = ancilData | 0x10;
            } else {
                data = ancilData;
            }
        
        }
        return data;
    } else if (state->block == 139) { /* cobblestone and mossy cobbleston wall  */
        /* check for walls and add one bit with the type of wall (mossy or cobblestone)*/
        if (ancilData == 0x1) {
            return check_adjacent_blocks(state, x, y, z, state->block) | 0x10;
        } else {
            return check_adjacent_blocks(state, x, y, z, state->block);
        }
    } else if (state->block == 111) { /* lilypads */
        int wx,wz,wy,rotation;
        long pr;
        /* calculate the global block coordinates of this position */
        wx = (state->chunkx * 16) + x;
        wz = (state->chunkz * 16) + z;
        wy = (state->chunky * 16) + y;
        /* lilypads orientation is obtained with these magic numbers */
        /* magic numbers obtained from: */
        /* http://llbit.se/?p=1537 */
        pr = (wx * 3129871) ^ (wz * 116129781) ^ (wy);
        pr = pr * pr * 42317861 + pr * 11;
        rotation = 3 & (pr >> 16);
        return rotation;
    } else if (is_stairs(state->block)) { /* stairs */
        /* 4 ancillary bits will be added to indicate which quarters of the block contain the 
         * upper step. Regular stairs will have 2 bits set & corner stairs will have 1 or 3.
         *     Southwest quarter is part of the upper step - 0x40
         *    / Southeast " - 0x20
         *    |/ Northeast " - 0x10
         *    ||/ Northwest " - 0x8
         *    |||/ flip upside down (Minecraft)
         *    ||||/ has North/South alignment (Minecraft)
         *    |||||/ ascends North or West, not South or East (Minecraft)
         *    ||||||/
         *  0b0011011 = Stair ascending north, upside up, with both north quarters filled
         */

        /* keep track of whether neighbors are stairs, and their data */
        unsigned char stairs_base[8];
        unsigned char neigh_base[8];
        unsigned char *stairs = stairs_base;
        unsigned char *neigh = neigh_base;

        /* amount to rotate/roll to get to east, west, south, north */
        size_t rotations[] = {0,2,3,1};

        /* masks for the filled (ridge) stair quarters: */
        /* Example: the ridge for an east-ascending stair are the two east quarters */
        /*                  ascending: east  west south north */
        unsigned char ridge_mask[] = { 0x30, 0x48, 0x60, 0x18 };

        /* masks for the open (trench) stair quarters: */
        unsigned char trench_mask[] = { 0x48, 0x30, 0x18, 0x60 };

        /* boat analogy! up the stairs is toward the bow of the boat */
        /* masks for port and starboard, i.e. left and right sides while ascending: */
        unsigned char port_mask[] = { 0x18, 0x60, 0x30, 0x48 };
        unsigned char starboard_mask[] = { 0x60, 0x18, 0x48, 0x30 };

        /* we may need to lock some quarters into place depending on neighbors */
        unsigned char lock_mask = 0;

        unsigned char repair_rot[] = { 0, 1, 2, 3,  2, 3, 1, 0,  1, 0, 3, 2,  3, 2, 0, 1 };

        /* need to get northdirection of the render */
        /* TODO: get this just once? store in state? */
        PyObject *texrot;
        int northdir;
        texrot = PyObject_GetAttrString(state->textures, "rotation");
        northdir = PyInt_AsLong(texrot);

        /* fix the rotation value for different northdirections */
        #define FIX_ROT(x) (((x) & ~0x3) | repair_rot[((x) & 0x3) | (northdir << 2)])
        ancilData = FIX_ROT(ancilData);

        /* fill the ancillary bits assuming normal stairs with no corner yet */
        ancilData |= ridge_mask[ancilData & 0x3];

        /* get block & data for neighbors in this order: east, north, west, south */
        /* so we can rotate things easily */
        stairs[0] = stairs[4] = is_stairs(get_data(state, BLOCKS, x+1, y, z));
        stairs[1] = stairs[5] = is_stairs(get_data(state, BLOCKS, x, y, z-1));
        stairs[2] = stairs[6] = is_stairs(get_data(state, BLOCKS, x-1, y, z));
        stairs[3] = stairs[7] = is_stairs(get_data(state, BLOCKS, x, y, z+1));
        neigh[0] = neigh[4] = FIX_ROT(get_data(state, DATA, x+1, y, z));
        neigh[1] = neigh[5] = FIX_ROT(get_data(state, DATA, x, y, z-1));
        neigh[2] = neigh[6] = FIX_ROT(get_data(state, DATA, x-1, y, z));
        neigh[3] = neigh[7] = FIX_ROT(get_data(state, DATA, x, y, z+1));

        #undef FIX_ROT

        /* Rotate the neighbors so we only have to worry about one orientation
         * No matter which way the boat is facing, the the neighbors will be:
         *   0: bow
         *   1: port
         *   2: stern
         *   3: starboard */
        stairs += rotations[ancilData & 0x3];
        neigh += rotations[ancilData & 0x3];

        /* Matching neighbor stairs to the sides should prevent cornering on that side */
        /* If found, set bits in lock_mask to lock the current quarters as they are */
        if (stairs[1] && (neigh[1] & 0x7) == (ancilData & 0x7)) {
            /* Neighbor on port side is stairs of the same orientation as me */
            /* Do NOT allow changing quarters on the port side */
            lock_mask |= port_mask[ancilData & 0x3];
        }
        if (stairs[3] && (neigh[3] & 0x7) == (ancilData & 0x7)) {
            /* Neighbor on starboard side is stairs of the same orientation as me */
            /* Do NOT allow changing quarters on the starboard side */
            lock_mask |= starboard_mask[ancilData & 0x3];
        }

        /* Make corner stairs -- prefer outside corners like Minecraft */
        if (stairs[0] && (neigh[0] & 0x4) == (ancilData & 0x4)) {
            /* neighbor at bow is stairs with same flip */
            if ((neigh[0] & 0x2) != (ancilData & 0x2)) {
                /* neighbor is perpendicular, cut a trench, but not where locked */
                ancilData &= ~trench_mask[neigh[0] & 0x3] | lock_mask;
            }
        } else if (stairs[2] && (neigh[2] & 0x4) == (ancilData & 0x4)) {
            /* neighbor at stern is stairs with same flip */
            if ((neigh[2] & 0x2) != (ancilData & 0x2)) {
                /* neighbor is perpendicular, add a ridge, but not where locked */
                ancilData |= ridge_mask[neigh[2] & 0x3] & ~lock_mask;
            }
        }

        return ancilData;
    } else if (state->block == 175) { /* doublePlants */
        /* use bottom block data format plus one bit for top
         * block (0x8)
         */
        if( get_data(state, BLOCKS, x, y-1, z) == 175 ) {
            data = get_data(state, DATA, x, y-1, z) | 0x8;
        } else {
            data = ancilData;
        }

        return data;
    }

    return 0;
}


/* TODO triple check this to make sure reference counting is correct */
PyObject*
chunk_render(PyObject *self, PyObject *args) {
    RenderState state;
    PyObject *modeobj;
    PyObject *blockmap;

    int xoff, yoff;
    
    PyObject *imgsize, *imgsize0_py, *imgsize1_py;
    int imgsize0, imgsize1;
    
    PyArrayObject *blocks_py;
    PyArrayObject *left_blocks_py;
    PyArrayObject *right_blocks_py;
    PyArrayObject *up_left_blocks_py;
    PyArrayObject *up_right_blocks_py;

    RenderMode *rendermode;
    
    int i, j;

    PyObject *t = NULL;
    
    if (!PyArg_ParseTuple(args, "OOiiiOiiOO",  &state.world, &state.regionset, &state.chunkx, &state.chunky, &state.chunkz, &state.img, &xoff, &yoff, &modeobj, &state.textures))
        return NULL;
    
    /* set up the render mode */
    state.rendermode = rendermode = render_mode_create(modeobj, &state);
    if (rendermode == NULL) {
        return NULL; // note that render_mode_create will
                     // set PyErr.  No need to set it here
    }

    /* get the blockmap from the textures object */
    blockmap = PyObject_GetAttrString(state.textures, "blockmap");
    if (blockmap == NULL) {
        render_mode_destroy(rendermode);
        return NULL;
    }
    if (blockmap == Py_None) {
        render_mode_destroy(rendermode);
        PyErr_SetString(PyExc_RuntimeError, "you must call Textures.generate()");
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
    
    /* set all block data to unloaded */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            state.chunks[i][j].loaded = 0;
        }
    }
    
    /* get the block data for the center column, erroring out if needed */
    if (load_chunk(&state, 0, 0, 1)) {
        render_mode_destroy(rendermode);
        Py_DECREF(blockmap);
        return NULL;
    }
    if (state.chunks[1][1].sections[state.chunky].blocks == NULL) {
        /* this section doesn't exist, let's skeddadle */
        render_mode_destroy(rendermode);
        Py_DECREF(blockmap);
        unload_all_chunks(&state);
        Py_RETURN_NONE;
    }
    
    /* set blocks_py, state.blocks, and state.blockdatas as convenience */
    blocks_py = state.blocks = state.chunks[1][1].sections[state.chunky].blocks;
    state.blockdatas = state.chunks[1][1].sections[state.chunky].data;

    /* set up the random number generator again for each chunk
       so tallgrass is in the same place, no matter what mode is used */
    srand(1);
    
    for (state.x = 15; state.x > -1; state.x--) {
        for (state.z = 0; state.z < 16; state.z++) {

            /* set up the render coordinates */
            state.imgx = xoff + state.x*12 + state.z*12;
            /* 16*12 -- offset for y direction, 15*6 -- offset for x */
            state.imgy = yoff - state.x*6 + state.z*6 + 16*12 + 15*6;
            
            for (state.y = 0; state.y < 16; state.y++) {
                unsigned short ancilData;
                
                state.imgy -= 12;
		
                /* get blockid */
                state.block = getArrayShort3D(blocks_py, state.x, state.y, state.z);
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
                
                /* check for occlusion */
                if (render_mode_occluded(rendermode, state.x, state.y, state.z)) {
                    continue;
                }
                
                /* everything stored here will be a borrowed ref */
                
                if (block_has_property(state.block, NODATA)) {
                    /* block shouldn't have data associated with it, set it to 0 */
                    ancilData = 0;
                    state.block_data = 0;
                    state.block_pdata = 0;
                } else {
                    /* block has associated data, use it */
                    ancilData = getArrayByte3D(state.blockdatas, state.x, state.y, state.z);
                    state.block_data = ancilData;
                    /* block that need pseudo ancildata:
                     * grass, water, glass, chest, restone wire,
                     * ice, fence, portal, iron bars, glass panes,
                     * trapped chests, stairs */
                    if ((state.block ==  2) || (state.block ==  9) ||
                        (state.block == 20) || (state.block == 54) ||
                        (state.block == 55) ||
                        /* doors */
                        (state.block == 64) || (state.block == 193) ||
                        (state.block == 194) || (state.block == 195) ||
                        (state.block == 196) || (state.block == 197) ||
                        (state.block == 71) || /* end doors */
                        (state.block == 79) ||
                        (state.block == 85) || (state.block == 90) ||
                        (state.block == 101) || (state.block == 102) ||
                        (state.block == 111) || (state.block == 113) ||
                        (state.block == 139) || (state.block == 175) || 
                        (state.block == 160) || (state.block == 95) ||
                        (state.block == 146) || (state.block == 188) ||
                        (state.block == 189) || (state.block == 190) ||
                        (state.block == 191) || (state.block == 192) ||
                        is_stairs(state.block)) {
                        ancilData = generate_pseudo_data(&state, ancilData);
                        state.block_pdata = ancilData;
                    } else {
                        state.block_pdata = 0;
                    }
                }
                
                /* make sure our block info is in-bounds */
                if (state.block >= max_blockid || ancilData >= max_data)
                    continue;
                
                /* get the texture */
                t = PyList_GET_ITEM(blockmap, max_data * state.block + ancilData);
                /* if we don't get a texture, try it again with 0 data */
                if ((t == NULL || t == Py_None) && ancilData != 0)
                    t = PyList_GET_ITEM(blockmap, max_data * state.block);
                
                /* if we found a proper texture, render it! */
                if (t != NULL && t != Py_None)
                {
                    PyObject *src, *mask, *mask_light;
                    int do_rand = (state.block == 31 /*|| state.block == 38 || state.block == 175*/);
                    int randx = 0, randy = 0;
                    src = PyTuple_GetItem(t, 0);
                    mask = PyTuple_GetItem(t, 0);
                    mask_light = PyTuple_GetItem(t, 1);

                    if (mask == Py_None)
                        mask = src;

                    if (do_rand) {
                        /* add a random offset to the postion of the tall grass to make it more wild */
                        randx = rand() % 6 + 1 - 3;
                        randy = rand() % 6 + 1 - 3;
                        state.imgx += randx;
                        state.imgy += randy;
                    }
                    
                    render_mode_draw(rendermode, src, mask, mask_light);
                    
                    if (do_rand) {
                        /* undo the random offsets */
                        state.imgx -= randx;
                        state.imgy -= randy;
                    }
                }               
            }
        }
    }

    /* free up the rendermode info */
    render_mode_destroy(rendermode);
    
    Py_DECREF(blockmap);
    unload_all_chunks(&state);

    Py_RETURN_NONE;
}
