#include <Python.h>

#include "chunkrenderer.h"

/*
 * This file defines a number of routines for gathering the effective data
 * value to use when rendering a block. This mechanism is necessary because how
 * to render a block is not only a function of the block ID and the 4 bits of
 * data associated with it in the world file: it may also depend on a
 * neighboring block ID, neighboring block data, or other factors.
 *
 * Therefore, associated with each block definition is a function which takes
 * in the block ID, the block data value, and access to any other block in the
 * world. The output is an int, which is the effective data value used to
 * choose a model and render a block. Each function also takes a parameter that
 * is set on a per block definition basis, to allow more reusability of data
 * functions across different blocks.
 *
 * To define a new data function:
 * 1. write your function(s) in this file. They should be static.
 * 2. Add a new entry to the chunkrenderer_datatypes array at the bottom of
 *    this file
 *
 * Convenient functions to use in these implementations are:
 *
 * get_data(RenderState, data enum, x, y, z) - returns the requested value for
 * the requested block. data enum is one of: BLOCKS, DATA, BLOCKLIGHT,
 * SKYLIGHT, BIOMES.
 *
 * block_has_property(Renderstate, blockid, BlockProperty) - returns true or
 * false depending on whether the given block ID has the given property
 * according to the block definitions. BlockProperty is an enum, one of KNOWN,
 * TRANSPARENT, SOLID, FLUID, NOSPAWN, NODATA.
 */

/*
 * Null function for blocks that don't use data.
 */
static unsigned int nodata(void *param, RenderState *state, int x, int y, int z)
{
    return 0;
}


/*
 * passthrough function for blocks that don't use pseudodata; the model to use
 * only depends on their own data value.
 */
static unsigned int passthrough(void *param, RenderState *state, int x, int y, int z)
{
    return get_data(state, DATA, x, y, z);
}

/* The 4th parameter is the name of the python module attribute */
DataType chunkrenderer_datatypes[] = {
        {nodata, NULL, NULL, "BLOCK_DATA_NODATA"},
        {passthrough, NULL, NULL, "BLOCK_DATA_PASSTHROUGH"},

        {NULL, NULL, NULL, NULL} /* sentinel */
};

