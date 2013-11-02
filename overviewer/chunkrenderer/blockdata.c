#include <Python.h>

#include "chunkrenderer.h"
#include "buffer.h"

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


/*
 * Sticky Neighbors is for blocks whose model depends on the state of its
 * neighbors in the x/z plane. The parameter should be a set of block IDs that
 * "stick" to this block.
 * The output bits indicate which neighbors are sticky
 * xxx1 - px neighbor is sticky
 * xx1x - nx neighbor is sticky
 * x1xx - pz neighbor is sticky
 * 1xxx - nz neighbor is sticky
 *
 * Enumerated:
 * 0  - none are sticky
 * 1  - px neighbor is sticky
 * 2  - nx
 * 3  - px and nx
 * 4  - pz
 * 5  - px and pz
 * 6  - nx and pz
 * 7  - px and nx and pz
 * 8  - nz
 * 9  - px and nz
 * 10 - nx and nz
 * 11 - px and nx and nz
 * 12 - pz and nz
 * 13 - px and pz and nz
 * 14 - nx and pz and nz
 * 15 - px and nx and pz and nz
 *
 * Input to the parameter is a python sequence of blockIDs that are considered sticky
 *
 * The start method transforms it into a boolean table: 0 for not sticky, and 1 for sticky
 */
struct sticky_table {
    int length;
    char *entries;
};
static unsigned int sticky_neighbors(void *param, RenderState *state, int x, int y, int z)
{
    int px,nx,pz,nz;
    unsigned int ret;
    struct sticky_table *table = param;
    /* neighbors */
    px = get_data(state, BLOCKS, x+1, y, z);
    nx = get_data(state, BLOCKS, x-1, y, z);
    pz = get_data(state, BLOCKS, x, y, z+1);
    nz = get_data(state, BLOCKS, x, y, z-1);

    ret = 0;

    if (px < table->length && table->entries[px])
        ret |= 1 << 0;
    if (nx < table->length && table->entries[nx])
        ret |= 1 << 1;
    if (pz < table->length && table->entries[pz])
        ret |= 1 << 2;
    if (nz < table->length && table->entries[nz])
        ret |= 1 << 3;

    return ret;
}
static void *sticky_neighbors_start(PyObject *param)
{
    int i;
    int max_blockid;
    struct sticky_table *table;
    PyObject *sequence = PySequence_Fast(param, "Parameter must be a sequence of integers");
    if (!sequence) {
        return NULL;
    }

    max_blockid = 0;
    for (i=0; i<PySequence_Fast_GET_SIZE(sequence); i++) {
        int blockid = PyInt_AsLong(PySequence_Fast_GET_ITEM(sequence, i));
        if (blockid == -1 && PyErr_Occurred()) {
            Py_DECREF(sequence);
            return NULL;
        } else if (blockid < 0) {
            PyErr_SetString(PyExc_ValueError, "Block id cannot be negative");
            Py_DECREF(sequence);
            return NULL;
        }
        if (blockid > max_blockid)
            max_blockid = blockid;
    }

    table = malloc(sizeof(struct sticky_table));
    table->length = max_blockid+1;
    table->entries = calloc(table->length, sizeof(char));

    for (i=0; i<PySequence_Fast_GET_SIZE(sequence); i++) {
        int blockid = PyInt_AsLong(PySequence_Fast_GET_ITEM(sequence, i));
        table->entries[blockid] = 1;
    }

    Py_DECREF(sequence);
    return table;

}
static void free_sticky_table(void *param)
{
    struct sticky_table *table = param;
    free(table->entries);
    free(table);
}

/* Add new data types here.
 * The 4th parameter is the name of the python module attribute */
DataType chunkrenderer_datatypes[] = {
        {nodata, NULL, NULL, "BLOCK_DATA_NODATA"},
        {passthrough, NULL, NULL, "BLOCK_DATA_PASSTHROUGH"},
        {sticky_neighbors, sticky_neighbors_start, free_sticky_table, "BLOCK_DATA_STICKY"},
};

const int chunkrenderer_datatypes_length = sizeof(chunkrenderer_datatypes) / sizeof(DataType);
