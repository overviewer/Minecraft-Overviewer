#include <Python.h>
#include <oil.h>
#include <oil-python.h>
#include <numpy/arrayobject.h>

#define SECTIONS_PER_CHUNK 16
#define BLOCK_BUFFER_SIZE 32

/* macro for getting a value out of various numpy arrays the 3D arrays have
   interesting, swizzled coordinates because minecraft (anvil) stores blocks
   in y/z/x order for 3D, z/x order for 2D */
#define get_array_byte_3d(array, x,y,z) (*(unsigned char *)(PyArray_GETPTR3((array), (y), (z), (x))))
#define get_array_short_3d(array, x,y,z) (*(unsigned short *)(PyArray_GETPTR3((array), (y), (z), (x))))
#define get_array_byte_2d(array, x,y) (*(unsigned char *)(PyArray_GETPTR2((array), (y), (x))))

static PyTypeObject *PyOILMatrixType = NULL;
static PyTypeObject *PyOILImageType = NULL;

typedef struct {
    /* whether this chunk is loaded: use load_chunk to load */
    int loaded;
    /* biome array */
    PyObject *biomes;
    /* all the sections in a given chunk */
    struct {
        /* all there is to know about each seciton */
        PyObject *blocks, *data, *skylight, *blocklight;
    } sections[SECTIONS_PER_CHUNK];
} ChunkData;

typedef struct {
    void *data;
    unsigned int element_size;
    unsigned int length;
    unsigned int reserved;
} Buffer;

typedef enum {
    FACE_TYPE_PX,
    FACE_TYPE_NX,
    FACE_TYPE_PY,
    FACE_TYPE_NY,
    FACE_TYPE_PZ,
    FACE_TYPE_NZ,
    FACE_TYPE_COUNT
} FaceType;

typedef enum {
    KNOWN,
    TRANSPARENT,
    SOLID,
    FLUID,
    NOSPAWN,
    NODATA,
} BlockProperty;

typedef struct {
    int known;
    Buffer vertices;
    Buffer indices[FACE_TYPE_COUNT];
    OILImage *tex;
    
    int transparent;
    int solid;
    int fluid;
    int nospawn;
    int nodata;
} BlockDef;

typedef struct {
    BlockDef *defs;
    unsigned int max_blockid;
    unsigned int max_data;
} BlockDefs;

typedef struct {
    PyObject *regionset;
    int chunkx, chunky, chunkz;
    ChunkData chunks[3][3];
    
    OILImage *im;
    OILMatrix *matrix;
    
    BlockDefs *blockdefs;
} RenderState;

/* Buffer objects are conveniences for realloc'd dynamic arrays */

static inline void buffer_init(Buffer *buffer, unsigned int element_size, unsigned int initial_length) {
    buffer->data = NULL;
    buffer->element_size = element_size;
    buffer->length = 0;
    buffer->reserved = initial_length;
}

static inline void buffer_free(Buffer *buffer) {
    if (buffer->data)
        free(buffer->data);
}

static inline void buffer_reserve(Buffer *buffer, unsigned int length) {
    int needs_realloc = 0;
    while (buffer->length + length > buffer->reserved) {
        buffer->reserved *= 2;
        needs_realloc = 1;
    }
    if (buffer->data == NULL)
        needs_realloc = 1;
    
    if (needs_realloc) {
        buffer->data = realloc(buffer->data, buffer->element_size * buffer->reserved);
    }
}

static inline void buffer_append(Buffer *buffer, const void *newdata, unsigned int newdata_length) {
    buffer_reserve(buffer, newdata_length);
    memcpy(buffer->data + (buffer->element_size * buffer->length), newdata, buffer->element_size * newdata_length);
    buffer->length += newdata_length;
}

/* helper to load a chunk into state->chunks
 * returns false on error, true on success
 *
 * if required is true, failure will set a python error
 */
static inline int load_chunk(RenderState *state, int relx, int relz, int required) {
    ChunkData *dest = &(state->chunks[1 + relx][1 + relz]);
    int i, x, z;
    PyObject *chunk = NULL;
    PyObject *sections = NULL;
    
    if (dest->loaded)
        return 1;
    
    /* set up reasonable defaults */
    dest->biomes = NULL;
    for (i = 0; i < SECTIONS_PER_CHUNK; i++) {
        dest->sections[i].blocks = NULL;
        dest->sections[i].data = NULL;
        dest->sections[i].skylight = NULL;
        dest->sections[i].blocklight = NULL;
    }
    dest->loaded = 1;
    
    x = state->chunkx + relx;
    z = state->chunkz + relz;
    
    chunk = PyObject_CallMethod(state->regionset, "get_chunk", "ii", x, z);
    if (!chunk) {
        /* an exception is already set by get_chunk */
        if (!required) {
            PyErr_Clear();
        }
        return 0;
    }
    
    sections = PyDict_GetItemString(chunk, "Sections");
    if (sections) {
        sections = PySequence_Fast(sections, "Sections tag was not a list!");
    }
    if (!sections) {
        /* exception set already, again */
        Py_DECREF(chunk);
        if (!required) {
            PyErr_Clear();
        }
        return 0;
    }
    
    dest->biomes = PyDict_GetItemString(chunk, "Biomes");
    Py_XINCREF(dest->biomes);
    
    for (i = 0; i < PySequence_Fast_GET_SIZE(sections); i++) {
        PyObject *ycoord = NULL;
        int sectiony = 0;
        PyObject *section = PySequence_Fast_GET_ITEM(sections, i);
        ycoord = PyDict_GetItemString(section, "Y");
        if (!ycoord)
            continue;
        
        sectiony = PyInt_AsLong(ycoord);
        if (sectiony >= 0 && sectiony < SECTIONS_PER_CHUNK) {
            dest->sections[i].blocks = PyDict_GetItemString(section, "Blocks");
            Py_XINCREF(dest->sections[i].blocks);
            dest->sections[i].data = PyDict_GetItemString(section, "Data");
            Py_XINCREF(dest->sections[i].data);
            dest->sections[i].skylight = PyDict_GetItemString(section, "SkyLight");
            Py_XINCREF(dest->sections[i].skylight);
            dest->sections[i].blocklight = PyDict_GetItemString(section, "BlockLight");
            Py_XINCREF(dest->sections[i].blocklight);
        }
    }
    Py_DECREF(sections);
    Py_DECREF(chunk);
    
    return 1;
}

/* helper to unload all loaded chunks */
static inline void unload_all_chunks(RenderState *state) {
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

/* helper for indexing section data possibly across section boundaries */
typedef enum
{
    BLOCKS,
    DATA,
    BLOCKLIGHT,
    SKYLIGHT,
    BIOMES,
} DataType;
static inline unsigned int get_data(RenderState *state, DataType type, int x, int y, int z)
{
    int chunkx = 1, chunky = state->chunky, chunkz = 1;
    PyObject *data_array = NULL;
    unsigned int def = 0;
    if (type == SKYLIGHT)
        def = 15;
    
    if (x >= 16) {
        x -= 16;
        chunkx++;
    } else if (x < 0) {
        x += 16;
        chunkx--;
    }
    if (z >= 16) {
        z -= 16;
        chunkz++;
    } else if (z < 0) {
        z += 16;
        chunkz--;
    }

    while (y >= 16) {
        y -= 16;
        chunky++;
    }
    while (y < 0) {
        y += 16;
        chunky--;
    }
    if (chunky < 0 || chunky >= SECTIONS_PER_CHUNK)
        return def;
    
    if (!(state->chunks[chunkx][chunkz].loaded))
    {
        if (!load_chunk(state, chunkx - 1, chunkz - 1, 0))
            return def;
    }
    
    switch (type)
    {
    case BLOCKS:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].blocks;
        break;
    case DATA:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].data;
        break;
    case BLOCKLIGHT:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].blocklight;
        break;
    case SKYLIGHT:
        data_array = state->chunks[chunkx][chunkz].sections[chunky].skylight;
        break;
    case BIOMES:
        data_array = state->chunks[chunkx][chunkz].biomes;
    };
    
    if (data_array == NULL)
        return def;
    
    if (type == BLOCKS)
        return get_array_short_3d(data_array, x, y, z);
    if (type == BIOMES)
        return get_array_byte_2d(data_array, x, z);
    return get_array_byte_3d(data_array, x, y, z);
}

/* helper to get a block property */
static inline int block_has_property(RenderState *state, unsigned short b, BlockProperty prop) {
    int def = (prop == TRANSPARENT ? 1 : 0);
    BlockDef bd;
    if (b >= state->blockdefs->max_blockid)
        return def;
    
    /* assume blocks have uniform properties across all data
       (otherwise to do this right we'd need to calculate pseudo-data) */
    bd = state->blockdefs->defs[b * state->blockdefs->max_data];
    if (!bd.known)
        return def;
    
    switch (prop) {
    case KNOWN:
        return 1;
    case TRANSPARENT:
        return bd.transparent;
    case SOLID:
        return bd.solid;
    case FLUID:
        return bd.fluid;
    case NOSPAWN:
        return bd.nospawn;
    case NODATA:
        return bd.nodata;
    }
    
    return def;
}

/* helper to get a block definition */
static inline BlockDef *get_block_definition(RenderState *state, int x, int y, int z, unsigned int blockid, unsigned int data) {
    BlockDef *def;
    if (blockid >= state->blockdefs->max_blockid)
        return NULL;
    
    // first, check for nodata set on data == 0
    def = &(state->blockdefs->defs[blockid * state->blockdefs->max_data]);
    if (!(def->known))
        return NULL;
    if (def->nodata)
        return def;
    
    if (data >= state->blockdefs->max_data)
        return NULL;
    
    // data is used, so use it
    def = &(state->blockdefs->defs[blockid * state->blockdefs->max_data + data]);
    if (!(def->known))
        return NULL;
    
    // still todo: pseudo-data
    return def;
}

/* helper to emit some mesh info */
static inline void emit_mesh(RenderState *state, OILImage *tex, const Buffer *vertices, const Buffer *indices) {
    oil_image_draw_triangles(state->im, state->matrix, tex, vertices->data, vertices->length, indices->data, indices->length, OIL_DEPTH_TEST);
}

static PyObject *render(PyObject *self, PyObject *args) {
    RenderState state;
    PyOILImage *im;
    PyOILMatrix *mat;
    PyObject *pyblockdefs;
    Buffer blockvertices;
    Buffer blockindices;

    int i, j;
    int x, y, z;
    PyObject *blocks, *datas;
    
    if (!PyArg_ParseTuple(args, "OiiiO!O!O", &(state.regionset), &(state.chunkx), &(state.chunky), &(state.chunkz), PyOILImageType, &im, PyOILMatrixType, &mat, &pyblockdefs)) {
        return NULL;
    }
    
    state.blockdefs = PyCObject_AsVoidPtr(pyblockdefs);
    
    state.im = im->im;
    state.matrix = &(mat->matrix);
    
    /* load the center chunk */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            state.chunks[i][j].loaded = 0;
        }
    }
    if (!load_chunk(&state, 0, 0, 1)) {
        return NULL;
    }
    
    /* make sure our chunk section exists */
    if (state.chunks[1][1].sections[state.chunky].blocks == NULL) {
        unload_all_chunks(&state);
        Py_RETURN_NONE;
    }
    
    /* set up the mesh buffers */
    buffer_init(&blockvertices, sizeof(OILVertex), BLOCK_BUFFER_SIZE);
    buffer_init(&blockindices, sizeof(unsigned int), BLOCK_BUFFER_SIZE * 2);
    
    /* convenience */
    blocks = state.chunks[1][1].sections[state.chunky].blocks;
    datas = state.chunks[1][1].sections[state.chunky].data;
    
    /* set up the random number generator in a known state per chunk */
    srand(1);
    
    for (x = 0; x < 16; x++) {
        for (z = 0; z < 16; z++) {
            for (y = 15; y >= 0; y--) {
                unsigned short block = get_array_short_3d(blocks, x, y, z);
                unsigned short data = get_array_byte_3d(datas, x, y, z);
                BlockDef *bd = get_block_definition(&state, x, y, z, block, data);
                if (!bd)
                    continue;
                
                blockvertices.length = 0;
                buffer_append(&blockvertices, bd->vertices.data, bd->vertices.length);
                for (i = 0; i < blockvertices.length; i++) {
                    OILVertex *v = &((OILVertex *)(blockvertices.data))[i];
                    v->x += x;
                    v->y += y;
                    v->z += z;
                }

                blockindices.length = 0;
                for (j = 0; j < FACE_TYPE_COUNT; j++) {
                    unsigned short testblock = 0;
                    switch (j) {
                    case FACE_TYPE_PX:
                        testblock = get_data(&state, BLOCKS, x + 1, y, z);
                        break;
                    case FACE_TYPE_NX:
                        testblock = get_data(&state, BLOCKS, x - 1, y, z);
                        break;
                    case FACE_TYPE_PY:
                        testblock = get_data(&state, BLOCKS, x, y + 1, z);
                        break;
                    case FACE_TYPE_NY:
                        testblock = get_data(&state, BLOCKS, x, y - 1, z);
                        break;
                    case FACE_TYPE_PZ:
                        testblock = get_data(&state, BLOCKS, x, y, z + 1);
                        break;
                    case FACE_TYPE_NZ:
                        testblock = get_data(&state, BLOCKS, x, y, z - 1);
                        break;
                    };
                    
                    if (block_has_property(&state, testblock, TRANSPARENT)) {
                        buffer_append(&blockindices, bd->indices[j].data, bd->indices[j].length);
                    }
                }
                
                if (blockvertices.length && blockindices.length)
                    emit_mesh(&state, bd->tex, &(blockvertices), &(blockindices));
            }
        }
    }
    
    /* clean up */
    unload_all_chunks(&state);
    buffer_free(&blockvertices);
    buffer_free(&blockindices);
    
    Py_RETURN_NONE;
}

static void free_block_definitions(void *obj) {
    unsigned int i, j;
    BlockDefs *defs = (BlockDefs *)obj;
    
    for (i = 0; i < defs->max_blockid * defs->max_data; i++) {
        buffer_free(&(defs->defs[i].vertices));
        for (j = 0; j < FACE_TYPE_COUNT; j++) {
            buffer_free(&(defs->defs[i].indices[j]));
        }
    }
    
    free(defs->defs);
    free(defs);
}

/* helper for compile_block_definitions */
inline static int compile_block_definition(PyObject *pytextures, BlockDef *def, PyObject *pydef) {
    unsigned int i;
    PyObject *pyvertices = PyObject_GetAttrString(pydef, "vertices");
    unsigned int vertices_length;
    PyObject *pytriangles = PyObject_GetAttrString(pydef, "triangles");
    unsigned int triangles_length;
    PyObject *pytex = PyObject_GetAttrString(pydef, "tex");
    PyObject *pyteximg;
    PyObject *pyvertfast;
    PyObject *pytrifast;
    PyObject *prop;
    int prop_istrue = 0;
    if (!pyvertices || !pytriangles || !pytex) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }
    
    pyteximg = PyObject_CallMethod(pytextures, "load", "O", pytex);
    if (!pyteximg || !PyObject_TypeCheck(pyteximg, PyOILImageType)) {
        if (pyteximg) {
            PyErr_SetString(PyExc_TypeError, "Textures.load() did not return an OIL Image");
            Py_DECREF(pyteximg);
        }
        Py_DECREF(pyvertices);
        Py_DECREF(pytriangles);
        Py_DECREF(pytex);
        return 0;
    }
    Py_DECREF(pytex);
    pytex = pyteximg;
    
    prop = PyObject_GetAttrString(pydef, "transparent");
    if (prop) {
        def->transparent = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }

    prop = PyObject_GetAttrString(pydef, "solid");
    if (prop) {
        def->solid = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }

    prop = PyObject_GetAttrString(pydef, "fluid");
    if (prop) {
        def->fluid = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }

    prop = PyObject_GetAttrString(pydef, "nospawn");
    if (prop) {
        def->nospawn = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }

    prop = PyObject_GetAttrString(pydef, "nodata");
    if (prop) {
        def->nodata = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        Py_XDECREF(pytex);
        return 0;
    }
    
    pyvertfast = PySequence_Fast(pyvertices, "vertices were not a sequence");
    pytrifast = PySequence_Fast(pytriangles, "triangles were not a sequence");
    Py_DECREF(pyvertices);
    Py_DECREF(pytriangles);
    if (!pyvertfast || !pytrifast) {
        Py_XDECREF(pyvertfast);
        Py_XDECREF(pytrifast);
        return 0;
    }
    
    vertices_length = PySequence_Fast_GET_SIZE(pyvertfast);
    triangles_length = PySequence_Fast_GET_SIZE(pytrifast);
    
    buffer_init(&(def->vertices), sizeof(OILVertex), BLOCK_BUFFER_SIZE);
    for (i = 0; i < FACE_TYPE_COUNT; i++) {
        buffer_init(&(def->indices[i]), sizeof(unsigned int), BLOCK_BUFFER_SIZE * 2);
    }
    
    for (i = 0; i < vertices_length; i++) {
        OILVertex vert;
        PyObject *pyvert = PySequence_Fast_GET_ITEM(pyvertfast, i);
        if (!PyArg_ParseTuple(pyvert, "(fff)(ff)(bbbb)", &(vert.x), &(vert.y), &(vert.z), &(vert.s), &(vert.t), &(vert.color.r), &(vert.color.g), &(vert.color.b), &(vert.color.a))) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            Py_DECREF(pytex);
            PyErr_SetString(PyExc_ValueError, "vertex has invalid form");
            return 0;
        }
        buffer_append(&(def->vertices), &vert, 1);
    }

    for (i = 0; i < triangles_length; i++) {
        PyObject *pytri = PySequence_Fast_GET_ITEM(pytrifast, i);
        unsigned int tri[3];
        unsigned int facetype;
        if (!PyArg_ParseTuple(pytri, "(III)I", &(tri[0]), &(tri[1]), &(tri[2]), &facetype) || facetype >= FACE_TYPE_COUNT) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            Py_DECREF(pytex);
            PyErr_SetString(PyExc_ValueError, "triangle has invalid form");
            return 0;
        }
        buffer_append(&(def->indices[facetype]), tri, 3);
    }
    
    def->tex = ((PyOILImage *)pytex)->im;
    def->known = 1;
    
    Py_DECREF(pyvertfast);
    Py_DECREF(pytrifast);
    Py_DECREF(pytex);
    
    return 1;
}

static PyObject *compile_block_definitions(PyObject *self, PyObject *args) {
    PyObject *pytextures;
    PyObject *pyblockdefs;
    PyObject *pymaxblockid;
    PyObject *pymaxdata;
    PyObject *pyblocks;
    PyObject *compiled;
    BlockDefs *defs;
    unsigned int blockid, data;
    
    if (!PyArg_ParseTuple(args, "OO", &pytextures, &pyblockdefs)) {
        return NULL;
    }
    
    defs = malloc(sizeof(BlockDefs));
    if (!defs) {
        PyErr_SetString(PyExc_RuntimeError, "out of memory");
        return NULL;
    }
    
    pymaxblockid = PyObject_GetAttrString(pyblockdefs, "max_blockid");
    if (!pymaxblockid) {
        free(defs);
        return NULL;
    }
    defs->max_blockid = PyInt_AsLong(pymaxblockid);
    Py_DECREF(pymaxblockid);
    
    pymaxdata = PyObject_GetAttrString(pyblockdefs, "max_data");
    if (!pymaxdata) {
        free(defs);
        return NULL;
    }
    defs->max_data = PyInt_AsLong(pymaxdata);
    Py_DECREF(pymaxdata);
    
    pyblocks = PyObject_GetAttrString(pyblockdefs, "blocks");
    if (!pyblocks) {
        free(defs);
        return NULL;
    }
    if (!PyDict_Check(pyblocks)) {
        PyErr_SetString(PyExc_TypeError, "blocks index is not a dictionary");
        free(defs);
        Py_DECREF(pyblocks);
        return NULL;
    }

    defs->defs = calloc(defs->max_blockid * defs->max_data, sizeof(BlockDef));
    if (!(defs->defs)) {
        PyErr_SetString(PyExc_RuntimeError, "out of memory");
        free(defs);
        Py_DECREF(pyblocks);
        return NULL;
    }
    
    for (blockid = 0; blockid < defs->max_blockid; blockid++) {
        for (data = 0; data < defs->max_data; data++) {
            PyObject *key = Py_BuildValue("II", blockid, data);
            PyObject *val;
        
            if (!key) {
                free_block_definitions(defs);
                Py_DECREF(pyblocks);
                return NULL;
            }
        
            val = PyDict_GetItem(pyblocks, key);
            if (val) {
                if (!compile_block_definition(pytextures, &(defs->defs[blockid * defs->max_data + data]), val)) {
                    free_block_definitions(defs);
                    Py_DECREF(pyblocks);
                    return NULL;
                }
            }
                        
            Py_DECREF(key);   
        }
    }
    
    Py_DECREF(pyblocks);
    
    compiled = PyCObject_FromVoidPtr(defs, free_block_definitions);
    if (!compiled) {
        free_block_definitions(defs);
        return NULL;
    }
    
    return compiled;
}

static PyMethodDef chunkrenderer_methods[] = {
    {"render", render, METH_VARARGS,
     "Render a chunk to an image."},
    {"compile_block_definitions", compile_block_definitions, METH_VARARGS,
     "Compiles a Textures object and a BlockDefinitions object into a form usable by the render method."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initchunkrenderer(void) {
    PyObject *mod, *numpy;
    
    PyOILMatrixType = py_oil_get_matrix_type();
    PyOILImageType = py_oil_get_image_type();
    if (!(PyOILMatrixType && PyOILImageType)) {
        Py_XDECREF(PyOILMatrixType);
        Py_XDECREF(PyOILImageType);
        return;
    }
    
    mod = Py_InitModule3("chunkrenderer", chunkrenderer_methods,
                         "The Overviewer chunk renderer interface.");
    if (mod == NULL)
        return;

    PyModule_AddIntConstant(mod, "FACE_TYPE_PX", FACE_TYPE_PX);
    PyModule_AddIntConstant(mod, "FACE_TYPE_NX", FACE_TYPE_NX);
    PyModule_AddIntConstant(mod, "FACE_TYPE_PY", FACE_TYPE_PY);
    PyModule_AddIntConstant(mod, "FACE_TYPE_NY", FACE_TYPE_NY);
    PyModule_AddIntConstant(mod, "FACE_TYPE_PZ", FACE_TYPE_PZ);
    PyModule_AddIntConstant(mod, "FACE_TYPE_NZ", FACE_TYPE_NZ);
    
    /* tell the compiler to shut up about unused things
       sizeof(...) does not evaluate it's argument (:D) */
    (void)sizeof(import_array());
    
    /* import numpy on our own, because import_array breaks across
       numpy versions and we barely use numpy */
    numpy = PyImport_ImportModule("numpy.core.multiarray");
    Py_XDECREF(numpy);
}
