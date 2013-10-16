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

/* This holds information on a particular chunk */
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

/* A generic buffer type, aka an expandable array */
typedef struct {
    void *data;
    /* size of each element in bytes */
    unsigned int element_size;
    /* Number of elements currently in the array */
    unsigned int length;
    /* Number of element slots reserved */
    unsigned int reserved;
} Buffer;

/* Properties a model face can have. These correspond to bitfields in the
 * FaceDef struct below */
enum {
    FACE_TYPE_PX=1,
    FACE_TYPE_NX=2,
    FACE_TYPE_PY=4,
    FACE_TYPE_NY=8,
    FACE_TYPE_PZ=16,
    FACE_TYPE_NZ=32,
    /* Lower 6 bits are the face directions. upper bits are other flags. */
    BIOME_COLORED=64
};
typedef unsigned char FaceType;

typedef struct {
    /* these are points. They reference an index into the vertices array */
    unsigned int p[3];

    /* Flags a face may have. See the FaceType enum */
    FaceType face_type;
} FaceDef;

/* Properties a block can have. These correspond to bitfields in the BlockDef
 * struct below. To add a new property, make sure to change the
 * block_has_property function as well. */
typedef enum {
    KNOWN,
    TRANSPARENT,
    SOLID,
    FLUID,
    NOSPAWN,
    NODATA,
} BlockProperty;

typedef struct {
    int known: 1;
    /* This is a buffer of OILVertex structs */
    Buffer vertices;
    /* This is a buffer of FaceDef structs */
    Buffer faces;
    OILImage *tex;
    
    int transparent: 1;
    int solid: 1;
    int fluid: 1;
    int nospawn: 1;
    int nodata: 1;
} BlockDef;

typedef struct {
    BlockDef *defs;
    unsigned int max_blockid;
    unsigned int max_data;
} BlockDefs;

/* This struct holds information necessary to render a particular chunk section
 * */
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

/* helper to get a block property for block id `b`.
 * If such a block is not defined, the default is 0 for all properties except
 * TRANSPARENT, where the default is 1 */
static inline int block_has_property(RenderState *state, unsigned short b, BlockProperty prop) {
    int def = (prop == TRANSPARENT ? 1 : 0);
    BlockDef bd;
    if (b >= state->blockdefs->max_blockid)
        return def;
    
    /* assume blocks have uniform properties across all data -- that these
     * properties do not depend on the data field.
     * (otherwise to do this right we'd need to calculate pseudo-data) */
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

/* helper to get a block definition
 * This takes the entire RenderState struct instead of just the block
 * definitions because we will one day probably need to get neighboring blocks
 * for pseudo data. */
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

/* 
 * Renders a single block to an image, given a model vertices and faces.
 * opaque_neighbors is a bitfield showing which neighboring blocks are
 * not transparent (using the FACE_TYPE_* bitflags).
 */
static inline void render_block(OILImage *im, OILMatrix *matrix, Buffer *vertices, Buffer *faces, FaceType opaque_neighbors, OILImage *tex)
{
    int i;
    Buffer indices;

    buffer_init(&indices, sizeof(unsigned int), faces->length * 3);

    for (i=0; i < faces->length; i++) {
        FaceDef *face = &((FaceDef *)(faces->data))[i];

        /* 
         * the first 6 bits of face_type define which directions the face is
         * visible from.
         * opaque_neighbors is a bit mask showing which directions have
         * transparent blocks.
         * Therefore, only render the face if at least one of the directions
         * has a transparent block
         */
        if ((face->face_type & 63) & ~opaque_neighbors) {
            buffer_append(&indices, &face->p, 3);
        }
    }

    if (indices.length && vertices->length)
        oil_image_draw_triangles(im, matrix, tex, vertices->data, vertices->length, indices.data, indices.length, OIL_DEPTH_TEST);

    buffer_free(&indices);
}

static PyObject *render(PyObject *self, PyObject *args) {
    RenderState state;
    PyOILImage *im;
    PyOILMatrix *mat;
    PyObject *pyblockdefs;
    Buffer blockvertices;

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
    
    /* convenience */
    blocks = state.chunks[1][1].sections[state.chunky].blocks;
    datas = state.chunks[1][1].sections[state.chunky].data;
    
    /* set up the random number generator in a known state per chunk */
    srand(1);
    
    /* Here starts the loop over every block in this chunk section */
    for (x = 0; x < 16; x++) {
        for (z = 0; z < 16; z++) {
            for (y = 15; y >= 0; y--) {
                unsigned short block = get_array_short_3d(blocks, x, y, z);
                unsigned short data = get_array_byte_3d(datas, x, y, z);
                FaceType opaque_neighbors = 0;
                BlockDef *bd = get_block_definition(&state, x, y, z, block, data);
                if (!bd)
                    continue;
                
                /* Clear out the block vertices buffer and then fill it with
                 * this block's vertices. We have to copy this buffer because
                 * we adjust the vertices below. */
                blockvertices.length = 0;
                buffer_append(&blockvertices, bd->vertices.data, bd->vertices.length);

                /* Adjust each vertex coordinate for the position within the
                 * chunk. (model vertex coordinates are typically within the
                 * unit cube, and this adds the offset for where that cube is
                 * within the chunk section) */
                for (i = 0; i < blockvertices.length; i++) {
                    OILVertex *v = &((OILVertex *)(blockvertices.data))[i];
                    v->x += x;
                    v->y += y;
                    v->z += z;
                }

                /* Here we look at the neighboring blocks and build a bitmask
                 * of the faces that we should render on the model depending on
                 * which neighboring blocks are transparent.
                 */
                if (!block_has_property(&state, get_data(&state, BLOCKS, x+1, y, z), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_PX;
                }
                if (!block_has_property(&state, get_data(&state, BLOCKS, x-1, y, z), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_NX;
                }
                if (!block_has_property(&state, get_data(&state, BLOCKS, x, y+1, z), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_PY;
                }
                if (!block_has_property(&state, get_data(&state, BLOCKS, x, y-1, z), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_NY;
                }
                if (!block_has_property(&state, get_data(&state, BLOCKS, x, y, z+1), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_PZ;
                }
                if (!block_has_property(&state, get_data(&state, BLOCKS, x, y, z-1), TRANSPARENT)) {
                    opaque_neighbors |= FACE_TYPE_NZ;
                }

                /* Now draw the block. This function takes care of figuring out
                 * what faces to draw based on the above */
                render_block(state.im, state.matrix, &blockvertices, &bd->faces, opaque_neighbors, bd->tex);
            }
        }
    }
    
    /* clean up */
    unload_all_chunks(&state);
    buffer_free(&blockvertices);
    
    Py_RETURN_NONE;
}

static void free_block_definitions(void *obj) {
    unsigned int i, j;
    BlockDefs *defs = (BlockDefs *)obj;
    
    for (i = 0; i < defs->max_blockid * defs->max_data; i++) {
        buffer_free(&(defs->defs[i].vertices));
        buffer_free(&(defs->defs[i].faces));
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
    buffer_init(&(def->faces), sizeof(FaceDef), BLOCK_BUFFER_SIZE);
    
    for (i = 0; i < vertices_length; i++) {
        OILVertex vert;
        PyObject *pyvert = PySequence_Fast_GET_ITEM(pyvertfast, i);
        if (!PyArg_ParseTuple(pyvert, "(fff)(ff)(bbbb)", &(vert.x), &(vert.y), &(vert.z), &(vert.s), &(vert.t), &(vert.color.r), &(vert.color.g), &(vert.color.b), &(vert.color.a))) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            Py_DECREF(pytex);
            PyErr_SetString(PyExc_ValueError, "vertex has invalid form. expected ((float, float, float), (float, float), (byte, byte, byte))");
            return 0;
        }
        buffer_append(&(def->vertices), &vert, 1);
    }

    for (i = 0; i < triangles_length; i++) {
        FaceDef face;
        unsigned int type;
        PyObject *triangle = PySequence_Fast_GET_ITEM(pytrifast, i);
        if (!PyArg_ParseTuple(triangle, "(III)I", &face.p[0], &face.p[1], &face.p[2], &type)) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            Py_DECREF(pytex);
            PyErr_SetString(PyExc_ValueError, "triangle has invalid form. expected ((int, int, int), int)");
            return 0;
        }
        face.face_type = (FaceType) type;
        if ((face.face_type & 63) == 0) {
            /* No face direction information given. Assume it faces all
             * directions instead. This bit of logic should probably go
             * someplace more conspicuous.
             */
            face.face_type |= 63;
        }
        buffer_append(&(def->faces), &face, 1);
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

    /* Important to use calloc so the "known" member of the BlockDef struct is
     * by default 0 */
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

/*
 * Python export of the render_block function
 * signature: render_block(image, matrix, compiled_blockdefs, blockid, data=0)
 */
static PyObject *py_render_block(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyOILImage *im;
    PyOILMatrix *mat;
    PyObject *pyblockdefs;
    BlockDefs *bd;
    BlockDef *block;
    unsigned int blockid;
    unsigned int data=0;

    static char *kwlist[] = {"image", "matrix", "blockdefs", "blockid", "data"};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!OI|I", kwlist, 
                PyOILImageType, &im,
                PyOILMatrixType, &mat,
                &pyblockdefs,
                &blockid,
                &data
                )) {
        return 0;
    }

    bd = PyCObject_AsVoidPtr(pyblockdefs);

    /* Look up this block's definition */
    block = &(bd->defs[blockid * bd->max_data + data]);

    render_block(im->im, &mat->matrix, &block->vertices, &block->faces, 0, block->tex);
    Py_RETURN_NONE;
}

static PyMethodDef chunkrenderer_methods[] = {
    {"render", render, METH_VARARGS,
     "Render a chunk to an image."},
    {"compile_block_definitions", compile_block_definitions, METH_VARARGS,
     "Compiles a Textures object and a BlockDefinitions object into a form usable by the render method."},
    {"render_block", py_render_block, METH_VARARGS | METH_KEYWORDS,
     "Renders a single block to the given image with the given matrix"},
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
       sizeof(...) does not evaluate its argument (:D) */
    (void)sizeof(import_array());
    
    /* import numpy on our own, because import_array breaks across
       numpy versions and we barely use numpy */
    numpy = PyImport_ImportModule("numpy.core.multiarray");
    Py_XDECREF(numpy);
}
