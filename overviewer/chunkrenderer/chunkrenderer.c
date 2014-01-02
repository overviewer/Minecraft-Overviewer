#include <Python.h>
#include <oil.h>
#include <oil-python.h>
#include <numpy/arrayobject.h>

#include "chunkrenderer.h"

/* helper to load a chunk into state->chunks
 * returns false on error, true on success
 *
 * x, y, and z are given relative to the center chunk (the chunk currently
 * being rendered)
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

/* helper for indexing section data possibly across section boundaries.
 * x, y, and z are given relative to the center chunk section according to the
 * render state. 
 * Not static so that this can be used in blockdata.c
 */
inline unsigned int get_data(RenderState *state, enum _get_data_type_enum type, int x, int y, int z)
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
 * TRANSPARENT, where the default is 1
 * Not static so that this can be used in blockdata.c
 */
inline int block_has_property(RenderState *state, unsigned short b, BlockProperty prop) {
    int def = (prop == TRANSPARENT ? 1 : 0);
    BlockDef bd;
    if (b >= state->blockdefs->blockdefs_length)
        return def;
    
    /* assume blocks have uniform properties across all data -- that these
     * properties do not depend on the data field.
     * (otherwise to do this right we'd need to calculate pseudo-data) */
    bd = state->blockdefs->defs[b];
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
    }
    
    return def;
}

static inline BlockModel *get_block_def(RenderState *state, int x, int y, int z) {
    BlockDef *def;
    unsigned int blockid;

    /* we need to find the block definition of the requested block. */
    blockid = get_data(state, BLOCKS, x, y, z);
    if (blockid >= state->blockdefs->blockdefs_length)
        return NULL;

    def = &state->blockdefs->defs[blockid];
    if (def->known == 0) {
        return NULL;
    }
    return def;
}

/* helper to get a block model to use for a given block in a minecraft world.
 * This function is used by render() This takes the entire RenderState struct
 * instead of just the block definitions because we need to account for pseudo
 * data.
 * x, y, and z are given relative to the center chunk section (chunk [1][1] in
 * the state struct).
 */
static inline BlockModel *get_block_model(RenderState *state, BlockDef *def, int x, int y, int z) {
    BlockModel *model;
    unsigned int effectivedata;

    if (def == NULL)
        return NULL;

    /* Now that we have a block definition, call the data function to determine
     * which model to use. */
    effectivedata = def->datatype.datafunc(def->dataparameter, state, x, y, z);

    if (effectivedata >= def->models_length)
        return NULL;
    
    model = &(def->models[effectivedata]);

    if (!model->known) {
        return NULL;
    }
    
    return model;
}

static inline void emit_mesh(OILImage *im, OILMatrix *matrix, Buffer *vertices, Buffer *indices, OILImage *tex)
{
    if (indices->length && vertices->length)
        oil_image_draw_triangles(im, matrix, tex, vertices->data, vertices->length, indices->data, indices->length, OIL_DEPTH_TEST);

}

/* 
 * Renders a single block to an image, given a model vertices and faces.
 * opaque_neighbors is a bitfield showing which neighboring blocks are
 * not transparent (using the FACE_TYPE_* bitflags).
 */
static inline void render_block(OILImage *im, OILMatrix *matrix, Buffer *vertices, Buffer *faces, FaceType opaque_neighbors)
{
    int i;
    Buffer indices;
    /* Assume there is at least one face in the model */
    OILImage *tex = (&((FaceDef *)(faces->data))[0])->tex;

    /* Make a buffer of only the faces that we're going to draw, depending on
     * the opaque neighbors */
    buffer_init(&indices, sizeof(unsigned int), BLOCK_BUFFER_SIZE);

    for (i=0; i < faces->length; i++) {
        FaceDef *face = &((FaceDef *)(faces->data))[i];

        /* This face has a different texture than the last face. Render all the
         * faces accumulated so far.
         */
        if (tex != face->tex) {
            emit_mesh(im, matrix, vertices, &indices, tex);
            buffer_clear(&indices);
            tex = face->tex;
        }

        /* 
         * the first 6 bits of face_type define which directions the face is
         * visible from.
         * opaque_neighbors is a bit mask showing which directions have
         * non-transparent blocks.
         * Therefore, only render the face if at least one of the directions
         * has a transparent block
         */
        if ((face->face_type & FACE_TYPE_MASK) & ~opaque_neighbors) {
            buffer_append(&indices, &face->p, 3);
        }
    }

    /* Render any outstanding faces */
    emit_mesh(im, matrix, vertices, &indices, tex);

    buffer_free(&indices);
}

static inline void set_biome_color(RenderState *state, BlockDef *def, OILPixel *out) {
    int tmp;
    if (def->biomecolors) {
        *out = *oil_image_get_pixel(def->biomecolors, 0, 0);
    }
}

static PyObject *render(PyObject *self, PyObject *args) {
    RenderState state;
    PyOILImage *im;
    PyOILMatrix *mat;
    PyObject *pyblockdefs;
    Buffer blockvertices;

    int i, j;
    int x, y, z;
    
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
    
    /* set up the mesh buffer */
    buffer_init(&blockvertices, sizeof(OILVertex), BLOCK_BUFFER_SIZE);
    
    /* set up the random number generator in a known state per chunk */
    srand(1);
    
    /* Here starts the loop over every block in this chunk section */
    for (x = 0; x < 16; x++) {
        for (z = 0; z < 16; z++) {
            for (y = 15; y >= 0; y--) {
                FaceType opaque_neighbors = 0;

                /* Get the model to use */
                BlockDef *def = get_block_def(&state, x, y, z);
                BlockModel *model = get_block_model(&state, def, x, y, z);
                if (!model)
                    continue;
                
                /* Clear out the block vertices buffer and then fill it with
                 * this block's vertices. We have to copy this buffer because
                 * we adjust the vertices below. */
                buffer_clear(&blockvertices);
                buffer_append(&blockvertices, model->vertices.data, model->vertices.length);

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
                
                /* Do biome coloring... */
                for (i = 0; i < model->faces.length; i++) {
                    FaceDef *f = &((FaceDef *)(model->faces.data))[i];
                    if ((f->face_type & FACE_BIOME_COLORED) == 0)
                        continue;
                    
                    /* assume that biome-colored verts and other verts are
                     * mutually-exclusive. */
                    for (j = 0; j < 3; j++) {
                        OILVertex *v = &((OILVertex *)(blockvertices.data))[f->p[j]];
                        set_biome_color(&state, def, &(v->color));
                    }
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

                /* Now draw the block. The render_block function takes care of
                 * figuring out what faces to draw based on the above.
                 * Skip calling this if there is an opaque block in every direction. */
                if (opaque_neighbors != FACE_TYPE_MASK)
                    render_block(state.im, state.matrix, &blockvertices, &model->faces, opaque_neighbors);
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
    BlockDef *d;
    BlockModel *m;
    
    /* For each block definition... */
    for (i = 0; i < defs->blockdefs_length; i++) {
        d = &(defs->defs[i]);
        if (!d->known) {
            /* Necessary because unused block definitions won't have allocated
             * a BlockModel array */
            continue;
        }

        /* Go through each block model ... */
        for (j=0; j < d->models_length; j++) {
            m = &(d->models[j]);
            if (!m->known) {
                /* not *really* necessary but only because buffer_free() won't
                 * attempt to free a null pointer and we use calloc to allocate
                 * all the BlockModel structs */
                continue;
            }
            /* And free these buffers */
            buffer_free(&(m->vertices));
            buffer_free(&(m->faces));
        }

        /* continue deallocation for this block definition */
        
        /* Free the model array */
        free(d->models);

        /* Deallocate the data parameter, if any */
        if (d->dataparameter && d->datatype.end) {
            d->datatype.end(d->dataparameter);
        }
    }
    /* Free the def array */
    free(defs->defs);
    Py_DECREF(defs->images);
    
    free(defs);
}

/*
 * Helper to compile a block model definition. One or more models makes up a
 * block definition.
 */
inline static int compile_model_definition(PyObject *pytextures, BlockModel *model, PyObject *pydef, PyObject *images)
{
    unsigned int i;
    unsigned int vertices_length;
    unsigned int triangles_length;
    PyObject *pyvertfast;
    PyObject *pytrifast;
    PyObject *pytriangles = PyObject_GetAttrString(pydef, "triangles");
    PyObject *pyvertices = PyObject_GetAttrString(pydef, "vertices");
    if (!pyvertices || !pytriangles) {
        Py_XDECREF(pyvertices);
        Py_XDECREF(pytriangles);
        return 0;
    }

    /* Turn the vertices and triangles sequences into tuples for fast access.
     * Throw away the original objects.
     */
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
    
    buffer_init(&(model->vertices), sizeof(OILVertex), BLOCK_BUFFER_SIZE);
    buffer_init(&(model->faces), sizeof(FaceDef), BLOCK_BUFFER_SIZE);

    /* Now that we have allocated memory, set this entry to known so that if we
     * return 0, the destructor will properly deallocate this definition.
     */
    model->known = 1;

    /* Load the vertices list */
    for (i = 0; i < vertices_length; i++) {
        OILVertex vert;
        PyObject *pyvert = PySequence_Fast_GET_ITEM(pyvertfast, i);
        if (!PyArg_ParseTuple(pyvert, "(fff)(ff)(bbbb)", &(vert.x), &(vert.y), &(vert.z), &(vert.s), &(vert.t), &(vert.color.r), &(vert.color.g), &(vert.color.b), &(vert.color.a))) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            PyErr_SetString(PyExc_ValueError, "vertex has invalid form. expected ((float, float, float), (float, float), (byte, byte, byte))");
            return 0;
        }
        buffer_append(&(model->vertices), &vert, 1);
    }

    /* Load the faces (triangles) list */
    for (i = 0; i < triangles_length; i++) {
        FaceDef face;
        unsigned int type;
        PyObject *texstring;
        PyObject *texobj;
        PyObject *triangle = PySequence_Fast_GET_ITEM(pytrifast, i);
        /* The three points of the triangle get loaded driectly into the face struct */
        if (!PyArg_ParseTuple(triangle, "(III)IO", &face.p[0], &face.p[1], &face.p[2], &type,
                              &texstring)) {
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            PyErr_SetString(PyExc_ValueError, "triangle has invalid form. expected ((int, int, int), int, str)");
            return 0;
        }

        /* Load the texture for this face. We rely on the Texture object
         * caching textures that are already loaded (or this would create many
         * copies of the same texture in memory)
         */
        texobj = PyObject_CallMethod(pytextures, "load", "O", texstring);
        if (!texobj || !PyObject_TypeCheck(texobj, PyOILImageType)) {
            if (texobj) {
                PyErr_SetString(PyExc_TypeError, "Textures.load() did not return an oil.Image object");
                Py_DECREF(texobj);
            }
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            return 0;
        }
        /* Make sure we have a reference to this image for the lifetime of these
         * block definitions */
        if (PySet_Add(images, texobj) == -1) {
            Py_DECREF(texobj);
            Py_DECREF(pyvertfast);
            Py_DECREF(pytrifast);
            return 0;
        }
        Py_DECREF(texobj); /* our reference is in the above set */
        face.tex = ((PyOILImage *)texobj)->im;

        /* Now load the face type */
        face.face_type = (FaceType) type;
        if ((face.face_type & FACE_TYPE_MASK) == 0) {
            /* No face direction information given. Assume it faces all
             * directions instead. This bit of logic should probably go
             * someplace more conspicuous.
             */
            face.face_type |= FACE_TYPE_MASK;
        }

        /* Finally, copy this face definition into the faces buffer */
        buffer_append(&(model->faces), &face, 1);
    }

    Py_DECREF(pyvertfast);
    Py_DECREF(pytrifast);

    return 1;
    
}

/* helper for compile_block_definitions.
 * pytextures argument is the python Textures object
 * def is the BlockDef struct we are to mutate according to the python
 * BlockDefinition object in pydef
 * images is a python set. We should add all oil.Image objects to it so
 * we can guarantee they are not garbage collected.
 * */
inline static int compile_block_definition(PyObject *pytextures, BlockDef *def, PyObject *pydef, PyObject *images) {
    int i;
    PyObject *models;
    PyObject *modelsfast;
    PyObject *prop;
    int prop_istrue = 0;

    /* First fill in the properties of this block model */
    /* transparent */
    prop = PyObject_GetAttrString(pydef, "transparent");
    if (prop) {
        def->transparent = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        return 0;
    }

    /* solid */
    prop = PyObject_GetAttrString(pydef, "solid");
    if (prop) {
        def->solid = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        return 0;
    }

    /* fluid */
    prop = PyObject_GetAttrString(pydef, "fluid");
    if (prop) {
        def->fluid = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        return 0;
    }

    /* nospawn */
    prop = PyObject_GetAttrString(pydef, "nospawn");
    if (prop) {
        def->nospawn = prop_istrue = PyObject_IsTrue(prop);
        Py_DECREF(prop);
    }
    if (!prop || prop_istrue == -1) {
        return 0;
    }

    /* biomecolors */
    prop = PyObject_GetAttrString(pydef, "biomecolors");
    if (prop) {
        if (prop == Py_None) {
            def->biomecolors = NULL;
        } else {
            PyObject *texobj = PyObject_CallMethod(pytextures, "load", "O", prop);
            if (!texobj || !PyObject_TypeCheck(texobj, PyOILImageType)) {
                if (texobj) {
                    PyErr_SetString(PyExc_TypeError, "Textures.load() did not return an oil.Image object");
                    Py_DECREF(texobj);
                }
                return 0;
            }
            
            /* ensure we hold on to this for the lifetime of the
               compiled defs */
            if (PySet_Add(images, texobj) == -1) {
                Py_DECREF(texobj);
                return 0;
            }
            
            def->biomecolors = ((PyOILImage *)texobj)->im;
            Py_DECREF(texobj);
        }
        Py_DECREF(prop);
    }
    if (!prop) {
        return 0;
    }

    /* Now determine which data function to use */
    prop = PyObject_GetAttrString(pydef, "datatype");
    if (!prop) {
        return 0;
    }
    /* the block definition holds the DataType struct itself, not a pointer to
     * it. Is this worth it? saves a dereference later I suppose. */
    {
        int datatype_index = PyInt_AsLong(prop);
        Py_DECREF(prop);
        if (datatype_index == -1 && PyErr_Occurred())
            return 0;
        if (datatype_index >= chunkrenderer_datatypes_length) {
            PyErr_SetString(PyExc_ValueError, "This is not a defined Data Type");
            return 0;
        }
        memcpy(&def->datatype, &chunkrenderer_datatypes[datatype_index], sizeof(DataType));
    }

    /* If the datatype declares a start function, get the parameter */
    if (def->datatype.start) {
        PyObject *pyparam = PyObject_GetAttrString(pydef, "dataparameter");
        if (!pyparam) {
            return 0;
        }
        def->dataparameter = def->datatype.start(pyparam);
        Py_DECREF(pyparam);
        if (!def->dataparameter) {
            return 0;
        }
    } else {
        /* block defs are initialized with calloc, but this can't hurt */
        def->dataparameter = NULL;
    }

    /* Now get the models list and allocate memory for the BlockModel array */
    models = PyObject_GetAttrString(pydef, "models");
    if (!models) {
        return 0;
    }
    modelsfast = PySequence_Fast(models, "models was not a sequence");
    Py_DECREF(models);
    if (!modelsfast) {
        return 0;
    }
    def->models_length = PySequence_Fast_GET_SIZE(modelsfast);
    def->models = calloc(def->models_length, sizeof(BlockModel));
    if (!(def->models)) {
        PyErr_SetString(PyExc_RuntimeError, "out of memory");
        Py_DECREF(modelsfast);
        return 0;
    }

    /* At this point we have allocated memory. Mark this BlockDefinition as
     * "known" so that if the destructor routine is run, memory is properly
     * freed. We don't free it ourself in event of an exception because models
     * may or may not have allocated memory themselves. So we let the caller
     * call free_block_definitions to properly free *everything* if we return
     * 0.
     */
    def->known = 1;

    for (i=0; i<def->models_length; i++) {
        /* modeldef is a borrowed reference */
        PyObject *modeldef = PySequence_Fast_GET_ITEM(modelsfast, i);
        if (modeldef == Py_None) {
            /* no definition for this data value */
            continue;
        }
        if (!compile_model_definition(pytextures,
                                      &(def->models[i]),
                                      modeldef,
                                      images)) {
            Py_DECREF(modelsfast);
            return 0;
        }
    }

    Py_DECREF(modelsfast);
    
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

    /* Create a set to hold the images, so they don't get garbage collected */
    defs->images = PySet_New(NULL);
    
    pymaxblockid = PyObject_GetAttrString(pyblockdefs, "max_blockid");
    if (!pymaxblockid) {
        Py_DECREF(defs->images);
        free(defs);
        return NULL;
    }
    if (!PyInt_Check(pymaxblockid)) {
        Py_DECREF(pymaxblockid);
        Py_DECREF(defs->images);
        free(defs);
        PyErr_SetString(PyExc_TypeError, "max_blockid was not an integer wtf are you trying to pull here");
        return NULL;
    }
    /* Add 1 since the number of blocks we can actually have includes index 0,
     * so the array needs to be one larger than this number */
    defs->blockdefs_length = PyInt_AsLong(pymaxblockid) + 1;
    Py_DECREF(pymaxblockid);
    
    pyblocks = PyObject_GetAttrString(pyblockdefs, "blocks");
    if (!pyblocks) {
        Py_DECREF(defs->images);
        free(defs);
        return NULL;
    }
    if (!PyDict_Check(pyblocks)) {
        PyErr_SetString(PyExc_TypeError, "blocks is not a dictionary");
        Py_DECREF(defs->images);
        Py_DECREF(pyblocks);
        free(defs);
        return NULL;
    }

    /* Important to use calloc so the "known" member of the BlockDef struct is
     * by default 0; we may have gaps in our known block array */
    defs->defs = calloc(defs->blockdefs_length, sizeof(BlockDef));
    if (!(defs->defs)) {
        PyErr_SetString(PyExc_RuntimeError, "out of memory");
        Py_DECREF(defs->images);
        free(defs);
        Py_DECREF(pyblocks);
        return NULL;
    }

    /* At this point, defs is valid enough such that we can run it through
     * free_block_definitions() if we need to clean up. */
    
    /* Loop over every block up to blockdefs_length and see if there's an entry for
     * it in the pyblocks dict */
    for (blockid = 0; blockid < defs->blockdefs_length; blockid++) {
        PyObject *key = PyInt_FromLong(blockid);
        PyObject *val;
    
        if (!key) {
            free_block_definitions(defs);
            Py_DECREF(pyblocks);
            return NULL;
        }
    
        val = PyDict_GetItem(pyblocks, key);
        if (val) {
            /* Compile a block definition from the BlockDefinition object in
             * val */
            if (!compile_block_definition(pytextures,
                                          &(defs->defs[blockid]),
                                          val,
                                          defs->images)
                                          ) {
                free_block_definitions(defs);
                Py_DECREF(pyblocks);
                return NULL;
            }
        }
                    
        Py_DECREF(key);   
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
    BlockModel *model;
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

    if (blockid >= bd->blockdefs_length) {
        PyErr_SetString(PyExc_ValueError, "No such block with that ID exists");
        return NULL;
    }

    /* Look up this block's definition */
    block = &(bd->defs[blockid]);

    if (!block->known) {
        PyErr_SetString(PyExc_ValueError, "No such block with that ID exists");
        return NULL;
    }

    if (data >= block->models_length) {
        PyErr_SetString(PyExc_ValueError, "No block model with that data value exists");
        return NULL;
    }

    model = &(block->models[data]);

    render_block(im->im, &mat->matrix, &model->vertices, &model->faces, 0);
    Py_RETURN_NONE;
}

static PyMethodDef chunkrenderer_methods[] = {
    {"render", render, METH_VARARGS,
     "Render a chunk to an image."},
    {"compile_block_definitions", compile_block_definitions, METH_VARARGS,
     "Compiles a Textures object and a BlockDefinitions object into a form usable by the render method."},
    {"render_block", (PyCFunction)py_render_block, METH_VARARGS | METH_KEYWORDS,
     "Renders a single block to the given image with the given matrix"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initchunkrenderer(void) {
    PyObject *mod, *numpy;
    DataType *dt_def;
    int i;
    
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
    PyModule_AddIntConstant(mod, "FACE_TYPE_MASK", FACE_TYPE_MASK);
    PyModule_AddIntConstant(mod, "FACE_BIOME_COLORED", FACE_BIOME_COLORED);

    /* Add the data function pointers to the module. */
    dt_def = chunkrenderer_datatypes;
    for (i=0; i < chunkrenderer_datatypes_length; i++) {
        PyModule_AddObject(mod, dt_def->name,
                PyInt_FromLong(i)
        );
        dt_def++;
    }
    
    /* tell the compiler to shut up about unused things
       sizeof(...) does not evaluate its argument (:D) */
    (void)sizeof(import_array());
    
    /* import numpy on our own, because import_array breaks across
       numpy versions and we barely use numpy */
    numpy = PyImport_ImportModule("numpy.core.multiarray");
    Py_XDECREF(numpy);
}
