#ifndef __CHUNKRENDERER_H_INCLUDED__
#define __CHUNKRENDERER_H_INCLUDED__

#include <oil.h>
#include <oil-python.h>

#include "buffer.h"

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

/* forward declaration */
typedef struct _blockdefs_struct BlockDefs;

/* This struct holds information necessary to render a particular chunk section
 * */
typedef struct {
    PyObject *regionset;
    /* chunky is the section number within the chunk */
    int chunkx, chunky, chunkz;
    ChunkData chunks[3][3];
    
    OILImage *im;
    OILMatrix *matrix;
    
    BlockDefs *blockdefs;
} RenderState;


/* Properties a model face can have. These correspond to bitfields in the
 * FaceDef struct below */
enum {
    FACE_TYPE_PX=(1<<0),
    FACE_TYPE_NX=(1<<1),
    FACE_TYPE_PY=(1<<2),
    FACE_TYPE_NY=(1<<3),
    FACE_TYPE_PZ=(1<<4),
    FACE_TYPE_NZ=(1<<5),
    FACE_TYPE_MASK=(1<<6)-1,
    /* Lower 6 bits are the face directions. upper bits are other flags. */
    /* BIOME_COLOED tells the renderer to apply biome coloring to this face */
    FACE_BIOME_COLORED=(1<<6)
};
typedef unsigned char FaceType;

/* Represents a triangle, as part of a block model definition */
typedef struct {
    /* these are the three points of the triangle. They reference an index into
     * the vertices array of the block definition */
    unsigned int p[3];

    /* Flags a face may have. See the FaceType enum */
    FaceType face_type;

    /* The texture to draw on this face */
    OILImage *tex;
} FaceDef;

/* Defines the mesh for a particular block model */
typedef struct {
    unsigned int known:1;
    /* This is a buffer of OILVertex structs */
    Buffer vertices;
    /* This is a buffer of FaceDef structs */
    Buffer faces;
} BlockModel;


typedef struct {
    /* The function for returning a data value given the world and the block to
     * analyze */
    unsigned int (*datafunc)(void* param, RenderState *state, int x, int y, int z);
    /* This function takes a python parameter and translates it into some kind
     * of C structure that is used with the above data func call. A returned null
     * pointer indicates an error. A null function pointer indicates no data
     * parameter is used (a null pointer will be passed in to the datafunc) */
    void * (*start)(PyObject *param);
    /* This function deallocates whatever structure was allocated above */
    void (*end)(void *param);

    char * name;

} DataType;

typedef struct {
    unsigned int known: 1;

    BlockModel *models;
    unsigned int models_length;
    
    unsigned int transparent: 1;
    unsigned int solid: 1;
    unsigned int fluid: 1;
    unsigned int nospawn: 1;
    
    OILImage *biomecolors;

    /* The data function, used to determine which block model to use */
    DataType datatype;
    void * dataparameter;
} BlockDef;

struct _blockdefs_struct {
    BlockDef *defs;
    unsigned int blockdefs_length;

    /* This is a python set of the images used by the block definitions. This
     * keeps references to the images so that if the textures object gets
     * garbage collected, the images we use won't be freed */
    PyObject *images;
};

/*
 * Prototypes for chunkrenderer convenience functions, defined in
 * chunkrenderer.c, used by chunkrenderer.c and blockdata.c
 */

enum _get_data_type_enum
{
    BLOCKS,
    DATA,
    BLOCKLIGHT,
    SKYLIGHT,
    BIOMES,
};

extern unsigned int get_data(RenderState *state, enum _get_data_type_enum type, int x, int y, int z);

typedef enum {
    KNOWN,
    TRANSPARENT,
    SOLID,
    FLUID,
    NOSPAWN,
} BlockProperty;
inline int block_has_property(RenderState *state, unsigned short b, BlockProperty prop);

/* DataType declarations (defined in blockdata.c) */
extern DataType chunkrenderer_datatypes[];
extern const int chunkrenderer_datatypes_length;


#endif /* __CHUNKRENDERER_H_INCLUDED__ */
