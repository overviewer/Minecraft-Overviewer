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

/*
 * To make a new render mode (the C part, at least):
 *
 *     * add a data struct and extern'd interface declaration below
 *
 *     * fill in this interface struct in rendermode-(yourmode).c
 *         (see rendermodes-normal.c for an example: the "normal" mode)
 *
 *     * if you want to derive from (say) the "normal" mode, put
 *       a RenderModeNormal entry at the top of your data struct, and
 *       be sure to call your parent's functions in your own!
 *         (see rendermode-night.c for a simple example derived from
 *          the "lighting" mode)
 *
 *     * add your mode to the list in rendermodes.c
 */

#ifndef __RENDERMODES_H_INCLUDED__
#define __RENDERMODES_H_INCLUDED__

#include <Python.h>

typedef struct {
    const char *name;
    const char *description;
} RenderModeOption;

/* rendermode interface */
typedef struct _RenderModeInterface RenderModeInterface;
struct _RenderModeInterface {
    /* the name of this mode */
    const char *name;
    /* the short description of this render mode */
    const char *description;
    
    /* a NULL-terminated list of render mode options, or NULL */
    RenderModeOption *options;
    
    /* the rendermode this is derived from, or NULL */
    RenderModeInterface *parent;
    /* the size of the local storage for this rendermode */
    unsigned int data_size;
    
    /* may return non-zero on error, last arg is options */
    int (*start)(void *, RenderState *, PyObject *);
    void (*finish)(void *, RenderState *);
    /* returns non-zero to skip rendering this block */
    int (*occluded)(void *, RenderState *);
    /* last two arguments are img and mask, from texture lookup */
    void (*draw)(void *, RenderState *, PyObject *, PyObject *, PyObject *);
};

/* wrapper for passing around rendermodes */
typedef struct {
    void *mode;
    RenderModeInterface *iface;
    RenderState *state;
} RenderMode;

/* functions for creating / using rendermodes */
RenderMode *render_mode_create(const char *mode, RenderState *state);
void render_mode_destroy(RenderMode *self);
int render_mode_occluded(RenderMode *self);
void render_mode_draw(RenderMode *self, PyObject *img, PyObject *mask, PyObject *mask_light);

/* python metadata bindings */
PyObject *get_render_modes(PyObject *self, PyObject *args);
PyObject *get_render_mode_info(PyObject *self, PyObject *args);
PyObject *get_render_mode_inheritance(PyObject *self, PyObject *args);
PyObject *get_render_mode_children(PyObject *self, PyObject *args);

/* python rendermode options bindings */
PyObject *set_render_mode_options(PyObject *self, PyObject *args);
PyObject *add_custom_render_mode(PyObject *self, PyObject *args);

/* individual rendermode interface declarations follow */

/* NORMAL */
typedef struct {
    /* coordinates of the chunk, inside its region file */
    int chunk_x, chunk_y;
    /* biome data for the region */
    PyObject *biome_data;
    /* grasscolor and foliagecolor lookup tables */
    PyObject *grasscolor, *foliagecolor;
    /* biome-compatible grass/leaf textures */
    PyObject *grass_texture, *leaf_texture, *tall_grass_texture, *tall_fern_texture;
    /* top facemask for grass biome tinting */
    PyObject *facemask_top;
    
    float edge_opacity;
    unsigned int min_depth;
    unsigned int max_depth;
} RenderModeNormal;
extern RenderModeInterface rendermode_normal;

/* OVERLAY */
typedef struct {
    /* top facemask and white color image, for drawing overlays */
    PyObject *facemask_top, *white_color;
    /* only show overlay on top of solid or fluid blocks */
    PyObject *solid_blocks, *fluid_blocks;
    /* can be overridden in derived classes to control
       overlay alpha and color
       last four vars are r, g, b, a out */
    void (*get_color)(void *, RenderState *,
                      unsigned char *, unsigned char *, unsigned char *, unsigned char *);
} RenderModeOverlay;
extern RenderModeInterface rendermode_overlay;

/* LIGHTING */
typedef struct {
    /* inherits from normal render mode */
    RenderModeNormal parent;
    
    PyObject *black_color, *facemasks_py;
    PyObject *facemasks[3];
    
    /* extra data, loaded off the chunk class */
    PyObject *skylight, *blocklight;
    PyObject *left_skylight, *left_blocklight;
    PyObject *right_skylight, *right_blocklight;
    
    /* can be overridden in derived rendermodes to control lighting
       arguments are skylight, blocklight */
    float (*calculate_darkness)(unsigned char, unsigned char);
    
    float shade_strength;
} RenderModeLighting;
extern RenderModeInterface rendermode_lighting;

/* NIGHT */
typedef struct {
    /* inherits from lighting */
    RenderModeLighting parent;
} RenderModeNight;
extern RenderModeInterface rendermode_night;

/* SPAWN */
typedef struct {
    /* inherits from overlay */
    RenderModeOverlay parent;
    
    /* used to figure out which blocks are spawnable */
    PyObject *nospawn_blocks;
    PyObject *skylight, *blocklight;
} RenderModeSpawn;
extern RenderModeInterface rendermode_spawn;

/* CAVE */
typedef struct {
    /* render blocks with lighting mode */
    RenderModeNormal parent;

    /* data used to know where the surface is */
    PyObject *skylight;
    PyObject *left_skylight;
    PyObject *right_skylight;
    PyObject *up_left_skylight;
    PyObject *up_right_skylight;

    /* colors used for tinting */
    PyObject *depth_colors;
    
    int depth_tinting;
} RenderModeCave;
extern RenderModeInterface rendermode_cave;

#endif /* __RENDERMODES_H_INCLUDED__ */
