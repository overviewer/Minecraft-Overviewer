#include "oil.h"
#include "oil-image-private.h"
#include "oil-backend-private.h"

#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

static int oil_backend_opengl_initialize() {
    static int initialized = 0;
    static int initialization_result = 0;
    Display *dpy;
    Window root;
    GLint attr[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
    XVisualInfo *vi;
    GLXContext glc;
    
    /* only run this once, ever */
    if (initialized)
        return initialization_result;
    initialized = 1;
    
    /* open display */
    if (!(dpy = XOpenDisplay(NULL))) {
        return 0;
    }
    
    /* get root window */
    root = DefaultRootWindow(dpy);
    
    /* get visual matching attr */
    if (!(vi = glXChooseVisual(dpy, 0, attr))) { 
        return 0;
    }
    
    /* create a context */
    if (!(glc = glXCreateContext(dpy, vi, NULL, GL_TRUE))) {
        return 0;
    }
    
    glXMakeCurrent(dpy, root, glc);
    
    /* printf("vendor: %s\n", (const char*)glGetString(GL_VENDOR)); */
    
    initialization_result = 1;
    return 1;
}

static void oil_backend_opengl_new(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_opengl_free(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_opengl_load(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_opengl_save(OILImage *im) {
    /* nothing to do */
}

static int oil_backend_opengl_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    return 1;
}

static void oil_backend_opengl_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    /* nothing to do */
}

OILBackend oil_backend_opengl = {
    oil_backend_opengl_initialize,
    oil_backend_opengl_new,
    oil_backend_opengl_free,
    oil_backend_opengl_load,
    oil_backend_opengl_save,
    oil_backend_opengl_composite,
    oil_backend_opengl_draw_triangles,
};
