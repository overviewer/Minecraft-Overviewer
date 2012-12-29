#ifdef ENABLE_OPENGL_BACKEND

#include "oil.h"
#include "oil-image-private.h"
#include "oil-backend-private.h"

#include <stdio.h>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

typedef struct {
    GLuint framebuffer;
    GLuint depthbuffer;
    GLuint colorbuffer;
} OpenGLPriv;

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
    
    if (glewInit() != GLEW_OK) {
        return 0;
    }
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    /* printf("vendor: %s\n", (const char*)glGetString(GL_VENDOR)); */
    
    initialization_result = 1;
    return 1;
}

static void oil_backend_opengl_new(OILImage *im) {
    /* add our data struct. FIXME fail if any of these are NULL */
    OpenGLPriv *priv;
    priv = im->backend_data = malloc(sizeof(OpenGLPriv));
    
    glGenFramebuffers(1, &(priv->framebuffer));
    glGenRenderbuffers(1, &(priv->depthbuffer));
    glGenTextures(1, &(priv->colorbuffer));
    
    /* set up the color buffer */
    glBindTexture(GL_TEXTURE_2D, priv->colorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im->width, im->height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    /* set up our depth buffer */
    glBindRenderbuffer(GL_RENDERBUFFER, priv->depthbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, im->width, im->height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    
    /* bind everything to our framebuffer */
    glBindFramebuffer(GL_FRAMEBUFFER, priv->framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, priv->colorbuffer, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, priv->depthbuffer);
    
    /* make sure we're good */
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        /* FIXME fail! */
        printf("ack! failed to make framebuffer! %i %i %i (%i, %i) 0x%x\n", priv->framebuffer, priv->colorbuffer, priv->depthbuffer, im->width, im->height, glCheckFramebufferStatus(GL_FRAMEBUFFER));
    }
    
    /* clear the default data */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void oil_backend_opengl_free(OILImage *im) {
    /* free our data struct */
    OpenGLPriv *priv = im->backend_data;
    
    glDeleteFramebuffers(1, &(priv->framebuffer));
    glDeleteRenderbuffers(1, &(priv->depthbuffer));
    glDeleteTextures(1, &(priv->colorbuffer));
    
    free(priv);
}

static void oil_backend_opengl_load(OILImage *im) {
    OpenGLPriv *priv = im->backend_data;
    glBindFramebuffer(GL_FRAMEBUFFER, priv->framebuffer);
    glReadPixels(0, 0, im->width, im->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, im->data);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void oil_backend_opengl_save(OILImage *im) {
    OpenGLPriv *priv = im->backend_data;
    glBindTexture(GL_TEXTURE_2D, priv->colorbuffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width, im->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, im->data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

static int oil_backend_opengl_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    return 1;
}

static void oil_backend_opengl_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    unsigned int i;
    OpenGLPriv *priv = im->backend_data;
    OILMatrix fullmat;
    oil_matrix_set_identity(&fullmat);
    oil_matrix_scale(&fullmat, 1.0f, -1.0f, 1.0f);
    oil_matrix_multiply(&fullmat, &fullmat, matrix);
    glLoadMatrixf((GLfloat *)fullmat.data);
    
    if (tex) {
        OpenGLPriv *texpriv = tex->backend_data;
        glBindTexture(GL_TEXTURE_2D, texpriv->colorbuffer);
        if (glGetError())
            printf("couln't bind: 0x%x\n", glGetError());
        glEnable(GL_TEXTURE_2D);
    }
    
    if (flags & OIL_DEPTH_TEST) {
        glEnable(OIL_DEPTH_TEST);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, priv->framebuffer);
    glViewport(0, 0, im->width, im->height);
    
    glBegin(GL_TRIANGLES);
    for (i = 0; i < indices_length; i += 3) {
        OILVertex v0 = vertices[indices[i]];
        OILVertex v1 = vertices[indices[i + 1]];
        OILVertex v2 = vertices[indices[i + 2]];
        
        glColor4f(v0.color.r, v0.color.g, v0.color.b, v0.color.a);
        glTexCoord2f(v0.s, v0.t);
        glVertex3f(v0.x, v0.y, v0.z);
        glColor4f(v1.color.r, v1.color.g, v1.color.b, v1.color.a);
        glTexCoord2f(v1.s, v1.t);
        glVertex3f(v1.x, v1.y, v1.z);
        glColor4f(v2.color.r, v2.color.g, v2.color.b, v2.color.a);
        glTexCoord2f(v2.s, v2.t);
        glVertex3f(v2.x, v2.y, v2.z);
    }
    glEnd();

    if (flags & OIL_DEPTH_TEST) {
        glDisable(OIL_DEPTH_TEST);
    }
    
    if (tex) {
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
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

#endif /* ENABLE_OPENGL_BACKEND */
