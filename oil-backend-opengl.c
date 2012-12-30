#ifdef ENABLE_OPENGL_BACKEND

#include "oil.h"
#include "oil-image-private.h"
#include "oil-backend-private.h"

#include <stdio.h>

#include <stddef.h>
#include <string.h>

#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

static const char *vshadersrc = " \
void main() { \
    gl_Position = gl_ModelViewMatrix * gl_Vertex; \
    gl_FrontColor = gl_Color; \
    gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0; \
} \
";

static const char *fshadersrc = " \
uniform sampler2D tex; \
void main() { \
    gl_FragColor = gl_Color * texture2D(tex, gl_TexCoord[0]); \
} \
";

typedef struct {
    GLuint framebuffer;
    GLuint depthbuffer;
    GLuint colorbuffer;
} OpenGLPriv;

static OILMatrix modelview = {{{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}, {0.0, 0.0, 0.0, 1.0}}};
static GLuint framebuffer = 0;
static GLuint colorbuffer = 0;

static int oil_backend_opengl_initialize() {
    static int initialized = 0;
    static int initialization_result = 0;
    Display *dpy;
    Window root;
    GLint attr[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
    XVisualInfo *vi;
    GLXContext glc;
    int tmp;
    GLuint vshader, fshader;
    GLuint shader;
    char infolog[1024];
    
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
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CW);
    
    glEnable(GL_ALPHA_TEST);
    glAlphaFunc(GL_GREATER, 0.0f);
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);
        
    /* set up the texture matrix to handle our flipped images */
    glMatrixMode(GL_TEXTURE);
    glScalef(1.0, -1.0, 1.0);
    glTranslatef(0.0, 1.0, 0.0);
    glMatrixMode(GL_MODELVIEW);
    
    vshader = glCreateShader(GL_VERTEX_SHADER);
    tmp = strlen(vshadersrc);
    glShaderSource(vshader, 1, &vshadersrc, &tmp);
    glCompileShader(vshader);
    glGetShaderiv(vshader, GL_COMPILE_STATUS, &tmp);
    if (tmp == GL_FALSE) {
        glGetShaderInfoLog(vshader, 1024, &tmp, infolog);
        infolog[tmp] = 0;
        printf("%s", infolog);
        return 0;
    }
    
    fshader = glCreateShader(GL_FRAGMENT_SHADER);
    tmp = strlen(fshadersrc);
    glShaderSource(fshader, 1, &fshadersrc, &tmp);
    glCompileShader(fshader);
    glGetShaderiv(fshader, GL_COMPILE_STATUS, &tmp);
    if (tmp == GL_FALSE) {
        glGetShaderInfoLog(fshader, 1024, &tmp, infolog);
        infolog[tmp] = 0;
        printf("%s", infolog);
        return 0;
    }
    
    shader = glCreateProgram();
    glAttachShader(shader, vshader);
    glAttachShader(shader, fshader);
    glLinkProgram(shader);
    glGetProgramiv(shader, GL_LINK_STATUS, &tmp);
    if (tmp == GL_FALSE) {
        glGetProgramInfoLog(shader, 1024, &tmp, infolog);
        infolog[tmp] = 0;
        printf("%s", infolog);
        return 0;
    }
    
    glUseProgram(shader);

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
    colorbuffer = priv->colorbuffer;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, im->width, im->height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, NULL);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    
    /* set up our depth buffer */
    glBindRenderbuffer(GL_RENDERBUFFER, priv->depthbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, im->width, im->height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    
    /* bind everything to our framebuffer */
    glBindFramebuffer(GL_FRAMEBUFFER, priv->framebuffer);
    framebuffer = priv->framebuffer;
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
    framebuffer = 0;
}

static void oil_backend_opengl_free(OILImage *im) {
    /* free our data struct */
    OpenGLPriv *priv = im->backend_data;
    
    if (colorbuffer == priv->colorbuffer) {
        glBindTexture(GL_TEXTURE_2D, 0);
        colorbuffer = 0;
    }
    
    if (framebuffer == priv->framebuffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        framebuffer = 0;
    }
    
    glDeleteFramebuffers(1, &(priv->framebuffer));
    glDeleteRenderbuffers(1, &(priv->depthbuffer));
    glDeleteTextures(1, &(priv->colorbuffer));
    
    free(priv);
}

static inline void load_matrix(OILMatrix *matrix) {
    if (memcmp(matrix, &modelview, sizeof(OILMatrix)) != 0) {
        OILMatrix fullmat;
        oil_matrix_set_identity(&fullmat);
        oil_matrix_scale(&fullmat, 1.0f, -1.0f, -1.0f);
        oil_matrix_multiply(&fullmat, &fullmat, matrix);
        glLoadTransposeMatrixf((GLfloat *)fullmat.data);
        oil_matrix_copy(&modelview, matrix);
    }
}

static inline void bind_framebuffer(OILImage *im) {
    if (im) {
        OpenGLPriv *priv = im->backend_data;
        if (framebuffer != priv->framebuffer) {
            glBindFramebuffer(GL_FRAMEBUFFER, priv->framebuffer);
            glViewport(0, 0, im->width, im->height);
            framebuffer = priv->framebuffer;
        }
    } else {
        if (framebuffer != 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            framebuffer = 0;
        }
    }
}

static inline void bind_colorbuffer(OILImage *im) {
    if (im) {
        OpenGLPriv *priv = im->backend_data;
        if (colorbuffer != priv->colorbuffer) {
            glBindTexture(GL_TEXTURE_2D, priv->colorbuffer);
            colorbuffer = priv->colorbuffer;
        }
    } else {
        if (colorbuffer != 0) {
            glBindTexture(GL_TEXTURE_2D, 0);
            colorbuffer = 0;
        }
    }
}

static void oil_backend_opengl_load(OILImage *im) {
    bind_framebuffer(im);
    glReadPixels(0, 0, im->width, im->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, im->data);
}

static void oil_backend_opengl_save(OILImage *im) {
    bind_colorbuffer(im);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, im->width, im->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, im->data);
}

static void oil_backend_opengl_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags);

static int oil_backend_opengl_composite(OILImage *im, OILImage *src, unsigned char alpha, int dx, int dy, unsigned int sx, unsigned int sy, unsigned int xsize, unsigned int ysize) {
    OILVertex vertices[] = {
        {0.0, 0.0, 0.0, ((float)sx) / src->width, ((float)src->height - sy) / src->height, {255, 255, 255, 255}},
        {1.0, 0.0, 0.0, ((float)sx + xsize) / src->width, ((float)src->height - sy) / src->height, {255, 255, 255, 255}},
        {1.0, 1.0, 0.0, ((float)sx + xsize) / src->width, ((float)src->height - sy - ysize) / src->height, {255, 255, 255, 255}},
        {0.0, 1.0, 0.0, ((float)sx) / src->width, ((float)src->height - sy - ysize) / src->height, {255, 255, 255, 255}},
    };
    unsigned int indices[] = {
        0, 2, 1,
        0, 3, 2,
    };
    OILMatrix mat;
    oil_matrix_set_identity(&mat);
    oil_matrix_orthographic(&mat, 0, im->width, im->height, 0, -1.0, 1.0);
    oil_matrix_translate(&mat, dx, dy, 0.0);
    oil_matrix_scale(&mat, xsize, ysize, 1.0);
    
    oil_backend_opengl_draw_triangles(im, &mat, src, vertices, 4, indices, 6, 0);
    
    return 1;
}

static void oil_backend_opengl_draw_triangles(OILImage *im, OILMatrix *matrix, OILImage *tex, OILVertex *vertices, unsigned int vertices_length, unsigned int *indices, unsigned int indices_length, OILTriangleFlags flags) {
    unsigned int i;
    
    load_matrix(matrix);
    bind_framebuffer(im);
    bind_colorbuffer(tex);
    
    /* set up any drawing flags */
    if (!(flags & OIL_DEPTH_TEST)) {
        glDisable(GL_DEPTH_TEST);
    }

    glBegin(GL_TRIANGLES);
    for (i = 0; i < indices_length; i += 3) {
        OILVertex v0 = vertices[indices[i]];
        OILVertex v1 = vertices[indices[i + 1]];
        OILVertex v2 = vertices[indices[i + 2]];
        
        glTexCoord2f(v0.s, v0.t);
        glColor4f(v0.color.r / 255.0, v0.color.g / 255.0, v0.color.b / 255.0, v0.color.a / 255.0);
        glVertex3f(v0.x, v0.y, v0.z);

        glTexCoord2f(v1.s, v1.t);
        glColor4f(v1.color.r / 255.0, v1.color.g / 255.0, v1.color.b / 255.0, v1.color.a / 255.0);
        glVertex3f(v1.x, v1.y, v1.z);

        glTexCoord2f(v2.s, v2.t);
        glColor4f(v2.color.r / 255.0, v2.color.g / 255.0, v2.color.b / 255.0, v2.color.a / 255.0);
        glVertex3f(v2.x, v2.y, v2.z);
    }
    glEnd();
    
    /* undo our drawing flags */
    if (!(flags & OIL_DEPTH_TEST)) {
        glEnable(GL_DEPTH_TEST);
    }    
}

static void oil_backend_opengl_clear(OILImage *im) {
    bind_framebuffer(im);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

OILBackend oil_backend_opengl = {
    oil_backend_opengl_initialize,
    oil_backend_opengl_new,
    oil_backend_opengl_free,
    oil_backend_opengl_load,
    oil_backend_opengl_save,
    oil_backend_opengl_composite,
    oil_backend_opengl_draw_triangles,
    oil_backend_opengl_clear,
};

#endif /* ENABLE_OPENGL_BACKEND */
