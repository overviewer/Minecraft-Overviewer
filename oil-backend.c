#include "oil.h"
#include "oil-backend-private.h"

extern OILBackend oil_backend_cpu;
extern OILBackend oil_backend_debug;
extern OILBackend oil_backend_cpu_sse;
extern OILBackend oil_backend_opengl;

OILBackend *oil_backend = &oil_backend_cpu;

int oil_backend_set(OILBackendName backend) {
    OILBackend *new_backend;
    
    switch (backend) {
    case OIL_BACKEND_CPU:
        new_backend = &oil_backend_cpu;
        break;
    case OIL_BACKEND_DEBUG:
        new_backend = &oil_backend_debug;
        break;
    case OIL_BACKEND_CPU_SSE:
        new_backend = &oil_backend_cpu_sse;
        break;
    case OIL_BACKEND_OPENGL:
        new_backend = &oil_backend_opengl;
        break;
    default:
        /* invalid backend */
        return 0;
    };
    
    /* try to initialize new backend */
    if (!(new_backend->initialize())) {
        /* initialization failed! keep the old backend */
        return 0;
    }
    
    /* new backend online! */
    oil_backend = new_backend;
    return 1;
}
