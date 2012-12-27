#include "oil.h"
#include "oil-backend-private.h"

extern OILBackend oil_backend_cpu;
extern OILBackend oil_backend_debug;

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
