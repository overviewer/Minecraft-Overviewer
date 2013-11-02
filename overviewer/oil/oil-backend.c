#include "oil.h"
#include "oil-backend-private.h"

#define BACKEND(name, symbol) extern OILBackend symbol;
#include "oil-backends.cdef"
#undef BACKEND

/* default backend */
OILBackend *oil_backend = &oil_backend_cpu;

int oil_backend_set(OILBackendName backend) {
    OILBackend *new_backend;
    
    switch (backend) {
#define BACKEND(name, symbol)                   \
        case OIL_BACKEND_##name:                \
            new_backend = &symbol;              \
            break;
#include "oil-backends.cdef"
#undef BACKEND
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
