#include "oil.h"
#include "oil-backend-private.h"

extern OILBackend oil_backend_cpu;
extern OILBackend oil_backend_debug;

OILBackend *oil_backend = &oil_backend_cpu;

void oil_backend_set(OILBackendName backend) {
    switch (backend) {
    case OIL_BACKEND_CPU:
        oil_backend = &oil_backend_cpu;
        break;
    case OIL_BACKEND_DEBUG:
        oil_backend = &oil_backend_debug;
        break;
    };
}
