#include "oil.h"
#include "oil-backend-private.h"

static void oil_backend_cpu_new(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_free(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_load(OILImage *im) {
    /* nothing to do */
}

static void oil_backend_cpu_save(OILImage *im) {
    /* nothing to do */
}

OILBackend oil_backend_cpu = {
    oil_backend_cpu_new,
    oil_backend_cpu_free,
    oil_backend_cpu_load,
    oil_backend_cpu_save,
};
