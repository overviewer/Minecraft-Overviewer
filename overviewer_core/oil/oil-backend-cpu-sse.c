#ifdef ENABLE_CPU_SSE_BACKEND

/* include the cpu template with the right name */
#define CPU_BACKEND_NAME oil_backend_cpu_sse
#define SSE
#include "oil-backend-cpu.def"

#endif /* ENABLE_CPU_SSE_BACKEND */
