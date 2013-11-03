#include "oil.h"
#include "oil-format-private.h"

#include <stdlib.h>

#define FORMAT(name, symbol) extern OILFormat symbol;
#include "oil-formats.cdef"
#undef FORMAT

OILFormat *oil_formats[] = {
#define FORMAT(name, symbol) &symbol,
#include "oil-formats.cdef"
#undef FORMAT
    NULL
};
