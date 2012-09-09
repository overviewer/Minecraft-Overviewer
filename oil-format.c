#include "oil.h"
#include "oil-format-private.h"

#include <stdlib.h>

extern OILFormat oil_format_png;

OILFormat *oil_formats[] = {
    &oil_format_png,
    NULL
};
