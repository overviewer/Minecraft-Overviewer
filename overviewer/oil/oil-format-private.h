#ifndef __OIL_FORMAT_PRIVATE_H_INCLUDED__
#define __OIL_FORMAT_PRIVATE_H_INCLUDED__

#include "oil.h"

typedef struct {
    int (*save)(OILImage *im, OILFile *file, OILFormatOptions *opts);
    OILImage *(*load)(OILFile *file);
} OILFormat;

extern OILFormat *oil_formats[];

#endif /* __OIL_FORMAT_PRIVATE_H_INCLUDED__ */
