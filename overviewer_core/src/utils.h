#ifndef __OV_UTILS_H_INCLUDED__
#define __OV_UTILS_H_INCLUDED__

/* generally useful MAX / MIN macros */
#define OV_MAX(a, b) ((a) > (b) ? (a) : (b))
#define OV_MIN(a, b) ((a) < (b) ? (a) : (b))
#define OV_CLAMP(x, a, b) (OV_MIN(OV_MAX(x, a), b))

/* like (a * b + 127) / 255), but much faster on most platforms
   from PIL's _imaging.c */
#define OV_MULDIV255(a, b, tmp) \
    (tmp = (a) * (b) + 128, ((((tmp) >> 8) + (tmp)) >> 8))

#define OV_BLEND(mask, in1, in2, tmp1, tmp2) \
    (OV_MULDIV255(in1, 255 - mask, tmp1) + OV_MULDIV255(in2, mask, tmp2))

#define COUNT_OF(array) \
    (sizeof(array) / sizeof(array[0]))

#endif
