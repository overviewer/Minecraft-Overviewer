#ifndef __INDEXING_H_INCLUDED__
#define __INDEXING_H_INCLUDED__

/* OpenGL-style naming conventions, god help me.
 * b = byte (8 bits)
 * s = short (16 bytes)
 * n = nibble (4 bits)
 * # = dimensions of the array
 *
 * 2d arrays are 16 x 16, 3d arrays are 16 x 16 x 16
 *
 * So, bytes3n(pybuf, x, y, z) will get a nibble out of a 3d array in 
 * a buffer Python bytes object in pybuf.
 *
 * Keep in mind that minecraft uses y/z/x order for 3d arrays, and
 * z/x order for 2d. (most-significant to least-significant)
 * Also, for nibbles, the least-significant half of each byte is first.
 */

/* first, 1d indexing for each type */
#define bytes1b(b, i) (((unsigned char *)PyBytes_AS_STRING(b))[i])
#define bytes1n(b, i) (((i) & 1) ? (bytes1b(b, (i) >> 1) >> 4) : (bytes1b(b, (i) >> 1) & 0x0F))

/* then, transformations from n-dimensions to 1 */
#define __index2d(x, z) ((x) + (z) * 16)
#define __index3d(x, y, z) ((x) + (z) * 16 + (y) * 256)

/* now, rote combinatorics! */
#define bytes2b(b, x, z) bytes1b(b, __index2d(x, z))
#define bytes2n(b, x, z) bytes1n(b, __index2d(x, z))
#define bytes3b(b, x, y, z) bytes1b(b, __index3d(x, y, z))
#define bytes3n(b, x, y, z) bytes1n(b, __index3d(x, y, z))

#endif /* __INDEXING_H_INCLUDED__ */
