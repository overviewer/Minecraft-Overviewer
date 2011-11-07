__kernel void paste_gpu (__global uchar* imDest,        /* pointer to destination image data */
                         __global uchar* imSrc,         /* pointer to source image data */
                         uint srcXSize,                 /* width of src img */
                         uint srcYSize,                 /* height of src img */
                         uint dstWidth,                 /* total width of dst image, in bytes */
                         uint srcWidth,                 /* total width of the src iamge, in bytes */
                         uint dx,                       /* x location in dst */
                         uint dy                        /* y location in dst */
)
{
    /* get_global_id(0) returns the ID of the thread in execution.
    As many threads are launched at the same time, executing the same kernel,
    each one will receive a different ID, and consequently perform a different computation.*/

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    const int pixelSize = 4;


    if ((idx < srcXSize) && (idy < srcYSize)) {

    __global uchar *out = imDest + (dy + idy)*dstWidth + ((dx+idx)*pixelSize);

    __global uchar *in = imSrc + (idy)*srcWidth + (idx * pixelSize);
 
   
    /* simple copy */
    *out = *in;
    out++; in++;

    *out = *in;
    out++; in++;

    *out = *in;
    out++; in++;
 
     *out = *in;

   }
}


