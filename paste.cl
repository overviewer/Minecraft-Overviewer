const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; 

__kernel void paste_gpu (__write_only image2d_t imDest,        /* pointer to destination image data */
                         __read_only image2d_t imSrc,          /* pointer to source image data */
                         __private uchar quadrant              /* what quadrant are we working in?   0  1
                                                                                                     2  3  */
)
{

    /* get_global_id(0) returns the ID of the thread in execution.
    As many threads are launched at the same time, executing the same kernel,
    each one will receive a different ID, and consequently perform a different computation.*/

    // these will be the x and y in the *source* image.  
    // we need use to 'quandrant' parameter to calculate the pixel in 
    // the destination image

    const int idx = get_global_id(0);
    const int idy = get_global_id(1);

    const int src_x = idx * 2;
    const int src_y = idy * 2;
    
    int dst_x = idx;
    int dst_y = idy;


    if (quadrant == 1 || quadrant == 3)
        dst_x += 192;
    if (quadrant == 2|| quadrant == 3)
        dst_y += 192;

    const int2 dst_coord = {dst_x, dst_y};


    const int2 coord0 = {src_x+0, src_y+0};
    const int2 coord1 = {src_x+0, src_y+1};
    const int2 coord2 = {src_x+1, src_y+0};
    const int2 coord3 = {src_x+1, src_y+1};


    const uint4 pix0 = read_imageui(imSrc, smp, coord0);
    const uint4 pix1 = read_imageui(imSrc, smp, coord1);
    const uint4 pix2 = read_imageui(imSrc, smp, coord2);
    const uint4 pix3 = read_imageui(imSrc, smp, coord3);

    const uint4 pixSum = (pix0 + pix1 + pix2 + pix3);
    const uint4 pixAvg = (pixSum >> 2);

    // average these pixels

    //const uint4 xxx = {128,0,0,255};
    write_imageui(imDest, dst_coord, pixAvg);

}


