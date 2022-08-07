const sampler_t sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void SobelFilter3x3Image(
        read_only image2d_t src,
        __global unsigned char *dst
        , int rows
        , int cols
        , int dy
        , int dx
        , int chans
        )
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    int offset = 0;//j * cols + i;
    float xkernelWeights[9] = {
        -1.0f, 0.0f,  1.0f
        , -2.0f, 0.0f, 2.0f
        ,  -1.0f, 0.0f, 1.0f};
    float ykernelWeights[9] = {
        -1.0f, -2.0f, -1.0f
        , 0.0f, 0.0f, 0.0f
        ,  1.0f,  2.0f,  1.0f};

    float xoutColor = 0.0f;
    float youtColor = 0.0f;

    if (i   > 1 &  i  < cols  - 2) {
        if (j > 1 &  j < rows - 2) {
            //   xoutColor = 0.0f;
            //   youtColor = 0.0f;
            //  xoutColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
            for(int k=0;k<chans;k++){
                offset =  (j * cols  + i) *chans +k ;
                //  if( 0 )
                //  水平的sobel
                if (dy) {
                    xoutColor =
                            xkernelWeights[0] * (float)read_imageui(src, sampler, (int2)(chans*(i - 1)+k, j - 1)).w +
                            xkernelWeights[1] *  (float)read_imageui(src, sampler, (int2)(chans*(i - 1)+k, j)).w +
                            xkernelWeights[2] * (float)read_imageui(src, sampler, (int2)(chans*(i - 1)+k, j + 1)).w +

                            xkernelWeights[3] *  (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j - 1)).w +
                            xkernelWeights[4] * (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j)).w +
                            xkernelWeights[5] *  (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j + 1)).w +

                            xkernelWeights[6] *(float)read_imageui(src, sampler,(int2)(chans*(i+ 1)+k, j-1)).w+
                            xkernelWeights[7] *(float)read_imageui(src, sampler, (int2)(chans*(i + 1)+k, j)).w +
                            xkernelWeights[8] *(float)read_imageui(src, sampler,(int2)(chans*(i+ 1)+k, j+ 1)).w;
                    // xoutColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
                }
                //  垂直的sobel
                if (dx) {
                    youtColor =
                            ykernelWeights[0] * (float)read_imageui(src, sampler, (int2)(chans*(i- 1)+k,j- 1)).w +
                            ykernelWeights[1] * (float)read_imageui(src, sampler, (int2)(chans*(i - 1)+k, j)).w +
                            ykernelWeights[2] * (float)read_imageui(src, sampler, (int2)(chans*(i -1)+k, j+ 1)).w +

                            ykernelWeights[3] * (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j - 1)).w +
                            ykernelWeights[4] * (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j)).w +
                            ykernelWeights[5] * (float)read_imageui(src, sampler, (int2)(chans*(i)+k, j + 1)).w +

                            ykernelWeights[6] * (float)read_imageui(src, sampler,(int2)(chans*(i + 1)+k, j-1)).w +
                            ykernelWeights[7] * (float)read_imageui(src, sampler, (int2)(chans*(i + 1)+k, j)).w +
                            ykernelWeights[8] * (float)read_imageui(src, sampler,(int2)(chans*(i+ 1)+k, j+ 1)).w;
                    // youtColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
                }
                  dst[offset] = (unsigned char)(xoutColor + youtColor);
            }
        }
        // printf("%f %f", xoutColor, youtColor);
        //赋值到目标图像
//        dst[offset] = (unsigned char)(xoutColor + youtColor);
    } else{
        dst[offset] = 128;
    }
//     dst[offset] = (unsigned char)(xoutColor + youtColor);
}
