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

    //    float xoutColor = 0.0f;
    //    float youtColor = 0.0f;

    float4 xoutColor = {0.0f,0.0f,0.0f,0.0f};
    float4 youtColor = {0.0f,0.0f,0.0f,0.0f};

    if (i   > 1 &  i  < cols  - 2) {
        if (j > 1 &  j < rows - 2) {

            //   xoutColor = 0.0f;
            //   youtColor = 0.0f;
            //  xoutColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
            uint4 val4_0 ={0ui,0ui,0ui,0ui};
            uint4 val4_1 ={0ui,0ui,0ui,0ui};
            uint4 val4_2 ={0ui,0ui,0ui,0ui};

            uint4  val4_3 ={0ui,0ui,0ui,0ui};
            uint4 val4 ={0ui,0ui,0ui,0ui};
            uint4 val4_5 ={0ui,0ui,0ui,0ui};

            uint4 val4_6 ={0ui,0ui,0ui,0ui};
            uint4 val4_7 ={0ui,0ui,0ui,0ui};
            uint4 val4_8 ={0ui,0ui,0ui,0ui};

            val4_0= read_imageui(src, sampler, (int2)(i-1, j-1 )  );

            val4_1= read_imageui(src, sampler, (int2)(i-1, j  )  );
            val4_2= read_imageui(src, sampler, (int2)(i-1, j+1 )  );

            val4_3 = read_imageui(src, sampler, (int2)(i, j-1)  );
            val4 = read_imageui(src, sampler, (int2)(i, j)  );
            val4_5 = read_imageui(src, sampler, (int2)(i, j+1)  );

            val4_6= read_imageui(src, sampler, (int2)(i+1, j-1 )  );
            val4_7= read_imageui(src, sampler, (int2)(i+1, j  )  );
            val4_8= read_imageui(src, sampler, (int2)(i+1, j+1 )  );

//            int k=0;
            //  if( 0 )
            //  水平的sobel
            if (dy) {
//                xoutColor =
//                       uint4(  xkernelWeights[0] * (uint4) val4_0 )+
//                        uint4( xkernelWeights[1] *  (uint4)val4_1 ) +
//                       uint4(  xkernelWeights[2] *  (uint4)val4_2 ) +

//                        uint4( xkernelWeights[3] *  (uint4) val4_3 ) +
//                        uint4( xkernelWeights[4] * (uint4) val4  )+
//                        uint4( xkernelWeights[5] *   (uint4)val4_5 ) +

//                       uint4(  xkernelWeights[6] * (uint4)val4_6 ) +
//                        uint4( xkernelWeights[7] * (uint4)val4_7 )  +
//                       uint4(  xkernelWeights[8] * (uint4)val4_8  )  ;

                xoutColor.x =
                       xkernelWeights[0] *  (float)val4_0.x +
                        xkernelWeights[1] *  (float)val4_1.x +
                        xkernelWeights[2] *  (float) val4_2.x +

                        xkernelWeights[3] *   (float)val4_3.x +
                        xkernelWeights[4] *  (float) val4.x +
                        xkernelWeights[5] *  (float) val4_5.x  +

                        xkernelWeights[6] *  (float) val4_6.x  +
                        xkernelWeights[7] *  (float) val4_7.x +
                        xkernelWeights[8] *  (float)val4_8.x ;

                xoutColor.y =
                       xkernelWeights[0] *  (float)val4_0.y +
                        xkernelWeights[1] *  (float)val4_1.y +
                        xkernelWeights[2] *  (float) val4_2.y +

                        xkernelWeights[3] *   (float)val4_3.y +
                        xkernelWeights[4] *  (float) val4.y +
                        xkernelWeights[5] *  (float) val4_5.y  +

                        xkernelWeights[6] *  (float) val4_6.y  +
                        xkernelWeights[7] *  (float) val4_7.y +
                        xkernelWeights[8] *  (float)val4_8.y ;

                xoutColor.z =
                       xkernelWeights[0] *  (float)val4_0.z +
                        xkernelWeights[1] *  (float)val4_1.z +
                        xkernelWeights[2] *  (float) val4_2.z +

                        xkernelWeights[3] *   (float)val4_3.z +
                        xkernelWeights[4] *  (float) val4.z +
                        xkernelWeights[5] *  (float) val4_5.z  +

                        xkernelWeights[6] *  (float) val4_6.z  +
                        xkernelWeights[7] *  (float) val4_7.z +
                        xkernelWeights[8] *  (float)val4_8.z ;

                // xoutColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
            }
#if  1
            //  垂直的sobel
            if (dx) {
                youtColor.x =
                       ykernelWeights[0] *  (float)val4_0.x +
                        ykernelWeights[1] *  (float)val4_1.x +
                        ykernelWeights[2] *  (float) val4_2.x +

                        ykernelWeights[3] *   (float)val4_3.x +
                        ykernelWeights[4] *  (float) val4.x +
                        ykernelWeights[5] *  (float) val4_5.x  +

                        ykernelWeights[6] *  (float) val4_6.x  +
                        ykernelWeights[7] *  (float) val4_7.x +
                        ykernelWeights[8] *  (float)val4_8.x ;

                youtColor.y =
                       ykernelWeights[0] *  (float)val4_0.y +
                        ykernelWeights[1] *  (float)val4_1.y +
                        ykernelWeights[2] *  (float) val4_2.y +

                        ykernelWeights[3] *   (float)val4_3.y +
                        ykernelWeights[4] *  (float) val4.y +
                        ykernelWeights[5] *  (float) val4_5.y  +

                        ykernelWeights[6] *  (float) val4_6.y  +
                        ykernelWeights[7] *  (float) val4_7.y +
                        ykernelWeights[8] *  (float)val4_8.y ;

                youtColor.z =
                       ykernelWeights[0] *  (float)val4_0.z +
                        ykernelWeights[1] *  (float)val4_1.z +
                        ykernelWeights[2] *  (float) val4_2.z +

                        ykernelWeights[3] *   (float)val4_3.z +
                        ykernelWeights[4] *  (float) val4.z +
                        ykernelWeights[5] *  (float) val4_5.z  +

                        ykernelWeights[6] *  (float) val4_6.z  +
                        ykernelWeights[7] *  (float) val4_7.z +
                        ykernelWeights[8] *  (float)val4_8.z ;

                //                printf("val4.x,  val4.y, val4.z, val4.w = %d, %d,%d, %d,", val4.x,  val4.y, val4.z, val4.w);
                //                printf("read_imageui(src, sampler, (int2)(i, j z))  = %fl ",  (float) read_imageui(src, sampler, (int2)(i, j ))   );
                //                     youtColor = (float)read_imageui(src, sampler, (int2)(i, j)  ).w ;
                //                     youtColor = (float)read_imageui(src, sampler,  (int2)(chans*(i)+k, j)  ).w ;
                //  youtColor = (float) read_imageui(src, sampler,  (int2)( i, j)  )  ;
            }

            for(int k=0;k<chans;k++){
                offset =  (j * cols  + i) *chans +k ;
                //                if(k==0) youtColor = val4.x;
                //                if(k==1) youtColor = val4.y;
                //                if(k==2) youtColor = val4.z;
                //                youtColor = 255;
                // printf("%f %f", xoutColor, youtColor);
                //赋值到目标图像
                //                dst[offset] = (unsigned char)(xoutColor + youtColor);
                if(k==0)  dst[offset] =  youtColor.x +  xoutColor.x;
                if(k==1)  dst[offset] =  youtColor.y +  xoutColor.y;
                if(k==2)  dst[offset] =  youtColor.z +  xoutColor.z;
            }
#endif
        }
    } else{
        dst[offset] = 128;
    }
    //     dst[offset] = (unsigned char)(xoutColor + youtColor);
}
