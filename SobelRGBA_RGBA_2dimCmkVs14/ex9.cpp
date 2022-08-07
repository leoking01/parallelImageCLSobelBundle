#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"

#include "CL/cl.hpp"

#include <chrono>

using namespace cv;

void PrintPlatformMsg(
        cl_platform_id *platform
        , cl_platform_info platform_info
        ,  const char *platform_msg
        )
{
    size_t size;
    int err_num;
    // 1. 第一步通过size获取打印字符串长度
    err_num = clGetPlatformInfo(*platform, platform_info, 0, NULL, &size);
    char *result_string = (char *)malloc(size);
    // 2. 第二步获取平台信息到result_string
    err_num = clGetPlatformInfo(*platform, platform_info, size, result_string, NULL);
    printf("%s = %s\n", platform_msg, result_string);
    free(result_string);
    result_string = NULL;
}


void
info()
{
    //    cl_platform_id  platform_list[2]={0 };
    cl_int err_num;
    cl_uint num_platform;
    cl_platform_id *platform_list;
    // 1. 第一次调用获取平台数
    err_num = clGetPlatformIDs(0, NULL, &num_platform);
    printf("平台数量 num_platform = %d\n", num_platform);
    platform_list = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platform);
    // 2. 第二次调用获取平台对象数组
    err_num  = clGetPlatformIDs(num_platform, platform_list, NULL);
    printf("错误码 err_num = %d\n", err_num);
    // 打印平台信息
    for(int i=0;i<num_platform;i++){
        printf("i = %d\n", i);
        PrintPlatformMsg(&platform_list[i], CL_PLATFORM_PROFILE, "Platform Profile");
        PrintPlatformMsg(&platform_list[i], CL_PLATFORM_VERSION, "Platform Version");
        PrintPlatformMsg(&platform_list[i], CL_PLATFORM_NAME, "Platform Name");// 平台名字
        PrintPlatformMsg(&platform_list[i], CL_PLATFORM_VENDOR, "Platform Vendor");// 平台经销商
    }



    cl_uint num_device;
    cl_device_id device;
    // 1. 获取平台GPU类型OpenCL设备的数量
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    printf("第一种平台上的gpu数量： GPU num_device=%d\n", num_device);
    // 2. 获取一个GPU类型的OpenCL设备
    err_num = clGetDeviceIDs(platform_list[0], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // 对于cl_uint cl_ulong等返回类型参数只需要一步查询
    cl_uint max_compute_units;
    // 获取并打印OpenCL设备的并行计算单元数量
    err_num = clGetDeviceInfo(
                device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                &max_compute_units, NULL);
    printf("最大计算元数量： max_compute_units=%d\n", max_compute_units);

    cl_ulong global_mem_size;
    // 获取并打印OpenCL设备的全局内存大小
    err_num = clGetDeviceInfo(
                device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                &global_mem_size, NULL);
    printf("全局内存大小： global_mem_size = %ld \n", global_mem_size);

    size_t *p_max_work_item_sizes=NULL;
    size_t size;
    // CL_DEVICE_MAX_WORK_ITEM_SIZES表示work_group每个维度的最大工作项数目
    // 1. 返回类型是size_t[]，首先查询返回信息的大小
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &size);
    p_max_work_item_sizes = (size_t *)malloc(size);
    // 2. 申请空间后查询结果并打印
    err_num = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, size, p_max_work_item_sizes, NULL);
    for (size_t i = 0; i < size / sizeof(size_t);i++)
    {
        printf("每个维度的最大工作项数目， max_work_item_size_of_work_group_dim %zu=%zu\n", i, p_max_work_item_sizes[i]);
    }


    cl_context   context  ;
    {
        cl_context_properties context_prop[16] = {0};
        context_prop[0] = CL_CONTEXT_PLATFORM;
        context_prop[1] = (cl_context_properties)platform_list[0];

        context = clCreateContext(context_prop, 1, &device, NULL, NULL, &err_num);
        if (err_num != CL_SUCCESS)
        {
            printf("创建上下文，Create Context failed with code=%d!\n", err_num);
        }
        else
        {
            printf("创建上下文， Context successfully created!\n");
        }
    }


    {
        // OpenCL设备命令执行分为入队、提交、启动、结束和完成5个时间点，创建命令队列时使能CL_QUEUE_PROFILING_ENABLE才能获取设备记录的相应时间。
        cl_command_queue_properties queue_prop[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        cl_command_queue command_queue           = NULL;
    }
}


char *snuclLoadFile(const char *filename, size_t *length_ret);

int main()
{
    info();

    Mat src = imread("80modi.jpg",ImreadModes:: IMREAD_COLOR );

    cv::Mat  imgrgba;
    cv::cvtColor( src,imgrgba, cv::COLOR_RGB2RGBA);
    src = imgrgba;

    std::cout<< "src.size() = " <<src.size() <<std::endl;
//      resize(src, src, Size(512, 512));
    std::cout<< "src.type() = " <<src.type() <<std::endl;
    std::cout<< "src.channels() = " <<src.channels() <<std::endl;
    Mat dst_cl(src.size(), src.type());
    Mat dst_cv(src.size(), src.type());
    int  chans =  src.channels();


    // OCL init
    int dy = 0;
    int dx = 1;
    cl_platform_id platform;
    cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
    cl_device_id device;
    cl_context context;
    cl_command_queue cmq;
    cl_program program;
    cl_kernel kernel;
    cl_mem memSrc, memDst; //  主机cl缓存
    cl_int err;

    //    cl_sampler sampler;
    int i, j;

    // OpenCL program
    //// Step 1:  获取平台和上下文
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, dev_type, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cmq = clCreateCommandQueue(context, device, 0, NULL);


    //// Step 2  ： 创建程序，编译程序，建立内核
    size_t program_src_len;
    char *program_src = snuclLoadFile("Device.cl", &program_src_len);
    std::cout<< "程序长度--program_src_len  = " <<program_src_len  <<std::endl;
    program = clCreateProgramWithSource(context, 1, (const char **)&program_src, &program_src_len, NULL);//创建程序
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); //编译程序
    //    cl_program_build_info   proBinfo;
    //    size_t  paraValSz;

    static const size_t LOG_SIZE = 2048;
    char log[LOG_SIZE];
    log[0] = 0;
    cl_int res = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, LOG_SIZE, log, NULL);
    //   cl_int res =  clGetProgramBuildInfo(program,0,proBinfo,paraValSz,0,0);
    std::cout<< "log = " <<log<<std::endl;
    std::cout<< "获取程序构建信息-res = " <<res<<std::endl;
    kernel = clCreateKernel(program, "SobelFilter3x3Image", NULL); //建立内核
    if(program_src)free(program_src);


    //// Step 3: 建立cl图像缓存，input， output
    size_t global[2], local[2];// 规格， 全局，本地
    global[0] = src.cols ; //  src.rows;
    global[1] = src.rows;
    size_t size; //  输入输出图像的缓存的规格
    size = global[0] * global[1] * sizeof(unsigned char)*chans ;
    std::cout<< "图形数据规模 size  = " <<size  <<std::endl;
    local[0] = 32;// 8;//4;//4;//32;
    local[1] = 4;//32;//4;//64;//128;//8;


    // cl图像缓存规格
    size_t image_row_pitch = src.cols* chans  ;
    void * host_ptr = NULL;
    cl_int  errcode_ret ;
    cl_image_format clImageFormat;// cl图像的格式
//    clImageFormat.image_channel_order = CL_A;
     clImageFormat.image_channel_order = CL_RGBA;
    clImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    memSrc = clCreateImage2D(context, CL_MEM_READ_WRITE
                             , &clImageFormat
                             , src.cols // * chans
                             , src.rows
                             , image_row_pitch
                             ,host_ptr,  &errcode_ret );
    std::cout<< "创建原图像-errcode_ret  = " <<errcode_ret  <<std::endl;
    void *  host_ptr2 = NULL;
    cl_int   errcode_ret2;
    memDst = clCreateBuffer(context, CL_MEM_READ_WRITE, size, host_ptr2,  &errcode_ret2 );
    std::cout<< "创建目标buffer---errcode_ret2 = " <<errcode_ret2 <<std::endl;

    //// Step 4： 给内核设置参数
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memSrc);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memDst);
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&src.rows);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&src.cols );
    err = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&dy);
    err = clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&dx);
    err = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&chans);

    // Step 5：
    size_t origin[3] = {0, 0, 0};//cl内存中图像数据的第一个像素的位置
    // std::size_t region[3] = {( std::size_t)src.rows, ( std::size_t)src.cols, 1};// cl内存中图像数据的区域大小
    std::size_t region[3] = {( std::size_t)src.cols , ( std::size_t)src.rows, 1 };
    auto s1 = std::chrono::high_resolution_clock::now();
    // 写buffer，表示向gpu内核内存写数据
    err = clEnqueueWriteImage(cmq, memSrc, CL_FALSE, origin, region, src.cols*chans, 0, src.data, 0, NULL, NULL);
    cl_uint work_dim = 2;
    const size_t * global_work_offset = NULL;
    cl_uint  num_events_in_wait_list = 0 ;
    const cl_event * event_wait_list = NULL;
    cl_event *   event = NULL;
    err = clEnqueueNDRangeKernel(
                cmq, kernel
                , work_dim
                , global_work_offset
                , global
                , local
                , num_events_in_wait_list
                , event_wait_list, event
                );
     std::cout<< "运行结果- err  = " <<err  <<std::endl;
//     #define CL_INVALID_KERNEL       -48
    //  读取buffer，表示从gpu内核内存读取数据
    err = clEnqueueReadBuffer(cmq, memDst, CL_FALSE, 0, size, dst_cl.data, 0, NULL, NULL);
    auto f1 = std::chrono::high_resolution_clock::now();
    double d1= double(std::chrono::duration_cast<std::chrono::microseconds>(f1 -s1).count())/ 1000000;
    printf("OpenCL Processing Time:  %1f msec \n", d1);
    // 将主机内存写到图像对象中   // 将主机内存写到缓存对象中  clEnqueueWriteBuffer
    // 将缓存对象中的数据读到内存对象中 // 将图像对象中的数据读到内存对象中 clEnqueueReadImage

    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(memSrc);
    clReleaseMemObject(memDst);

    // OpenCV
    auto s2 = std::chrono::high_resolution_clock::now();
    //    cv::Sobel(src, dst_cv, -1, xorder, yorder, 3, 1, 0, BorderTypes::BORDER_DEFAULT);
    cv::Sobel(src, dst_cv, -1, dx,dy,  3, 1, 0, BorderTypes::BORDER_DEFAULT);
    auto f2 = std::chrono::high_resolution_clock::now();
    double d2 = double(std::chrono::duration_cast<std::chrono::microseconds>(f2 - s2).count()) / 1000000;
    printf("OpenCV Processing Time:  %1f msec \n", d2);

    imshow("result_OpenCL", dst_cl);
    imshow("result_OpenCV", dst_cv);

    imwrite("result_OpenCL.jpg", dst_cl);
    imwrite("result_OpenCV.jpg", dst_cv);
    std::cout<< "save to :  " <<"result_OpenCL.jpg" <<std::endl;
    std::cout<< "save to :  " <<"result_OpenCV.jpg" <<std::endl;
    waitKey(0);
    return 0;
}

char *snuclLoadFile(const char *filename, size_t *length_ret)
{
    FILE *fp;
    size_t length;
    fp = fopen(filename, "rb");
    if (fp == 0) return NULL;

    fseek(fp, 0, SEEK_END);
    length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    char *ret = (char *)malloc(length + 1);
    if (fread(ret, length, 1, fp) != 1)
    {
        fclose(fp);
        free(ret);
        return NULL;
    }

    fclose(fp);
    if (length_ret)
        *length_ret = length;
    ret[length] = '\0';
    return ret;
}
