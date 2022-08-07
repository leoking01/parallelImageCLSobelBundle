##cmake_minimum_required(VERSION 3.16)
##project( parallelImageEx9Compare2 LANGUAGES CXX  C )


#if (CMAKE_SYSTEM_NAME MATCHES "Android")
##    set( LLVM_TARGETS_TO_BUILD aarch64)
##    set(CMAKE_BUILD_TYPE Release)
#    #SET ( CMAKE_C_COMPILER "clang" )
#    SET ( CMAKE_CXX_COMPILER "/home/kent/Android/Sdk/ndk/21.1.6352462/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++" )
#    # set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -fopenmp -Ofast -O3 -mfloat-abi=hard -mfpu=neon   -frtti -fexceptions  -lcxxrt  ")
#    # set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -foptimization-record-file=Opt.txt ")
#    # set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -march=native  -std=c++11 -stdlib=libc++  -ldl ")
#    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -std=c++14 -stdlib=libc++  -ldl  -mfpu=neon -ftree-vectorize  ")
##    set(CMAKE_CXX_FLAGS_MINSIZEREL "  -Os -DNDEBUG")
##    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2 -g -DNDEBUG")
##    set(CMAKE_CXX_STANDARD_LIBRARIES " -latomic -lm")
#    set(ANDROID_ARM_NEON TRUE)
#endif()

# set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -O3 -Ofast -std=c++14 -stdlib=libc++  -ldl  -mfpu=neon -ftree-vectorize  ")


#if(0)
#    find_package( OpenCL REQUIRED )
#    find_package(CUDA REQUIRED)
#    include_directories(/usr/local/cuda-11.4/include)
#endif()



#set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
#find_package( OpenCL  REQUIRED )
#message(STATUS "OpenCL found: ${OPENCL_FOUND}")
#message(STATUS "OpenCL includes: ${OPENCL_INCLUDE_DIRS}")
#message(STATUS "OpenCL CXX includes: ${OPENCL_HAS_CPP_BINDINGS}")
#message(STATUS "OpenCL libraries: ${OPENCL_LIBRARIES}")
#include_directories( ${OPENCL_INCLUDE_DIRS} )



#set(OpenCV_DIR  "D:/win10/Cv451GLVs2015install/x64/vc14/lib"  )
#find_package( OpenCV EXACT 4.5.1 REQUIRED PATHS "D:/win10/Cv451GLVs2015install")
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#message(STATUS "OpenCV_INCLUDE_DIRS : ${OpenCV_INCLUDE_DIRS}")
#message(STATUS "OpenCV_INSTALL_PATH : ${OpenCV_INSTALL_PATH}")
#message(STATUS "OpenCV_LIB_COMPONENTS : ${OpenCV_LIB_COMPONENTS}")
#message(STATUS "OpenCV_LIBRARIES : ${OpenCV_LIBRARIES}")



add_executable( SobelSolidSqureCmkVs14OK ex9.cpp Device.cl)
target_compile_features(SobelSolidSqureCmkVs14OK PRIVATE cxx_auto_type)
target_link_libraries(
    SobelSolidSqureCmkVs14OK PRIVATE
    ${CUDA_OpenCL_LIBRARY} ${OpenCV_LIBS} ${OPENCL_LIBRARIES}
    ${OpenCV_LIB_COMPONENTS}
    )

configure_file(Device.cl ${CMAKE_CURRENT_BINARY_DIR}/Device.cl COPYONLY)
#configure_file(../data/image/bd.jpeg ${CMAKE_CURRENT_BINARY_DIR}/bd.jpeg COPYONLY)
#configure_file(../data/image/6.jpeg ${CMAKE_CURRENT_BINARY_DIR}/6.jpeg COPYONLY)


#configure_file(../data/image/Grab_Image.jpg ${CMAKE_CURRENT_BINARY_DIR}/Grab_Image.jpg COPYONLY)
configure_file( ${CMAKE_SOURCE_DIR}/../data/image/80modi.jpg  ${CMAKE_CURRENT_BINARY_DIR}/80modi.jpg  COPYONLY)

