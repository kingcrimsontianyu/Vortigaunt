#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <iomanip>

#if !defined(UTILITY_H)
#define UTILITY_H


// We want to use a compile-time constant warp size.
// According to PTX reference manual, WARP_SZ is a runtime immediate constant.
// The runtime API's device property warpSize is a runtime value as well.
// Therefore we had better not use either of them.
// The most straightforward way is to just define the compile-time constant
// for use in the kernel, and assert that it equal the runtime warp size outside kernel.
// Reference: https://stackoverflow.com/questions/36047035/when-should-i-use-cudas-built-in-warpsize-as-opposed-to-my-own-proper-constant
// See the comment by Robert Crovella.
constexpr int nvidiaWarpSize = 32;

//------------------------------------------------------------
//------------------------------------------------------------
#ifndef HANDLE_GPU_ERROR
#define HANDLE_GPU_ERROR(err) \
    do \
    { \
        if(err != cudaSuccess) \
        { \
            int currentDevice; \
            cudaGetDevice(&currentDevice); \
            std::cerr << "CUDA device assert: device = " \
                      << currentDevice << ", " << cudaGetErrorString(err) \
                      << " in " << __FILE__ \
                      << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } \
    while(0)
#endif

//------------------------------------------------------------
//------------------------------------------------------------
namespace Vortigaunt {
    //------------------------------------------------------------
    //------------------------------------------------------------
    inline void QueryDeviceInfo() {
        int GPUCount = 0;
        cudaDeviceProp GPUProperty;
        HANDLE_GPU_ERROR(cudaGetDeviceCount(&GPUCount));

        for (int i = 0; i < GPUCount; ++i) {
            HANDLE_GPU_ERROR(cudaSetDevice(i));
            HANDLE_GPU_ERROR(cudaGetDeviceProperties(&GPUProperty, i));

            std::cout << "--> " << GPUProperty.name << "\n"
                << "    compute capability = " << GPUProperty.major << "." << GPUProperty.minor << "\n"
                << "    global memory [GB] = " << GPUProperty.totalGlobalMem / 1024.0 / 1024.0 / 1024.0 << "\n"
                << "    constant memory [KB] = " << GPUProperty.totalConstMem / 1024.0 << "\n"
                << "    shared memory per MP [KB] = " << GPUProperty.sharedMemPerMultiprocessor / 1024.0 << "\n"
                << "    MP count = " << GPUProperty.multiProcessorCount << "\n"
                << "    L2 cache [MB] = " << GPUProperty.l2CacheSize / 1024.0 / 1024.0 << "\n";

        }
    }

    //------------------------------------------------------------
    //------------------------------------------------------------
    inline bool IsAnticipatedWarpSizeCorrect() {
        bool result = true;
        int GPUCount = 0;
        cudaDeviceProp GPUProperty;
        HANDLE_GPU_ERROR(cudaGetDeviceCount(&GPUCount));

        for (int i = 0; i < GPUCount; ++i) {
            HANDLE_GPU_ERROR(cudaSetDevice(i));
            HANDLE_GPU_ERROR(cudaGetDeviceProperties(&GPUProperty, i));

            if (GPUProperty.warpSize != nvidiaWarpSize) {
                result = false;
            }
        }

        return result;
    }
} // end namespace Vortigaunt

#endif // end include guard