#include "Utility.h"
#include "Basic.h"
#include "Vortigaunt.h"
#include <cuda_runtime.h>

constexpr double doseIncrement = 10.0;

//------------------------------------------------------------
//------------------------------------------------------------
__global__ void Test(double* dose, int* idxVec, int numSubgroups) {
    unsigned int laneIdx;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneIdx));

    int idx = idxVec[laneIdx];

    if (idx != -1) {
        Vortigaunt::atomicAddFP64WAG(&dose[idx], doseIncrement);
#if __CUDA_ARCH__ >= 700
        //Vortigaunt::atomicAddFP64WAG<Vortigaunt::FindPeersByIntrinsics>(&dose[idx], doseIncrement);
#endif
        //Vortigaunt::atomicAddFP64WAG<Vortigaunt::FindPeersByHandcoding, Vortigaunt::ReduceByHandcodingSequential>(&dose[idx], doseIncrement);

        //Vortigaunt::atomicAddFP64WAG<Vortigaunt::FindPeersByHandcodingUnroll, Vortigaunt::ReduceByHandcodingSequentialUnroll>(&dose[idx], doseIncrement);
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::RunKernel() {
    dim3 numBlocks(1, 1, 1);
    dim3 numThreadsPerBlock(32, 1, 1);
    Test << < numBlocks, numThreadsPerBlock >> > (dose_d, idxVec_d, numSubgroups);
    HANDLE_GPU_ERROR(cudaDeviceSynchronize());
}

