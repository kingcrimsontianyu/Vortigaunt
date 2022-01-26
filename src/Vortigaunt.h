#if !defined(VORTIGAUNT_H)
#define VORTIGAUNT_H

#include "Detail.h"

namespace Vortigaunt {
#if defined(__CUDACC__)
    //------------------------------------------------------------
    //------------------------------------------------------------
    template <class FindPeerMethod = FindPeersByHandcoding, class ReduceMethod = ReduceByHandcoding>
    __device__ double atomicAddFP64WAG(double* address, double val) {
        unsigned int laneIdx;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneIdx));

        unsigned int activeMask = __activemask();
        unsigned int leaderIdx;
        unsigned int peerMask = FindPeerMethod::Impl(leaderIdx, activeMask, laneIdx, address);
        double tallySum = ReduceMethod::Impl(leaderIdx, activeMask, laneIdx, peerMask, val);

        double old = 0.0;
        if (laneIdx == leaderIdx) {
            old = atomicAdd(address, tallySum);
        }

        return old;
    }
#endif
} // end namespace Vortigaunt

#endif // end include guard