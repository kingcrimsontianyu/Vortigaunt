#if !defined(DETAIL_H)
#define DETAIL_H

namespace Vortigaunt {
#if defined(__CUDACC__)
    //------------------------------------------------------------
    //------------------------------------------------------------
    struct FindPeersByHandcoding {
        template <class T>
        static __device__ __forceinline__ unsigned int Impl(unsigned int& leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            T* address) {

            auto activeMaskTemp = activeMask;
            int isPeer;
            unsigned int peerMask;

            do {
                int srcIdx = __ffs(activeMaskTemp) - 1;

                leaderIdx = srcIdx;

                // avoid using the union hack for type punning, for it results in unspecified (not undefined) behavior.
                // instead, use uintptr_t !!!
                uintptr_t addressReinter = reinterpret_cast<uintptr_t>(address);
                uintptr_t addressOther = __shfl_sync(activeMaskTemp, addressReinter, srcIdx);

                isPeer = (addressReinter == addressOther);

                // determine which lanes have a match with the srcIdx-th lane
                // all currently actively lanes receive the same mask result
                peerMask = __ballot_sync(activeMaskTemp, isPeer);

                // remove lanes with the same address with the srcIdx-th lane
                activeMaskTemp ^= peerMask;
            } while (!isPeer); // continue searching if my peer has not been found

            return peerMask;
        }
    };

    //------------------------------------------------------------
    //------------------------------------------------------------
    struct FindPeersByHandcodingUnroll {
        template <class T>
        static __device__ __forceinline__ unsigned int Impl(unsigned int& leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            T* address) {

            auto activeMaskTemp = activeMask;
            unsigned int peerMask;

#pragma unroll
            for (int i = 0; i < nvidiaWarpSize; ++i) {
                int isActive = activeMaskTemp & (1U << i);

                if (!isActive) {
                    continue;
                }

                uintptr_t addressReinter = reinterpret_cast<uintptr_t>(address);
                uintptr_t addressOther = __shfl_sync(activeMaskTemp, addressReinter, i);

                int isPeer = (addressReinter == addressOther);
                int peerMaskTemp = __ballot_sync(activeMaskTemp, isPeer);

                if (isPeer) {
                    peerMask = peerMaskTemp;
                }

                activeMaskTemp ^= peerMaskTemp;
            }
            leaderIdx = __ffs(peerMask) - 1;
            return peerMask;
        }
    };

    //------------------------------------------------------------
    //------------------------------------------------------------
#if __CUDA_ARCH__ >= 700
    struct FindPeersByIntrinsics {
        template <class T>
        static __device__ __forceinline__ unsigned int Impl(unsigned int& leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            T* address) {

            unsigned int peerMask = __match_any_sync(activeMask, reinterpret_cast<uintptr_t>(address));
            leaderIdx = __ffs(peerMask) - 1;
            return peerMask;
        }
    };
#endif
    //------------------------------------------------------------
    //------------------------------------------------------------
    struct ReduceByHandcoding {
        template <class T>
        static __device__ __forceinline__ T Impl(unsigned int leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            unsigned int peerMask,
            T val) {

            int laneMaskLt;
            int laneMaskGt;
            auto peerMaskTemp = peerMask;
            T result = val;

            asm("mov.u32 %0, %%lanemask_lt;" : "=r"(laneMaskLt));
            asm("mov.u32 %0, %%lanemask_gt;" : "=r"(laneMaskGt));

            int relativeIdx = __popc(peerMaskTemp & laneMaskLt);

            while (__popc(peerMaskTemp) > 1) {
                int nextIdx = __ffs(peerMaskTemp & laneMaskGt) - 1;
                int safeNextIdx = nextIdx < 0 ? laneIdx : nextIdx;
                T resultOther = __shfl_sync(peerMaskTemp, result, safeNextIdx);

                int isRelativeOddLane = relativeIdx & 1;

                if (nextIdx >= 0) {
                    result += resultOther;
                }

                unsigned int relativeOddMask = __ballot_sync(peerMaskTemp, isRelativeOddLane);

                peerMaskTemp &= ~relativeOddMask;
                relativeIdx >>= 1;
            }

            return result;
        }
    };

    //------------------------------------------------------------
    //------------------------------------------------------------
    struct ReduceByHandcodingSequential {
        template <class T>
        static __device__ __forceinline__ T Impl(unsigned int leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            unsigned int peerMask,
            T val) {

            auto peerMaskTemp = peerMask;
            T result = val;

            while (__popc(peerMaskTemp) > 1) {
                peerMaskTemp &= (peerMaskTemp - 1);
                int nextIdx = __ffs(peerMaskTemp) - 1;

                T valOther = __shfl_sync(peerMaskTemp, val, nextIdx);

                if (laneIdx == leaderIdx) {
                    result += valOther;
                }
            }

            return result;
        }
    };

    //------------------------------------------------------------
    //------------------------------------------------------------
    struct ReduceByHandcodingSequentialUnroll {
        template <class T>
        static __device__ __forceinline__ T Impl(unsigned int leaderIdx,
            unsigned int activeMask,
            unsigned int laneIdx,
            unsigned int peerMask,
            T val) {

            T result = 0;

#pragma unroll
            for (int i = 0; i < nvidiaWarpSize; ++i) {
                int isActive = peerMask & (1U << i);
                if (isActive) {
                    T valAnother = __shfl_sync(peerMask, val, i);
                    result += valAnother;
                }
            }

            return result;
        }
    };
#endif
} // end namespace Vortigaunt

#endif // end include guard