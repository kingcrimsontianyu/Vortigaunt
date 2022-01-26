#include "Utility.h"
#include "Basic.h"
#include <algorithm>
#include <random>
#include <cassert>

//------------------------------------------------------------
//------------------------------------------------------------
TestManager::TestManager(const std::vector<int>& idxVecGenerated) :
    idxVec(idxVecGenerated) {

    // idxVec valid values are -1 (inactive thread), 0, 1, 2...
    // uniqueVec contains unique, sorted indexes obtained from idxVec

    uniqueVec = idxVec; // deep copy
    std::sort(uniqueVec.begin(), uniqueVec.end());
    auto last = std::unique(uniqueVec.begin(), uniqueVec.end());
    uniqueVec.erase(last, uniqueVec.end());

    for (auto&& item : uniqueVec) {
        int count = std::count(idxVec.begin(), idxVec.end(), item);
        countVec.push_back(count);
    }

    std::vector<int>::iterator resultMax = std::max_element(uniqueVec.begin(), uniqueVec.end());
    std::vector<int>::iterator resultMin = std::min_element(uniqueVec.begin(), uniqueVec.end());
    int adjustedMin = *resultMin < 0 ? 0 : *resultMin;
    numSubgroups = *resultMax - adjustedMin + 1;
    dose.resize(numSubgroups, 0.0);
}

//------------------------------------------------------------
//------------------------------------------------------------
void TestManager::Run() {
    assert(Vortigaunt::IsAnticipatedWarpSizeCorrect());

    HANDLE_GPU_ERROR(cudaSetDevice(0));

    HANDLE_GPU_ERROR(cudaMalloc(&dose_d, dose.size() * sizeof(double)));
    HANDLE_GPU_ERROR(cudaMalloc(&idxVec_d, idxVec.size() * sizeof(int)));

    HANDLE_GPU_ERROR(cudaMemcpy(dose_d, dose.data(), dose.size() * sizeof(double), cudaMemcpyDefault));
    HANDLE_GPU_ERROR(cudaMemcpy(idxVec_d, idxVec.data(), idxVec.size() * sizeof(int), cudaMemcpyDefault));
    HANDLE_GPU_ERROR(cudaMemset(dose_d, 0, dose.size() * sizeof(double)));

    RunKernel();

    HANDLE_GPU_ERROR(cudaMemcpy(dose.data(), dose_d, dose.size() * sizeof(double), cudaMemcpyDefault));
    HANDLE_GPU_ERROR(cudaFree(dose_d));
    HANDLE_GPU_ERROR(cudaFree(idxVec_d));

    std::cout << "--> Result\n";
    std::cout << "    " << std::setw(10) << "index"
        << std::setw(10) << "count"
        << std::setw(10) << "dose\n";

    for (std::size_t i = 0; i < uniqueVec.size(); ++i) {
        if (uniqueVec[i] >= 0) {
            std::cout << "    " << std::setw(10) << uniqueVec[i]
                << std::setw(10) << countVec[i]
                << std::setw(10) << dose[uniqueVec[i]] << "\n";
        }
    }
}

//------------------------------------------------------------
//------------------------------------------------------------
TestManager::~TestManager() {
    HANDLE_GPU_ERROR(cudaDeviceReset());
}

//------------------------------------------------------------
//------------------------------------------------------------
int main(int, char**) {

    {
        // test 1: randomly generate indices
        //
        // index and count
        // -1 means the thread should be set to inactive
        // 0, 1, 2 ... are valid index for the dose array
        // index:          -1    0  1  2  3
        // count:           8    8  6  4  6
        // expected dose:  n/a  80 60 40 60

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(-1, 3); // randomly sample integers from [-1, 3]

        std::vector<int> idxVec(warpSizeGPU);
        for (auto&& item : idxVec) {
            item = dist(rng);
        }

        TestManager tm(idxVec);
        tm.Run();
    }

    {
        // test 2: manually specify indexes

        std::vector<int> idxVec{ 0,  -1,  2,  3,  4,  5,  6,  7,
                                 8,  9, 10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23,
                                24, 25, 26, 27, 28, 29, -1, 31 };
        TestManager tm(idxVec);
        tm.Run();
    }

    {
        // test 3: manually specify indexes

        std::vector<int> idxVec{ -1, 4, -1, 4, -1, 4, -1, 4,
                                 -1, 4, -1, 4, -1, 4, -1, 4,
                                 -1, 4, -1, 4, -1, 4, -1, 4,
                                 -1, 4, -1, 4, -1, 4, -1, 4 };
        TestManager tm(idxVec);
        tm.Run();
    }

    return 0;

}
