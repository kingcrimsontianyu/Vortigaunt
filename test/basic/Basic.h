#if !defined(BASIC_H)
#define BASIC_H

#include <vector>

constexpr int warpSizeGPU = 32;

//------------------------------------------------------------
//------------------------------------------------------------
class TestManager
{
public:
    void Run();
    TestManager(const std::vector<int>& idxVecGenerated);
    ~TestManager();

protected:
    void RunKernel();

    std::vector<double> dose;
    double* dose_d;

    std::vector<int> idxVec;
    int* idxVec_d;
    int numSubgroups;

    std::vector<int> uniqueVec;
    std::vector<int> countVec;
};

#endif // end include guard
