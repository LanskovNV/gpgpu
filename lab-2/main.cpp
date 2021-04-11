// #include <vld.h>
#include <iostream>
#include "matrixMultiplication.cuh"
#include "matrix.h"

int main()
{
    type_t* a = matrixCreate();
    type_t* b = matrixCreate();
    type_t* cp = matrixCreate();
    type_t* gp = matrixCreate();
 
    std::cout << "Started, matrix size - " << size << ", gpu block size: " << blockSize << std::endl;
    float gpuTime = multiplyWithCuda(a, b, gp);
    std::cout << "GPU elapsed time (in seconds): " << gpuTime << std::endl;
    float cpuTime = multiplyWithCPU(a, b, cp);
    std::cout << "CPU elapsed time (in seconds): " << cpuTime << std::endl;
    std::cout << "Max diff: " << maxDiff(cp, gp) << std::endl;

    matrixRelease(a);
    matrixRelease(b);
    matrixRelease(cp);
    matrixRelease(gp);
    return 0;
}
