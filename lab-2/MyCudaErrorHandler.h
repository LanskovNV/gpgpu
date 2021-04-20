#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <exception>
#include <iostream>

namespace MyCudaErrorHandler
{
    enum errorCodes
    {
        CUDA_START,
        CUDA_MALLOC,
        CUDA_MEMCPY,
        CUDA_LAUNCH_KERNEL,
        CUDA_DEVICE_SYNCHRONIZE,
    };

    class MyCudaException : public std::exception
    {
        cudaError_t status;
        errorCodes err;
    public:
        MyCudaException(errorCodes errorType, cudaError_t& cudaStatus) : err(errorType), status(cudaStatus) {};

        void printInfo();
    };

    void checkCudaStatus(errorCodes errorType, cudaError_t& cudaStatus);
}