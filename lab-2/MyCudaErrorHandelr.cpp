#include "MyCudaErrorHandler.h"

void MyCudaErrorHandler::MyCudaException::printInfo()
{
    switch (err)
    {
    case CUDA_START:
        std::cout << "\ncudaSetDevice failed!\n Do you have a CUDA-capable GPU installed?\n";
        break;
    case CUDA_MALLOC:
        std::cout << "cudaMalloc failed!";
        break;
    case CUDA_MEMCPY:
        std::cout << "cudaMemcpy failed!";
        break;
    case CUDA_LAUNCH_KERNEL:
        std::cout << "addKernel launch failed: " << cudaGetErrorString(status) << std::endl;
        break;
    case CUDA_DEVICE_SYNCHRONIZE:
        std::cout << "cudaDeviceSynchronize returned error code " << status << " after launching addKernel!\n";
        break;
    default:
        std::cout << "Unsupported error type!!!\n" << cudaGetErrorString(status);
    }
}

void MyCudaErrorHandler::checkCudaStatus(errorCodes errorType, cudaError_t& cudaStatus)
{
    if (cudaStatus != cudaSuccess)
        throw MyCudaException(errorType, cudaStatus);
}