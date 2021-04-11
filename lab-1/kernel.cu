#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #include <vld.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <exception>
#include <random>

typedef double type_t;

const size_t size = 2000;
const size_t blockSize = 32;

type_t* matrixCreate();
void matrixRelease(type_t *matrix);
type_t maxDiff(type_t* a, type_t* b);
float multiplyWithCPU(type_t* a, type_t* b, type_t* c);
float multiplyWithCuda(type_t* a, type_t* b, type_t* c);

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

        void printInfo()
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
    };

    void checkCudaStatus(errorCodes errorType, cudaError_t& cudaStatus)
    {
        if (cudaStatus != cudaSuccess)
            throw MyCudaException(errorType, cudaStatus);
    }
}



__global__ void multiplyKernel(type_t* a, type_t* b, type_t* c)
{
    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= size || j >= size)
        return;

    size_t index = i * size + j;
    c[index] = 0;

    for (size_t k = 0; k < size; ++k)
    {
        c[index] += a[i * size + k] * b[k * size + j];
    }
}

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

type_t maxDiff(type_t* a, type_t* b)
{
    int n = size * size;
    type_t m = 0;

    for (int i = 0; i < n; ++i)
    {
        m = std::max(m, std::abs(a[i] - b[i]));
    }

    return m;
}

type_t* matrixCreate()
{
    const type_t min = -100;
    const type_t max = 100;

    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min, max);

    int n = size * size;
    type_t *matrix = new type_t[n];

    for (int i = 0; i < n; ++i)
    {
        matrix[i] = distrib(gen);
    }

    return matrix;
}

void matrixRelease(type_t* matrix)
{
    delete[] matrix;
}

float multiplyWithCPU(type_t* a, type_t* b, type_t* c)
{
    auto begin = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            c[row * size + col] = 0;
            for (int k = 0; k < size; ++k)
            {
                c[row * size + col] += a[size * row + k] * b[size * k + col];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    return elapsed.count() / 1000.0f;
}

float multiplyWithCuda(type_t *a, type_t *b, type_t *c) 
{
    type_t* dev_a = 0;
    type_t* dev_b = 0;
    type_t* dev_c = 0;

    int byteSize = size * size * sizeof(type_t);

    unsigned int gridDim = (unsigned int)ceil((double)size / blockSize);
    dim3 block(blockSize, blockSize);
    dim3 grid(gridDim, gridDim);

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    
    float gpuTime = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    try 
    {
        cudaStatus = cudaSetDevice(0);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_START, cudaStatus);

        // Allocate GPU buffers for three vectors (two input, one output)    .
        cudaStatus = cudaMalloc((void**)&dev_c, byteSize);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MALLOC, cudaStatus);
      
        cudaStatus = cudaMalloc((void**)&dev_a, byteSize);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MALLOC, cudaStatus);

        cudaStatus = cudaMalloc((void**)&dev_b, byteSize);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MALLOC, cudaStatus);

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dev_a, a, byteSize, cudaMemcpyHostToDevice);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MEMCPY, cudaStatus);

        cudaStatus = cudaMemcpy(dev_b, b, byteSize, cudaMemcpyHostToDevice);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MEMCPY, cudaStatus);

        // Launch a kernel on the GPU with one thread for each element.
        multiplyKernel <<<grid, block >>> (dev_a, dev_b, dev_c);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_LAUNCH_KERNEL, cudaStatus);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_DEVICE_SYNCHRONIZE, cudaStatus);

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(c, dev_c, byteSize, cudaMemcpyDeviceToHost);
        MyCudaErrorHandler::checkCudaStatus(MyCudaErrorHandler::CUDA_MEMCPY, cudaStatus);
    }
    catch (MyCudaErrorHandler::MyCudaException &e)
    {
        e.printInfo();

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
        cudaEventRecord(stop, 0);

        return -1;
    }
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    return gpuTime / 1000.0f;
}
