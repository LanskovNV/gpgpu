#include "matrixMultiplication.cuh"
#include "MyCudaErrorHandler.h"

/*
** Multiplication without shared memory */

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

float multiplyWithCuda(type_t* a, type_t* b, type_t* c)
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
        multiplyKernel << <grid, block >> > (dev_a, dev_b, dev_c);

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
    catch (MyCudaErrorHandler::MyCudaException& e)
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

/*
** Multiplication with shared memory */

__global__ void multiplyKernelShared(type_t* a, type_t* b, type_t* c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    size_t i = blockDim.y * blockIdx.y + threadIdx.y;
    size_t j = blockDim.x * blockIdx.x + threadIdx.x;

    type_t sum = 0;

    for (int subIndex = 0; subIndex * blockSize < size; ++subIndex)
    {
        __shared__ type_t as[blockSize][blockSize];
        __shared__ type_t bs[blockSize][blockSize];

        size_t jSubA = subIndex * blockSize + threadIdx.x;
        size_t iSubB = subIndex * blockSize + threadIdx.y;
        
        if (i < size && jSubA < size)
            as[ty][tx] = a[i * size + jSubA];
        else 
            as[ty][tx] = 0;
        if (j < size && iSubB < size)
            bs[ty][tx] = b[iSubB * size + j];
        else 
            bs[ty][tx] = 0;

        __syncthreads();

        for (int k = 0; k < blockSize; k++)
            sum += as[ty][k] * bs[k][tx];

        __syncthreads();
    }

    if (i < size && j < size)
        c[i * size + j] = sum;
}

float multiplyWithCudaShared(type_t* a, type_t* b, type_t* c)
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
        multiplyKernelShared << <grid, block >> > (dev_a, dev_b, dev_c);

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
    catch (MyCudaErrorHandler::MyCudaException& e)
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
