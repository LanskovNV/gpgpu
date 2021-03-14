#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

typedef double type_t;

const size_t size = 2000;
const size_t blockSize = 32;

type_t* matrixCreate();
type_t maxDiff(type_t* a, type_t* b);
float multiplyWithCPU(type_t* a, type_t* b, type_t* c);
float multiplyWithCuda(type_t* a, type_t* b, type_t* c);

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

    delete[] a;
    delete[] b;
    delete[] cp;
    delete[] gp;
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
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << stderr << "\ncudaSetDevice failed!\n Do you have a CUDA-capable GPU installed?\n";
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_a, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, byteSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_b, b, byteSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    multiplyKernel<<<grid, block>>>(dev_a, dev_b, dev_c);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, byteSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    return gpuTime / 1000.0f;
}

/*
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int oldM()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }



    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/