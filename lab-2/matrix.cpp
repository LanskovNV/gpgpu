#include <algorithm>
#include <random>
#include <chrono>
#include "matrix.h"

type_t* matrixCreate()
{
    const type_t min = -100;
    const type_t max = 100;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(min, max);

    int n = size * size;
    type_t* matrix = new type_t[n];

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
