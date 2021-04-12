#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrix.h"

float multiplyWithCuda(type_t* a, type_t* b, type_t* c);
float multiplyWithCudaShared(type_t* a, type_t* b, type_t* c);