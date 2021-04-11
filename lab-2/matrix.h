#pragma once

typedef double type_t;

const size_t size = 2000;
const size_t blockSize = 32;

type_t* matrixCreate();
void matrixRelease(type_t* matrix);
type_t maxDiff(type_t* a, type_t* b);
float multiplyWithCPU(type_t* a, type_t* b, type_t* c);
