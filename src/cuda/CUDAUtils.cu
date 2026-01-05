#include "CUDACommon.cuh"

// Implementation of shared kernels
__global__ void initRandomStatesKernel(curandState* states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// This file provides the single definition for shared kernels
// All other .cu files only declare (via the header) but don't define
