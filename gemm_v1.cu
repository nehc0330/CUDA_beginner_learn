/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

#include <cstdio>
#include <cstdlib>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

//------------------ naive_sgemm ------------------//
__global__ void
gemm_v1(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    float tmp = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < K; i++)
    {
        tmp += d_A[OFFSET(row, i, M)] * d_B[OFFSET(i, col, K)];
    }
    d_C[OFFSET(row, col, N)] = tmp;
}



