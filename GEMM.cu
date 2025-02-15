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

#define d_A(i, j) d_A[i * A_ROW + j] 


#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

//------------------ naive_sgemm ------------------//
__global__ void __launch_bounds__(1024)
    gemm_v1(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    int A_ROW = M;
    int B_ROW = K;
    int C_ROW = M;
    float tmp = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;
    for (int i = 0; i < K; i++)
    {
        tmp += d_A(col, i) * d_B(i, row);
    }
    d_C(col, row) = tmp;
}

#define BLOCK_SIZE 32

//------------------ block_gemm ------------------//
__global__ void __launch_bounds__(1024)
    gemm_v2(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    // 在 SMem 中存储 d_A 和 d_B 的块 读取
    __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    //
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < K; k+= BLOCK_SIZE)
    {
        // 记录A_block 和 B_block 的内容
        if (row < M && (k + tx) < K) {
            A_block[ty][tx] = d_A[row * K + (k + tx)];
        } else {
            A_block[ty][tx] = 0.0f;
        }

        if ((k + ty) < K && col < N) {
            B_block[ty][tx] = d_B[(k + ty) * N + col];
        } else {
            B_block[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += A_block[ty][k] * B_block[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N)
    {
        d_C[row * N + col] += sum;
    }
}