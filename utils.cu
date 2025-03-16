#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>

void init_matrix(int row, int col, float *matrix)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            matrix[i * row + j] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

void cpu_gemm(float *A, float *B, float *C, const int m, const int k, const int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int s = 0; s < k; ++s)
                C[i * m + j] += A[i * m + s] * B[s * k + j];
        }
    }
}

void compare_ans(float *h_C_cpu, float *h_C_gpu, int m, int n)
{
    float err = .0f;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            err += (h_C_cpu[i * m + j] - h_C_gpu[i * m + j]) * (h_C_cpu[i * m + j] - h_C_gpu[i * m + j]);
        }
    }
    err = sqrt(err);
    if (err < tol)
    {
        printf("right\n");
    }
    else
    {
        printf("error! err is %.6f\n", err);
    }
}

void GlobalMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    int Block = 32;
    dim3 block(Block, Block);
    dim3 grid((M + Block - 1) / Block, (N + Block - 1) / Block);
    gemm_v1<<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int Block = 32;
    dim3 block(Block, Block);
    dim3 grid((N + Block - 1) / Block, (M + Block - 1) / Block);
    gemm_v2<Block><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void STRIDE_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int Block = 32;
    constexpr int STRIDE = 2;
    dim3 block(Block / STRIDE, Block / STRIDE);
    dim3 grid((N + Block * STRIDE - 1) / (Block * STRIDE), (M + Block * STRIDE - 1) / (Block * STRIDE));
    gemm_v3<Block, STRIDE><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 32;
    constexpr int K_per_BLOCK = 32;
    constexpr int N_per_BLOCK = 32;
    constexpr int NUM_per_THREAD = 4;
    dim3 block(N_per_BLOCK / NUM_per_THREAD, M_per_BLOCK);
    dim3 grid((N + N_per_BLOCK - 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v4<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK, NUM_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 32;
    constexpr int K_per_BLOCK = 32;
    constexpr int N_per_BLOCK = 32;
    constexpr int NUM_per_THREAD = 4;
    dim3 block(N_per_BLOCK / NUM_per_THREAD, M_per_BLOCK / NUM_per_THREAD);
    dim3 grid((N + N_per_BLOCK - 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v5<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK, NUM_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Transpose_RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 32;
    constexpr int K_per_BLOCK = 32;
    constexpr int N_per_BLOCK = 32;
    constexpr int Y_per_THREAD = 4;
    constexpr int X_per_THREAD = 4;
    dim3 block(N_per_BLOCK / X_per_THREAD, M_per_BLOCK / Y_per_THREAD);
    dim3 grid((N + N_per_BLOCK - 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v6<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK, Y_per_THREAD, X_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Buffer_Transpose_RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 128;
    constexpr int K_per_BLOCK = 8;
    constexpr int N_per_BLOCK = 128;
    constexpr int Y_per_THREAD = 8;
    constexpr int X_per_THREAD = 8;
    dim3 block(N_per_BLOCK / X_per_THREAD, M_per_BLOCK / Y_per_THREAD);
    dim3 grid((N + N_per_BLOCK - 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v7<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK, Y_per_THREAD, X_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Double_Buffer_RMem_SMem(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 128;
    constexpr int K_per_BLOCK = 8;
    constexpr int N_per_BLOCK = 128;
    constexpr int Y_per_THREAD = 8;
    constexpr int X_per_THREAD = 8;
    dim3 block(N_per_BLOCK / X_per_THREAD, M_per_BLOCK / Y_per_THREAD); // 16 * 16
    dim3 grid((N + N_per_BLOCK - 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v8<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK, Y_per_THREAD, X_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}