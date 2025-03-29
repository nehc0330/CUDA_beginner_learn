#include "kernels.cuh"
#include "utlis.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i * cols + j] = (float)(i * cols + j); // 填充一些测试数据
        }
    }
}

void naive_16_16(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE = 16;
    dim3 grid_dim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_8_32(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 8;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_32_8(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 8;
    const int BLOCK_SIZE_Y = 32;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_64_4(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 4;
    const int BLOCK_SIZE_Y = 64;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_4_64(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 64;
    const int BLOCK_SIZE_Y = 4;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_2_128(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 128;
    const int BLOCK_SIZE_Y = 2;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}
void naive_128_2(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 2;
    const int BLOCK_SIZE_Y = 128;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void naive_1_256(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 256;
    const int BLOCK_SIZE_Y = 1;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}
void naive_256_1(const int M, const int N, float *input, float *output)
{
    const int BLOCK_SIZE_X = 1;
    const int BLOCK_SIZE_Y = 256;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_naive<<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void SMem_16_16(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 16;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_SMem<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void SMem_16_16_pad(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16;
    constexpr int BLOCK_SIZE_Y = 16;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    transpose_SMem_pad<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float2_2x2_32x8(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 8 * 2;
    constexpr int BLOCK_SIZE_Y = 32 * 2;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2);
    FLOAT2_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float2_2x2_8x32(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 32 * 2;
    constexpr int BLOCK_SIZE_Y = 8 * 2;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2);
    FLOAT2_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float2_2x2_16x16(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16 * 2;
    constexpr int BLOCK_SIZE_Y = 16 * 2;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2);
    FLOAT2_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_1x4_32x8(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 8 * 4;
    constexpr int BLOCK_SIZE_Y = 32;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y);
    FLOAT4_1x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_1x4_8x32(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 32 * 4;
    constexpr int BLOCK_SIZE_Y = 8;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y);
    FLOAT4_1x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_1x4_16x16(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16 * 4;
    constexpr int BLOCK_SIZE_Y = 16;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y);
    FLOAT4_1x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_4x4_32x8(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 8 * 4;
    constexpr int BLOCK_SIZE_Y = 32 * 4;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y / 4);
    FLOAT4_4x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_4x4_8x32(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 32 * 4;
    constexpr int BLOCK_SIZE_Y = 8 * 4;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y / 4);
    FLOAT4_4x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_4x4_16x16(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16 * 4;
    constexpr int BLOCK_SIZE_Y = 16 * 4;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y / 4);
    FLOAT4_4x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}

void Float4_4x4_16x16_resize_SMem(const int M, const int N, float *input, float *output)
{
    constexpr int BLOCK_SIZE_X = 16 * 4;
    constexpr int BLOCK_SIZE_Y = 16 * 4;
    dim3 grid_dim((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 block_dim(BLOCK_SIZE_X / 4, BLOCK_SIZE_Y / 4);
    int sharedMemSize = 1024;
    FLOAT4_4x4_transpose<BLOCK_SIZE_Y, BLOCK_SIZE_X><<<grid_dim, block_dim, sharedMemSize>>>(M, N, input, output);
    checkCudaErrors(cudaDeviceSynchronize());
}