#pragma once

#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

void initMatrix(float *matrix, int rows, int cols);
void naive_16_16(const int M, const int N, float *input, float *output);
void naive_32_8(const int M, const int N, float *input, float *output);
void naive_8_32(const int M, const int N, float *input, float *output);
void naive_4_64(const int M, const int N, float *input, float *output);
void naive_64_4(const int M, const int N, float *input, float *output);
void naive_2_128(const int M, const int N, float *input, float *output);
void naive_128_2(const int M, const int N, float *input, float *output);
void naive_1_256(const int M, const int N, float *input, float *output);
void naive_256_1(const int M, const int N, float *input, float *output);

void SMem_16_16(const int M, const int N, float *input, float *output);
void SMem_16_16_pad(const int M, const int N, float *input, float *output);

void Float2_2x2_32x8(const int M, const int N, float *input, float *output);
void Float2_2x2_8x32(const int M, const int N, float *input, float *output);
void Float2_2x2_16x16(const int M, const int N, float *input, float *output);

void Float4_1x4_32x8(const int M, const int N, float *input, float *output);
void Float4_1x4_8x32(const int M, const int N, float *input, float *output);
void Float4_1x4_16x16(const int M, const int N, float *input, float *output);

void Float4_4x4_32x8(const int M, const int N, float *input, float *output);
void Float4_4x4_8x32(const int M, const int N, float *input, float *output);
void Float4_4x4_16x16(const int M, const int N, float *input, float *output);

void Float4_16_16_resize_SMem(const int M, const int N, float *input, float *output);