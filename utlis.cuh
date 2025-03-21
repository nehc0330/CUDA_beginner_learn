#pragma once

#define tol 1e-2 // machine zero 0.01
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define fetch_float 4
#define warpSize 32
#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

void init_matrix(int row, int col, float *matrix);
void cpu_gemm(float *A, float *B, float *C, const int m, const int k, const int n);
void compare_ans(float *h_C_cpu, float *h_C_gpu, int m, int n);
void GlobalMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void STRIDE_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void Transpose_RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void Buffer_Transpose_RMem_Float4_ShareMemory(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);
void Double_Buffer_RMem_SMem(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C);