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
    int A_ROW = M;
    int B_ROW = K;
    int C_ROW = M;
    float tmp = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < K; i++)
    {
        tmp += d_A[OFFSET(idy, i, A_ROW)] * d_B[OFFSET(i, idx, B_ROW)];
    }
    d_C[OFFSET(idy, idx, C_ROW)] = tmp;
}


//------------------ block_gemm ------------------//
template<unsigned int BLOCK_SIZE>
__global__ void
gemm_v2(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    // �� SMem �д洢 d_A �� d_B �Ŀ� ��ȡ
    __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

    // �ҵ�����̵߳Ľ���Ĵ洢�� d_C ������
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // �ڷֿ�����е�����
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE)
    {
        // �������д� GMem �м�¼�� SMem ��

        A_block[ty][tx] = d_A[OFFSET(row, k + tx, K)]; // row * K + (k + tx)
        B_block[ty][tx] = d_B[OFFSET(k + ty, col, N)]; //(k + ty) * N + col
        // ͬ�� ��һ��Ҫ�ù����ڴ������
        __syncthreads();

        for (int inner_k = 0; inner_k < BLOCK_SIZE; inner_k++)
            sum += A_block[ty][inner_k] * B_block[inner_k][tx];

        // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������
        __syncthreads();
    }

    d_C[OFFSET(row, col, N)] = sum; // row * N + col
}

//------------------ idel_gemm ------------------//
template<unsigned int BLOCK_SIZE>
__global__ void
gemm_v3(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    // �� SMem �д洢 d_A �� d_B �Ŀ� ��ȡ
    __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

    // �ҵ�����̵߳Ľ���Ĵ洢�� d_C ������ 1 2 3 4
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // �ڷֿ�����е�����
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE)
    {
        // �������д� GMem �м�¼�� SMem ��
        A_block[ty][tx] = d_A[OFFSET(row, k + tx, K)]; // row * K + (k + tx)
        A_block[ty][tx+ BLOCK_SIZE / 2] = d_A[OFFSET(row, k + tx+ BLOCK_SIZE / 2, K)]; // row * K + (k + tx)
        A_block[ty+ BLOCK_SIZE / 2][tx] = d_A[OFFSET(row+ BLOCK_SIZE / 2, k + tx, K)]; // row * K + (k + tx)
        A_block[ty+ BLOCK_SIZE / 2][tx+ BLOCK_SIZE / 2] = d_A[OFFSET(row+ BLOCK_SIZE / 2, k + tx+ BLOCK_SIZE / 2, K)]; // row * K + (k + tx)
        B_block[ty][tx] = d_B[OFFSET(k + ty, col, N)]; //(k + ty) * N + col
        B_block[ty][tx+ BLOCK_SIZE / 2] = d_B[OFFSET(row, k + tx+ BLOCK_SIZE / 2, K)]; //(k + ty) * N + col
        B_block[ty+ BLOCK_SIZE / 2][tx] = d_B[OFFSET(row+ BLOCK_SIZE / 2, k + tx, K)]; //(k + ty) * N + col
        B_block[ty+ BLOCK_SIZE / 2][tx+ BLOCK_SIZE / 2] = d_B[OFFSET(row+ BLOCK_SIZE / 2, k + tx+ BLOCK_SIZE / 2, K)]; //(k + ty) * N + col
        // ͬ�� ��һ��Ҫ�ù����ڴ������
        __syncthreads();

        for (int inner_k = 0; inner_k < BLOCK_SIZE / 2; inner_k++)
        {
            sum1 += A_block[ty][inner_k] * B_block[inner_k][tx];
            sum2 += A_block[ty][inner_k] * B_block[inner_k][tx + BLOCK_SIZE / 2];
            sum3 += A_block[ty][inner_k] * B_block[inner_k][tx + BLOCK_SIZE / 2];
            sum4 += A_block[ty + BLOCK_SIZE / 2][inner_k] * B_block[inner_k][tx + BLOCK_SIZE / 2];
        }
        // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������
        __syncthreads();
    }

    d_C[OFFSET(row, col, N)] = sum1; // row * N + col
    d_C[OFFSET(row, col + BLOCK_SIZE / 2 , N)] = sum2;
    d_C[OFFSET(row+ BLOCK_SIZE / 2, col, N)]= sum3;
    d_C[OFFSET(row+ BLOCK_SIZE / 2, col+ BLOCK_SIZE / 2, N)]= sum4;
}