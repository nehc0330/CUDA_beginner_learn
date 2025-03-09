//------------------ naive_sgemm ------------------//
#pragma once
__global__ void
gemm_v1(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    float tmp = 0.0f;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for (int i = 0; i < K; ++i)
    {
        tmp += d_A[OFFSET(row, i, M)] * d_B[OFFSET(i, col, K)];
    }
    d_C[OFFSET(row, col, N)] = tmp;
}
