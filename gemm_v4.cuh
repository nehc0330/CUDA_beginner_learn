// v3 是跨越访问 v4 改进这一点 一次性拿 STRIDE * STRIDE (= 4)个
//------------------ float4_gemm ------------------//
#pragma once
template <unsigned int M_per_BLOCK,//32
          unsigned int K_per_BLOCK,//32
          unsigned int N_per_BLOCK,//32
          unsigned int NUM_per_THREAD> // 每个线程处理的 数据数量 4个 grid 里一共有 block（8 , 32）
__global__ void gemm_v4(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    __shared__ float A_block[M_per_BLOCK][K_per_BLOCK];
    __shared__ float B_block[K_per_BLOCK][N_per_BLOCK];
    int tx = threadIdx.x; // 每个 tx 处理原来 tx * NUM_per_THREAD 开始的连续四个值
    int ty = threadIdx.y;

    // 本线程要计算的 d_C 的值 (* 4)
    int row = blockIdx.y * M_per_BLOCK + ty;
    int col = blockIdx.x * N_per_BLOCK + tx * NUM_per_THREAD;

    float sum[NUM_per_THREAD] = {0.0f};

    for (int k = 0; k < K; k += K_per_BLOCK)
    {
        if (row < M && k + tx * NUM_per_THREAD + NUM_per_THREAD - 1< K)
            FETCH_FLOAT4(A_block[ty][tx * NUM_per_THREAD]) = FETCH_FLOAT4(d_A[OFFSET(row, k + tx * NUM_per_THREAD, K)]);
        if (k + ty < K && col + NUM_per_THREAD - 1 < N)
            FETCH_FLOAT4(B_block[ty][tx * NUM_per_THREAD]) = FETCH_FLOAT4(d_B[OFFSET(k + ty, col, N)]);
        __syncthreads();

        for (int i = 0; i < NUM_per_THREAD; ++i)
        {
            for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
            {
                    sum[i] += A_block[ty][inner_k] * B_block[inner_k][tx * NUM_per_THREAD + i];
            }
        }
        // 同步 下一步循环要清空SMem 必须要把数据用完
        __syncthreads();
    }

    for (int i = 0; i < NUM_per_THREAD; ++i)
    {
        if (row < M && col + i < N)
            d_C[OFFSET(row, col + i, N)] = sum[i];
    }
}