/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/
//------------------ RMem_float4_gemm ------------------//
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int M_per_BLOCK,
          unsigned int N_per_BLOCK,
          unsigned int K_per_BLOCK,
          unsigned int NUM_per_THREAD> // 每个线程处理的 数据数量 4个 grid 里一共 32 * 8个块 block（8,32）
__global__ void gemm_v5(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    __shared__ float A_block[M_per_BLOCK][K_per_BLOCK]; // 32 * 32
    __shared__ float B_block[K_per_BLOCK][N_per_BLOCK]; // 32 * 32
    int tx = threadIdx.x;                               // 每个 tx 处理原来 tx * NUM_per_THREAD 开始的连续四个值
    int ty = threadIdx.y;

    // 本线程要计算的 d_C 的值 (* 4 * 4)
    int row = blockIdx.y * M_per_BLOCK + ty * NUM_per_THREAD;
    int col = blockIdx.x * N_per_BLOCK + tx * NUM_per_THREAD;

    // 计算 sum 的时候也运用 FETCH_FLOAT4() 加速
    float sum[NUM_per_THREAD][NUM_per_THREAD] = {0.0f};

    for (int k = 0; k < K; k+=K_per_BLOCK)
    {
        for (int i = 0; i < NUM_per_THREAD; i++)
        {
            FETCH_FLOAT4(A_block[ty * NUM_per_THREAD + i][tx * NUM_per_THREAD]) =
                FETCH_FLOAT4(d_A[OFFSET(row + i, inner_k + tx * NUM_per_THREAD, K)]);
            FETCH_FLOAT4(B_block[ty * NUM_per_THREAD + i][tx * NUM_per_THREAD]) =
                FETCH_FLOAT4(d_B[OFFSET(inner_k + ty + i, col, N)]);
        }

        __syncthreads();

        // 利用寄存器
        for (int inner_k = 0; inner_k < K_per_BLOCK; inner_k++)
        {
            float a_val[4] = {
                A_block[ty * NUM_per_THREAD][inner_k],
                A_block[ty * NUM_per_THREAD + 1][inner_k],
                A_block[ty * NUM_per_THREAD + 2][inner_k],
                A_block[ty * NUM_per_THREAD + 3][inner_k]};
            float b_val[4] = FETCH_FLOAT4(B_block[inner_k][tx * NUM_per_THREAD]);
            for (int i = 0; i < NUM_per_THREAD; i++)
            {
                for (int j = 0; j < NUM_per_THREAD; ++j)
                {
                    sum[i][j] += a_val[i] * b_val[j];
                }
            }
        }

        __syncthreads();
    }

    // 同步 下一步循环要清空SMem 必须要把数据用完

    for (int i = 0; i < NUM_per_THREAD; ++i)
    {
        for (int j = 0; j < NUM_per_THREAD; ++j)
        {
            d_C[OFFSET(row + i, col + j, N)] = sum[i][j]; //?
        }
    }
}
