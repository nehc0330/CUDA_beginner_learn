/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/
// 在v5的基础上对 d_A 存入 SMem 中时进行转置 便于用 FETCH_FLOAT4 读取进入 RMem

//------------------ TransposeA_float4_gemm ------------------//
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int M_per_BLOCK,
          unsigned int N_per_BLOCK,
          unsigned int K_per_BLOCK,  // % 4 = 0
          unsigned int X_per_THREAD, // % 4 = 0
          unsigned int Y_per_THREAD> // % 4 = 0
__global__ void
gemm_v6(int M, int K, int N,
        float *__restrict__ d_A,
        float *__restrict__ d_B,
        float *__restrict__ d_C)
{
    // SMem note: transpose A
    __shared__ float A_block[K_per_BLOCK][M_per_BLOCK];
    __shared__ float B_block[K_per_BLOCK][N_per_BLOCK];

    // 每个 tx 处理原来 tx * X_per_THREAD 开始的连续四个值
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // d_C 地址
    int row = blockIdx.y * M_per_BLOCK;
    int col = blockIdx.x * N_per_BLOCK;

    // 计算 sum 的时候也运用 FETCH_FLOAT4() 加速
    float sum[Y_per_THREAD][X_per_THREAD] = {0.0f};
    float a_reg[Y_per_THREAD] = {0.0f};
    float b_reg[X_per_THREAD] = {0.0f};

    for (int k = 0; k < K; k += K_per_BLOCK)
    {
        for (int i = 0; i < Y_per_THREAD; i++) // sum[Y_per_THREAD][X_per_THREAD]
        {
            float4 tmp = {0.0f};
            for (int j = 0; j < K_per_BLOCK / 4; j++)
            {
                tmp = FETCH_FLOAT4(d_A[OFFSET(row + ty * Y_per_THREAD + i,
                                              k + j * 4, K)]);
                for (int s = 0; s < 4; ++s)
                    A_block[j * 4 + s][ty * Y_per_THREAD + i] = tmp[s];
            }
        }
        for (int i = 0; i < K_per_BLOCK; i++)
        {
            FETCH_FLOAT4(B_block[i][tx * X_per_THREAD]) =
                FETCH_FLOAT4(d_B[OFFSET(k + i, col + tx * X_per_THREAD, N)]);
        }
        __syncthreads();

        // 利用寄存器
        for (int inner_k = 0; inner_k < K_per_BLOCK; inner_k++)
        {
            // SMem2RMem per 4
            for (int i = 0; i < Y_per_THREAD / 4; i++)
                FETCH_FLOAT4(a_reg[4 * i]) = FETCH_FLOAT4(A_block[inner_k][ty * Y_per_THREAD + 4 * i]);
            for (int i = 0; i < X_per_THREAD / 4; i++)
                FETCH_FLOAT4(b_reg[4 * i]) = FETCH_FLOAT4(B_block[inner_k][tx * X_per_THREAD + 4 * i]);
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[i][j] += a_reg[i] * b_reg[j];
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
            d_C[OFFSET(row + i, col + j, N)] = sum[i][j];
        }
    }
}
