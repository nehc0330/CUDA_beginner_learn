/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

//------------------ Double_Buffer_SMem_gemm ------------------//
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int M_per_BLOCK,
          unsigned int N_per_BLOCK,
          unsigned int K_per_BLOCK,
          unsigned int X_per_THREAD,
          unsigned int Y_per_THREAD> // 每个线程处理的 数据数量 4个, grid 里一共 32 * 8个块 block（8,32）
__global__ void
gemm_v6(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK]; // 2 * 32 * 32
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK]; // 2 * 32 * 32
    int tx = threadIdx.x;                                  // 每个 tx 处理原来 tx * NUM_per_THREAD 开始的连续四个值
    int ty = threadIdx.y;

    // 本线程要计算的 d_C 的值 (Y_per_THREAD * X_per_THREAD  * 2)
    int row = blockIdx.y * M_per_BLOCK * 2 + ty * Y_per_THREAD;
    int col = blockIdx.x * N_per_BLOCK * 2 + tx * X_per_THREAD;


    // 计算 sum 的时候也运用 FETCH_FLOAT4() 加速
    float sum[2][Y_per_THREAD][X_per_THREAD] = {0.0f};

    for (int k = 0; k < K; k += K_per_BLOCK)
    {
        for (int i = 0; i < Y_per_THREAD * 2; ++i)
        {
            float4 tmp = {0.0f};
            tmp = FETCH_FLOAT4(d_A[OFFSET(row + i, k + tx * X_per_THREAD, K)]);
            for (int s = 0; s < 4; ++s)
            {
                A_block[0][k + tx * X_per_THREAD + s][row + i] = tmp[s];
            }
            FETCH_FLOAT4(B_block[0][ty * Y_per_THREAD + i][tx * X_per_THREAD]) =
                FETCH_FLOAT4(d_B[OFFSET(k + ty + i, col, N)]);
        }

        __syncthreads();

        for (int i = 0; i < Y_per_THREAD * 2; ++i)
        {            
            tmp = FETCH_FLOAT4(d_A[OFFSET(row + i + M_per_BLOCK , k + tx * X_per_THREAD, K)]);
            for (int s = 0; s < 4; ++s)
            {
                A_block[1][k + tx * Y_per_THREAD + s][row + i] = tmp[s];
            }
            FETCH_FLOAT4(B_block[1][ty * Y_per_THREAD + i][tx * X_per_THREAD ]) =
                FETCH_FLOAT4(d_B[OFFSET(k + ty + i, col + N_per_BLOCK, N)]);
        }

        // 利用寄存器
        for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
        {
            float a_val[4] = FETCH_FLOAT4(A_block[0][inner_k][ty * Y_per_THREAD]);
            float b_val[4] = FETCH_FLOAT4(B_block[0][inner_k][tx * X_per_THREAD]);
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[0][i][j] += a_val[i] * b_val[j];
                }
            }
        }

        __syncthreads();

        // 利用寄存器
        for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
        {
            float a_val[4] = FETCH_FLOAT4(A_block[1][inner_k][(ty + M_per_BLOCK) * Y_per_THREAD]);
            float b_val[4] = FETCH_FLOAT4(B_block[1][inner_k][tx * X_per_THREAD]);
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[1][i][j] += a_val[i] * b_val[j];
                }
            }
        }
    }

    // 同步 下一步循环要清空SMem 必须要把数据用完

    for (int i = 0; i < M_per_BLOCK; ++i)
    {
        for (int j = 0; j < N_per_BLOCK; ++j)
        {
            d_C[OFFSET(row + i, col + j, N)] = sum[0][i][j]; 
            d_C[OFFSET(row + i + M_per_BLOCK, col + j, N)] = sum[1][i][j]; 
        }
    }
}
