//-- -- -- -- -- -- -- -- --Double_Buffer_SMem &RMem_gemm-- -- -- -- -- -- -- -- -- //

template <unsigned int M_per_BLOCK,  // 128
          unsigned int K_per_BLOCK,  // 8
          unsigned int N_per_BLOCK,  // 128
          unsigned int Y_per_THREAD, // width of block of C that each thread calculate
          unsigned int X_per_THREAD> // height of block of C that each thread calculate

__global__ void
gemm_v8(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int X_Thread_Num_per_Block = N_per_BLOCK / X_per_THREAD;
    // const int Y_Thread_Num_per_Block = M_per_BLOCK / Y_per_THREAD;

    // per thread calc
    float sum[Y_per_THREAD][X_per_THREAD] = {0.0f};
    float a_reg[2][Y_per_THREAD] = {0.0f};
    float b_reg[2][X_per_THREAD] = {0.0f};

    // SMem
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK];
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK];

    // d_C ptr
    int row = by * M_per_BLOCK;
    int col = bx * N_per_BLOCK;

    // tid in thread block
    const int tid = ty * X_Thread_Num_per_Block + tx;

    const int A_tile_per_row_thread = K_per_BLOCK / 4; // 2
    const int B_tile_per_row_thread = N_per_BLOCK / 4; // 64

    // matrix block : tile(ty,tx)
    int A_tile_ty = tid / A_tile_per_row_thread;
    int A_tile_tx = tid % A_tile_per_row_thread;

    int B_tile_ty = tid / B_tile_per_row_thread;
    int B_tile_tx = tid % B_tile_per_row_thread;

    // tmp for loading GMem2SMem
    float a_tmp[4];
    // float b_tmp[4];

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    // warp_id * 8 + (lane_id / 16)*4; // (warp_id/4)*32 + ((lane_id%16)/2)*4;
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    // 在进入循环之前 先从 GMem 预取出一部分到 SMem 中 也就是原先的循环的第一部分的操作
    // 之后进入循环 从 GMem 中取出数据存入 SMem 的剩下部分，并计算第一部分的sum 再同步 也就是把原先的两次同步变味一次同步
    int write = 0;
    int read = 0;

    // GMem2SMem
    FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, A_tile_tx * 4, K)]);
    A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
    A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
    A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
    A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

    FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
        FETCH_FLOAT4(d_B[OFFSET(B_tile_ty, col + B_tile_tx * 4, N)]);

    __syncthreads();
    write ^= 1;

    //  SMem2RMem
    FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
    FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
    FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
    FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);

    int tile_idx = 0;
    // 进入大循环
    do
    {
        tile_idx += K_per_BLOCK;

        for (int inner_k = 0; inner_k < K_per_BLOCK - 1; ++inner_k)
        {
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index]);
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index + 64]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index + 64]);
            // 计算预取的和
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[i][j] += a_reg[inner_k % 2][i] * b_reg[inner_k % 2][j];
                }
            }
        }
        read ^= 1;
        if (tile_idx < K)
        {
            FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, tile_idx + A_tile_tx * 4, K)]);
            A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
            A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
            A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
            A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

            FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
                FETCH_FLOAT4(d_B[OFFSET(B_tile_ty + tile_idx, col + B_tile_tx * 4, N)]);
            write ^= 1;
        }

        __syncthreads();
        FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
        FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
        FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
        FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);
        for (int i = 0; i < Y_per_THREAD; ++i)
        {
            for (int j = 0; j < X_per_THREAD; ++j)
            {
                sum[i][j] += a_reg[1][i] * b_reg[1][j];
            }
        }

    } while (tile_idx < K);

    // 同步 下一步循环要清空SMem 必须要把数据用完

    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i][4]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i + 4][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i + 4][4]);
    }
}
