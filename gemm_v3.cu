/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

//------------------ 2 * 2 block_gemm ------------------//
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void
gemm_v3(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    // 在 SMem 中存储 d_A 和 d_B 的块 读取 STRIDE*STRIDE 个数据
    __shared__ float A_block[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];
    __shared__ float B_block[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];

    // 找到这个线程计算的第一个块的坐标，这个坐标还要计算其他 STRIDE*STRIDE - 1 个块的数据
    int row = blockIdx.y * BLOCK_SIZE * STRIDE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE * STRIDE + threadIdx.x;

    // 在第一个分块矩阵中的坐标 其他 STRIDE*STRIDE - 1 个块在这个基础上加
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum[STRIDE][STRIDE] = {0.0f};
    for (int k = 0; k < K; k += STRIDE * BLOCK_SIZE)
    {
        // 把数据中从 GMem 中记录到 SMem 中
        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                if (row + i * BLOCK_SIZE < M && k + tx + j * BLOCK_SIZE < K)
                    A_block[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = d_A[OFFSET(row + i * BLOCK_SIZE, k + tx + j * BLOCK_SIZE, K)];

                if (k + ty + i * BLOCK_SIZE < K && col + j * BLOCK_SIZE < N)
                    B_block[ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = d_B[OFFSET(k + ty + i * BLOCK_SIZE, col + j * BLOCK_SIZE, N)];
            }
        }

        // 同步 下一步要用共享内存的数据
        __syncthreads();
#pragma unroll
        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                for (int inner_k = 0; inner_k < BLOCK_SIZE * STRIDE; ++inner_k)
                {
                    if (row + i * BLOCK_SIZE < M && col + j * BLOCK_SIZE < N)
                        sum[i][j] += A_block[ty * STRIDE + i][inner_k] * B_block[inner_k][j + tx * STRIDE];
                }
            }
        }
        // 同步 下一步循环要清空SMem 必须要把数据用完
        __syncthreads();
    }

    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            d_C[OFFSET(row + i * BLOCK_SIZE, col + j * BLOCK_SIZE, N)] = sum[i][j];
        }
    }
}