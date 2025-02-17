/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))


//------------------ block_gemm ------------------//
template<unsigned int BLOCK_SIZE>
__global__ void
gemm_v2(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    // 在 SMem 中存储 d_A 和 d_B 的块 读取
    __shared__ float A_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_block[BLOCK_SIZE][BLOCK_SIZE];

    // 找到这个线程的结果的存储在 d_C 的坐标
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // 在分块矩阵中的坐标
    int tx = threadIdx.x, ty = threadIdx.y;

    float sum = 0.0f;
    for (int k = 0; k < K; k += BLOCK_SIZE)
    {
        // 把数据中从 GMem 中记录到 SMem 中

        A_block[ty][tx] = d_A[OFFSET(row, k + tx, K)]; // row * K + (k + tx)
        B_block[ty][tx] = d_B[OFFSET(k + ty, col, N)]; //(k + ty) * N + col
        // 同步 下一步要用共享内存的数据
        __syncthreads();

        for (int inner_k = 0; inner_k < BLOCK_SIZE; inner_k++)
            sum += A_block[ty][inner_k] * B_block[inner_k][tx];

        // 同步 下一步循环要清空SMem 必须要把数据用完
        __syncthreads();
    }

    d_C[OFFSET(row, col, N)] = sum; // row * N + col
}
