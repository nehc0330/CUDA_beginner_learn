/*
    A -- [M, K]
    B -- [K, N]
    C -- [M, N] = A * B
*/

// v3 �ǿ�Խ���� v4 �Ľ���һ�� һ������ STRIDE * STRIDE (= 4)��
//------------------ float4_gemm ------------------//
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
template <unsigned int M_per_BLOCK,
          unsigned int N_per_BLOCK,
          unsigned int K_per_BLOCK,
          unsigned int NUM_per_THREAD> // ÿ���̴߳���� �������� 4�� grid ��һ�� 32 * 8���� block��8,32��
__global__ void gemm_v4(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    __shared__ float A_block[M_per_BLOCK][K_per_BLOCK]; // 32 * 32
    __shared__ float B_block[K_per_BLOCK][N_per_BLOCK]; // 32 * 32
    int tx = threadIdx.x;                               // ÿ�� tx ����ԭ�� tx * NUM_per_THREAD ��ʼ�������ĸ�ֵ
    int ty = threadIdx.y;

    // ���߳�Ҫ����� d_C ��ֵ (* 4)
    int row = blockIdx.y * M_per_BLOCK + ty;
    int col = blockIdx.x * N_per_BLOCK + tx * NUM_per_THREAD;

    float sum[NUM_per_THREAD] = {0.0f};

    for (int k = 0; k < K; k += K_per_BLOCK)
    {
        FETCH_FLOAT4(A_block[ty][tx * NUM_per_THREAD]) = FETCH_FLOAT4(d_A[OFFSET(row, k + tx * NUM_per_THREAD, K)]);
        // A_block[ty][tx * NUM_per_THREAD] = d_A[OFFSET(row, k + tx * NUM_per_THREAD, K)];
        // A_block[ty][tx * NUM_per_THREAD + 1] = d_A[OFFSET(row, k + tx * NUM_per_THREAD + 1, K)];
        // A_block[ty][tx * NUM_per_THREAD + 2] = d_A[OFFSET(row, k + tx * NUM_per_THREAD + 2, K)];
        // A_block[ty][tx * NUM_per_THREAD + 3] = d_A[OFFSET(row, k + tx * NUM_per_THREAD + 3, K)];
        FETCH_FLOAT4(B_block[ty][tx * NUM_per_THREAD]) = FETCH_FLOAT4(d_B[OFFSET(k + ty, col, N)]);
        // B_block[ty][tx * NUM_per_THREAD] = d_B[OFFSET(k + idy, col, N)];
        // B_block[ty][tx * NUM_per_THREAD + 1] = d_B[OFFSET(k + idy, col + 1, N)];
        // B_block[ty][tx * NUM_per_THREAD + 2] = d_B[OFFSET(k + idy, col + 2, N)];
        // B_block[ty][tx * NUM_per_THREAD + 3] = d_B[OFFSET(k + idy, col + 3, N)];

        __syncthreads();

        for (int i = 0; i < NUM_per_THREAD; ++i)
        {
            for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
                sum[i] += A_block[ty][inner_k] * B_block[inner_k][tx * NUM_per_THREAD + i];
        }
        // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������
        __syncthreads();
    }
    for (int i = 0; i < NUM_per_THREAD; ++i)
        d_C[OFFSET(row, col + i, N)] = sum[i];
}