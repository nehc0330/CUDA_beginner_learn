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
