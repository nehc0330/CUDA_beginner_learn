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
          unsigned int Y_per_THREAD> // ÿ���̴߳���� �������� 4��, grid ��һ�� 32 * 8���� block��8,32��
__global__ void
gemm_v6(int M, int K, int N, float *d_A, float *d_B, float *d_C)
{
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK]; // 2 * 32 * 32
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK]; // 2 * 32 * 32
    int tx = threadIdx.x;                                  // ÿ�� tx ����ԭ�� tx * NUM_per_THREAD ��ʼ�������ĸ�ֵ
    int ty = threadIdx.y;

    // ���߳�Ҫ����� d_C ��ֵ (Y_per_THREAD * X_per_THREAD  * 2)
    int row = blockIdx.y * M_per_BLOCK * 2 + ty * Y_per_THREAD;
    int col = blockIdx.x * N_per_BLOCK * 2 + tx * X_per_THREAD;


    // ���� sum ��ʱ��Ҳ���� FETCH_FLOAT4() ����
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

        // ���üĴ���
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

        // ���üĴ���
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

    // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������

    for (int i = 0; i < M_per_BLOCK; ++i)
    {
        for (int j = 0; j < N_per_BLOCK; ++j)
        {
            d_C[OFFSET(row + i, col + j, N)] = sum[0][i][j]; 
            d_C[OFFSET(row + i + M_per_BLOCK, col + j, N)] = sum[1][i][j]; 
        }
    }
}
