//------------------ 2 * 2 block_gemm ------------------//
#pragma once
template <unsigned int BLOCK_SIZE, unsigned int STRIDE>
__global__ void
gemm_v3(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
 {   // �� SMem �д洢 d_A �� d_B �Ŀ� ��ȡ STRIDE*STRIDE ������
    __shared__ float A_block[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];
    __shared__ float B_block[BLOCK_SIZE * STRIDE][BLOCK_SIZE * STRIDE];

    int tx = threadIdx.x, ty = threadIdx.y;
    // �ҵ�����̼߳���ĵ�һ��������꣬������껹Ҫ�������� STRIDE*STRIDE - 1 ���������
    int row = blockIdx.y * BLOCK_SIZE * STRIDE + ty;
    int col = blockIdx.x * BLOCK_SIZE * STRIDE + tx;

    // �ڵ�һ���ֿ�����е����� ���� STRIDE*STRIDE - 1 ��������������ϼ�
    float sum[STRIDE][STRIDE] = {0.0f};
    for (int k = 0; k < K; k += STRIDE * BLOCK_SIZE)
    {
        // �������д� GMem �м�¼�� SMem ��
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

        // ͬ�� ��һ��Ҫ�ù����ڴ������
        __syncthreads();

        for (int i = 0; i < STRIDE; ++i)
        {
            for (int j = 0; j < STRIDE; ++j)
            {
                for (int inner_k = 0; inner_k < BLOCK_SIZE * STRIDE; ++inner_k)
                {
                    if (k + inner_k < K)
                    { // ȷ����Խ��
                        sum[i][j] += A_block[ty + BLOCK_SIZE * i][inner_k] * B_block[inner_k][tx + BLOCK_SIZE * j];
                    }
                }
            }
        }
        // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������
        __syncthreads();
    }

    // д����ʱ���߽�
    for (int i = 0; i < STRIDE; ++i)
    {
        for (int j = 0; j < STRIDE; ++j)
        {
            int write_row = row + i * BLOCK_SIZE;
            int write_col = col + j * BLOCK_SIZE;
            if (write_row < M && write_col < N)
            {
                d_C[OFFSET(write_row, write_col, N)] = sum[i][j];
            }
        }
    }
}