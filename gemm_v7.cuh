// grid(16, 16) block(16, 16) A_BLOCK = 128 * K
//------------------ Double_Buffer_SMem_gemm ------------------//
#pragma once
template <unsigned int M_per_BLOCK,  // 128
          unsigned int K_per_BLOCK,  // 8
          unsigned int N_per_BLOCK,  // 128
          unsigned int Y_per_THREAD, // width of block of C that each thread calculate
          unsigned int X_per_THREAD> // height of block of C that each thread calculate
__global__ void
gemm_v7(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // per thread calc
    float sum[Y_per_THREAD][X_per_THREAD] = {0.0f};
    // float a_reg[Y_per_THREAD] = {0.0f};
    // float b_reg[X_per_THREAD] = {0.0f};

    // SMem
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK];
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK];

    // d_C ��ַ
    int row = by * M_per_BLOCK;
    int col = bx * N_per_BLOCK;

    // ÿ��ѭ�������̶߳�Ҫʹ��һ�� tile ������ �Ұ�����һ�� tile ������ 128 * 8

    // per_tile
    const int tid = ty * blockDim.x + tx;

    int A_tile_col = K_per_BLOCK / 4;
    // int A_tile_row = M_per_BLOCK;
    // resize A ty tx
    int A_tile_ty = tid / A_tile_col;
    int A_tile_tx = tid % A_tile_col;

    int B_tile_col = N_per_BLOCK / 4;
    // int B_tile_row = K_per_BLOCK;
    // resize B ty tx
    int B_tile_ty = tid / B_tile_col;
    int B_tile_tx = tid % B_tile_col;

    // �ڽ���ѭ��֮ǰ �ȴ� GMem Ԥȡ��һ���ֵ� SMem �� Ҳ����ԭ�ȵ�ѭ���ĵ�һ���ֵĲ���
    // ֮�����ѭ�� �� GMem ��ȡ�����ݴ��� SMem ��ʣ�²��֣��������һ���ֵ�sum ��ͬ�� Ҳ���ǰ�ԭ�ȵ�����ͬ����ζһ��ͬ��
    int write = 0;
    int read = 0;
    float4 tmp = {0.0f};
    // 256 ���߳� GMem2SMem
    tmp = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, A_tile_tx * 4, K)]);

    A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = tmp.x;
    A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = tmp.y;
    A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = tmp.z;
    A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = tmp.w;

    
    FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
         FETCH_FLOAT4(d_B[OFFSET(B_tile_ty, col + B_tile_tx * 4, N)]);
    __syncthreads();
    write ^= 1;
    // ��ʼѭ��
    #pragma once
    for (int k = K_per_BLOCK; k < K; k += K_per_BLOCK)
    {
        tmp = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, k + A_tile_tx * 4, K)]);

        A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = tmp.x;
        A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = tmp.y;
        A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = tmp.z;
        A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = tmp.w;

        
        FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
            FETCH_FLOAT4(d_B[OFFSET(k + B_tile_ty, col + B_tile_tx * 4, N)]);
        write ^= 1;

        // ���üĴ���
        #pragma once
        for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
        {
            float a_reg[Y_per_THREAD];
            float b_reg[Y_per_THREAD];
            FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(A_block[read][inner_k][ty * Y_per_THREAD]);
            FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(A_block[read][inner_k][ty * Y_per_THREAD + 4]);
            FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(B_block[read][inner_k][tx * X_per_THREAD]);
            FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(B_block[read][inner_k][tx * X_per_THREAD + 4]);
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        __syncthreads();
        read ^= 1;
    }
    #pragma once
    for (int inner_k = 0; inner_k < K_per_BLOCK; ++inner_k)
    {
        float a_reg[Y_per_THREAD];
        float b_reg[X_per_THREAD];
        FETCH_FLOAT4(a_reg[0]) = FETCH_FLOAT4(A_block[read][inner_k][ty * Y_per_THREAD]);
        FETCH_FLOAT4(a_reg[4]) = FETCH_FLOAT4(A_block[read][inner_k][ty * Y_per_THREAD + 4]);
        FETCH_FLOAT4(b_reg[0]) = FETCH_FLOAT4(B_block[read][inner_k][tx * X_per_THREAD]);
        FETCH_FLOAT4(b_reg[4]) = FETCH_FLOAT4(B_block[read][inner_k][tx * X_per_THREAD + 4]);
        #pragma once
        for (int i = 0; i < Y_per_THREAD; ++i)
        {
            #pragma once
            for (int j = 0; j < X_per_THREAD; ++j)
            {
                sum[i][j] += a_reg[i] * b_reg[j];
            }
        }
    }

    // ͬ�� ��һ��ѭ��Ҫ���SMem ����Ҫ����������
    #pragma once
    for (int i = 0; i < Y_per_THREAD; ++i)
    {
        for (int j = 0; j < X_per_THREAD; ++j)
        {
            d_C[OFFSET(row + Y_per_THREAD * ty + i, col + X_per_THREAD * tx + j, N)] = sum[i][j];
        }
    }
}
