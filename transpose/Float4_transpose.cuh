#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define OFFSET(y, x, width) ((y) * (width) + (x))
#include <stdio.h>

template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
FLOAT4_SMem_transpose(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x * 4;
    int ty = threadIdx.y * 4;

    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    __shared__ float tile[BLOCK_SIZE_Y][BLOCK_SIZE_X + 1];

    if (col + tx + 3 < N && row + ty + 3 < M)
    {
        FETCH_FLOAT4(tile[ty + 0][tx]) = FETCH_FLOAT4(input[OFFSET(row + ty + 0, col + tx, N)]);
        FETCH_FLOAT4(tile[ty + 1][tx]) = FETCH_FLOAT4(input[OFFSET(row + ty + 1, col + tx, N)]);
        FETCH_FLOAT4(tile[ty + 2][tx]) = FETCH_FLOAT4(input[OFFSET(row + ty + 2, col + tx, N)]);
        FETCH_FLOAT4(tile[ty + 3][tx]) = FETCH_FLOAT4(input[OFFSET(row + ty + 3, col + tx, N)]);
        __syncthreads();

        float4 tmp1, tmp2, tmp3, tmp4;
        tmp1 = FETCH_FLOAT4(tile[ty + 0][tx]);
        tmp2 = FETCH_FLOAT4(tile[ty + 1][tx]);
        tmp3 = FETCH_FLOAT4(tile[ty + 2][tx]);
        tmp4 = FETCH_FLOAT4(tile[ty + 3][tx]);
        FETCH_FLOAT4(output[OFFSET(col + tx + 0, row + ty, M)]) = make_float4(tmp1.x, tmp2.x, tmp3.x, tmp4.x);
        FETCH_FLOAT4(output[OFFSET(col + tx + 1, row + ty, M)]) = make_float4(tmp1.y, tmp2.y, tmp3.y, tmp4.y);
        FETCH_FLOAT4(output[OFFSET(col + tx + 2, row + ty, M)]) = make_float4(tmp1.z, tmp2.z, tmp3.z, tmp4.z);
        FETCH_FLOAT4(output[OFFSET(col + tx + 3, row + ty, M)]) = make_float4(tmp1.w, tmp2.w, tmp3.w, tmp4.w);
    }
}

template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
FLOAT4_1x4_transpose(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x * 4;
    int ty = threadIdx.y;

    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    if (col + tx + 3 < N && row + ty < M)
    {
        float4 tmp;
        tmp = FETCH_FLOAT4(input[OFFSET(row + ty, col + tx, N)]);

        output[OFFSET(col + tx, row + ty, M)] = tmp.x;
        output[OFFSET(col + tx + 1, row + ty, M)] = tmp.y;
        output[OFFSET(col + tx + 2, row + ty, M)] = tmp.z;
        output[OFFSET(col + tx + 3, row + ty, M)] = tmp.w;
    }
}

template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
FLOAT4_4x4_transpose(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x * 4;
    int ty = threadIdx.y * 4;

    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    if (col + tx + 3 < N && row + ty + 3 < M)
    {
        float4 tmp1, tmp2, tmp3, tmp4;
        tmp1 = FETCH_FLOAT4(input[OFFSET(row + ty + 0, col + tx, N)]);
        tmp2 = FETCH_FLOAT4(input[OFFSET(row + ty + 1, col + tx, N)]);
        tmp3 = FETCH_FLOAT4(input[OFFSET(row + ty + 2, col + tx, N)]);
        tmp4 = FETCH_FLOAT4(input[OFFSET(row + ty + 3, col + tx, N)]);

        FETCH_FLOAT4(output[OFFSET(col + tx, row + ty, M)]) = make_float4(tmp1.x, tmp2.x, tmp3.x, tmp4.x);
        FETCH_FLOAT4(output[OFFSET(col + tx + 1, row + ty, M)]) = make_float4(tmp1.y, tmp2.y, tmp3.y, tmp4.y);
        FETCH_FLOAT4(output[OFFSET(col + tx + 2, row + ty, M)]) = make_float4(tmp1.z, tmp2.z, tmp3.z, tmp4.z);
        FETCH_FLOAT4(output[OFFSET(col + tx + 3, row + ty, M)]) = make_float4(tmp1.w, tmp2.w, tmp3.w, tmp4.w);
    }
}
// 例如，假设每个线程使用以下数量的寄存器：
// 参数：6（两个指针各占两，两个整数各占一）
// 基础变量（bx, by, tx, ty, col, row）：6 → 总12
// row_ty和col_tx：2 → 总14
// tmp1到tmp4：16 → 总30
// 地址计算中的临时变量：假设每个地址计算需要2，四个输入和四个输出地址，但可能部分可以复用 → 比如4输入和4输出各需要2 → 总30+8=38？这显然过高，可能实际情况并非如此。
