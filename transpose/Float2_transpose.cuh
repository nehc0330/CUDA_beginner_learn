#define FETCH_FLOAT2(ptr) (reinterpret_cast<float2 *>(&(ptr))[0])
#define OFFSET(y, x, width) ((y) * (width) + (x))

template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
FLOAT2_transpose(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x * 2;
    int ty = threadIdx.y * 2;

    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    if (col + tx + 1 < N && row + ty + 1 < M)
    {
        float2 tmp1, tmp2;
        tmp1 = FETCH_FLOAT2(input[OFFSET(row + ty + 0, col + tx, N)]);
        tmp2 = FETCH_FLOAT2(input[OFFSET(row + ty + 1, col + tx, N)]);

        FETCH_FLOAT2(output[OFFSET(col + tx + 0, row + ty, M)]) = make_float2(tmp1.x, tmp2.x);
        FETCH_FLOAT2(output[OFFSET(col + tx + 1, row + ty, M)]) = make_float2(tmp1.y, tmp2.y);
    }
}