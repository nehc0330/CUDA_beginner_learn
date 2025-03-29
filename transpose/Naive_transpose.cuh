#define FETCH_FLOAT4(ptr) (reinterpret_cast<float4 *>(&(ptr))[0])
#define OFFSET(y, x, width) ((y) * (width) + (x))

__global__ void
transpose_naive(const int M, const int N, float *input, float *output)
{
    // ԭ�����ϵ���id����id
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < M)
    {
        output[OFFSET(col, row, M)] = input[OFFSET(row, col, N)];
    }
}