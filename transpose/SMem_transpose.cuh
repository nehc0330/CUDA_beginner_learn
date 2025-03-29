template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
transpose_SMem(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 原矩阵上点列id和行id
    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    __shared__ float tile[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    if (col + tx < N && row + ty < M)
    {
        tile[ty][tx] = input[OFFSET(row + ty, col + tx, N)];
        __syncthreads();

        output[OFFSET(col + tx, row + ty, M)] = tile[ty][tx];
        // output[OFFSET(row + ty, col + tx, N)] = tile[tx][ty];
    }
}

template <unsigned int BLOCK_SIZE_Y, unsigned int BLOCK_SIZE_X>
__global__ void
transpose_SMem_pad(const int M, const int N, float *input, float *output)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 原矩阵上点列id和行id
    int col = bx * BLOCK_SIZE_X;
    int row = by * BLOCK_SIZE_Y;

    __shared__ float tile[BLOCK_SIZE_Y][BLOCK_SIZE_X + 1];

    if (col + tx < N && row + ty < M)
    {
        tile[ty][tx] = input[OFFSET(row + ty, col + tx, N)];
        __syncthreads();

        // output[OFFSET(col + tx, row + ty, M)] = tile[ty][tx];
        output[OFFSET(row + ty, col + tx, N)] = tile[tx][ty];
    }
}