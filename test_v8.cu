#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define tol 1e-6 // machine zero 0.01
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define checkCudaErrors(func)                                                      \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    }

void init_matrix(int row, int col, float *matrix)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            // matrix[i * row + j] = 2.0 * (float)drand48() - 1.0;
            matrix[i * row + j] = 2.0;
        }
    }
}

void compare_ans(float *h_C_cpu, float *h_C_gpu, int m, int n)
{
    float err = .0f;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            err += (h_C_cpu[i * m + j] - h_C_gpu[i * m + j]) * (h_C_cpu[i * m + j] - h_C_gpu[i * m + j]);
        }
    }
    err = sqrt(err);
    if (err < tol)
    {
        printf("right\n");
    }
    else
    {
        printf("error! err is %.6f\n", err);
    }
}





template <unsigned int M_per_BLOCK,  // 128
          unsigned int K_per_BLOCK,  // 8
          unsigned int N_per_BLOCK,  // 128
          unsigned int Y_per_THREAD, // width of block of C that each thread calculate
          unsigned int X_per_THREAD> // height of block of C that each thread calculate
__global__ void
gemm_v8(
    int M, int K, int N,
    float *__restrict__ d_A,
    float *__restrict__ d_B,
    float *__restrict__ d_C)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int X_Thread_Num_per_Block = N_per_BLOCK / X_per_THREAD;
    // const int Y_Thread_Num_per_Block = M_per_BLOCK / Y_per_THREAD;

    // per thread calc
    float sum[Y_per_THREAD][X_per_THREAD] = {0.0f};
    float a_reg[2][Y_per_THREAD] = {0.0f};
    float b_reg[2][X_per_THREAD] = {0.0f};

    // SMem
    __shared__ float A_block[2][K_per_BLOCK][M_per_BLOCK];
    __shared__ float B_block[2][K_per_BLOCK][N_per_BLOCK];

    // d_C ptr
    int row = by * M_per_BLOCK;
    int col = bx * N_per_BLOCK;

    //-------------------------------------    
    // tid in thread block
    const int tid = ty * X_Thread_Num_per_Block + tx;
    // part 1 
    const int A_tile_per_row_thread = K_per_BLOCK / 4; // 2
    const int B_tile_per_row_thread = N_per_BLOCK / 4; // 64

    // matrix block : tile(ty,tx)
    int A_tile_ty = tid / A_tile_per_row_thread;
    int A_tile_tx = tid % A_tile_per_row_thread;

    int B_tile_ty = tid / B_tile_per_row_thread;
    int B_tile_tx = tid % B_tile_per_row_thread;

    // tmp for loading GMem2SMem
    float a_tmp[4];
    // float b_tmp[4];

    // part 2
    // reg尽可能不要bank conflict
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    // const int b_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4; // *4 是因为float4
    const int b_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;
    // const int a_tile_index = warp_id / 2 * 16 + lane_id / 8 * 4;
    const int a_tile_index = warp_id % 2 * 32 + lane_id % 8 * 4;

    

    //-------------------------------------    
    // 在进入循环之前 先从 GMem 预取出一部分到 SMem 中 也就是原先的循环的第一部分的操作
    // 之后进入循环 从 GMem 中取出数据存入 SMem 的剩下部分，并计算第一部分的sum 再同步 也就是把原先的两次同步变味一次同步
    int write = 0;
    int read = 0;

    // GMem2SMem
    FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, A_tile_tx * 4, K)]);
    A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
    A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
    A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
    A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

    FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
        FETCH_FLOAT4(d_B[OFFSET(B_tile_ty, col + B_tile_tx * 4, N)]);

    __syncthreads();
    write ^= 1;

    //  SMem2RMem
    FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
    FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
    FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
    FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);

    int tile_idx = 0;
    // 进入大循环
    do
    {
        tile_idx += K_per_BLOCK;
    #pragma unroll
        for (int inner_k = 0; inner_k < K_per_BLOCK - 1; ++inner_k)
        {
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index]);
            FETCH_FLOAT4(a_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(A_block[read][inner_k + 1][a_tile_index + 64]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][0]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index]);
            FETCH_FLOAT4(b_reg[(inner_k + 1) % 2][4]) = FETCH_FLOAT4(B_block[read][inner_k + 1][b_tile_index + 64]);
            // 计算预取的和
            #pragma unroll
            for (int i = 0; i < Y_per_THREAD; ++i)
            {
                #pragma unroll
                for (int j = 0; j < X_per_THREAD; ++j)
                {
                    sum[i][j] += a_reg[inner_k % 2][i] * b_reg[inner_k % 2][j];
                }
            }
        }
        read ^= 1;
        if (tile_idx < K)
        {
            FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(d_A[OFFSET(row + A_tile_ty, tile_idx + A_tile_tx * 4, K)]);
            A_block[write][A_tile_tx * 4 + 0][A_tile_ty] = a_tmp[0];
            A_block[write][A_tile_tx * 4 + 1][A_tile_ty] = a_tmp[1];
            A_block[write][A_tile_tx * 4 + 2][A_tile_ty] = a_tmp[2];
            A_block[write][A_tile_tx * 4 + 3][A_tile_ty] = a_tmp[3];

            FETCH_FLOAT4(B_block[write][B_tile_ty][B_tile_tx * 4]) =
                FETCH_FLOAT4(d_B[OFFSET(B_tile_ty + tile_idx, col + B_tile_tx * 4, N)]);
            write ^= 1;
        }

        __syncthreads();
        FETCH_FLOAT4(a_reg[0][0]) = FETCH_FLOAT4(A_block[read][0][a_tile_index]);
        FETCH_FLOAT4(a_reg[0][4]) = FETCH_FLOAT4(A_block[read][0][a_tile_index + 64]);
        FETCH_FLOAT4(b_reg[0][0]) = FETCH_FLOAT4(B_block[read][0][b_tile_index]);
        FETCH_FLOAT4(b_reg[0][4]) = FETCH_FLOAT4(B_block[read][0][b_tile_index + 64]);
        #pragma unroll
        for (int i = 0; i < Y_per_THREAD; ++i)
        {
            #pragma unroll
            for (int j = 0; j < X_per_THREAD; ++j)
            {
                sum[i][j] += a_reg[1][i] * b_reg[1][j];
            }
        }

    } while (tile_idx < K);

    // 同步 下一步循环要清空SMem 必须要把数据用完

    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i][4]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index, N)]) = FETCH_FLOAT4(sum[i + 4][0]);
    }
    for (int i = 0; i < 4; ++i)
    {
        FETCH_FLOAT4(d_C[OFFSET(row + a_tile_index + i + 64, col + b_tile_index + 64, N)]) = FETCH_FLOAT4(sum[i + 4][4]);
    }
}

void Double_Buffer_RMem_SMem(int M, int K, int N, float *__restrict__ d_A, float *__restrict__ d_B, float *__restrict__ d_C)
{
    checkCudaErrors(cudaDeviceSynchronize());
    constexpr int M_per_BLOCK = 128;
    constexpr int K_per_BLOCK = 8;
    constexpr int N_per_BLOCK = 128;
    constexpr int Y_per_THREAD = 8;
    constexpr int X_per_THREAD = 8;
    dim3 block(N_per_BLOCK/X_per_THREAD,M_per_BLOCK/Y_per_THREAD);// 16 * 16 
    dim3 grid((N + N_per_BLOCK- 1) / (N_per_BLOCK), (M + M_per_BLOCK - 1) / (M_per_BLOCK));
    gemm_v8<M_per_BLOCK, K_per_BLOCK, N_per_BLOCK,Y_per_THREAD, X_per_THREAD><<<grid, block>>>(M, K, N, d_A, d_B, d_C);
    checkCudaErrors(cudaDeviceSynchronize());
}

int num = 2048;
int nIter = 100;
// C = A * B
__global__ void warmup()
{
    printf("warmup!\n");
}
int main()
{
    // sizeof A B C

    int M = num;
    int K = num;
    int N = num;

    // print gpu info
    cudaDeviceProp deviceProp;
    int devID = 0;
    checkCudaErrors(cudaSetDevice(devID));
    auto error = cudaGetDeviceProperties(&deviceProp, devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error,
               __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID,
           deviceProp.name, deviceProp.major, deviceProp.minor);

    // 计时
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0.0f;
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    // cublas 参数
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // sizeof A B C mem
    const size_t A_mem_size = M * K * sizeof(float);
    const size_t B_mem_size = N * K * sizeof(float);
    const size_t C_mem_size = M * N * sizeof(float);

    float *h_A = (float *)malloc(A_mem_size);
    float *h_B = (float *)malloc(B_mem_size);
    float *h_C_cublas = (float *)malloc(C_mem_size);
    float *h_C_cpu = (float *)malloc(C_mem_size);
    float *h_C_gpu = (float *)malloc(C_mem_size);

    init_matrix(M, K, h_A);
    init_matrix(K, N, h_B);
    memset(h_C_cublas, 0, C_mem_size);
    memset(h_C_cpu, 0, C_mem_size);
    memset(h_C_gpu, 0, C_mem_size);

    float *d_A, *d_B, *d_C, *d_C_cublas;
    checkCudaErrors(cudaMalloc((void **)&d_A, A_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_B, B_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_C, C_mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_C_cublas, C_mem_size));

    // host2dev
    checkCudaErrors(cudaMemcpy(d_A, h_A, A_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, B_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C_cublas, h_C_cublas, C_mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_C, h_C_gpu, C_mem_size, cudaMemcpyHostToDevice));

    warmup<<<1, 1>>>();
    //-------------------------------------gpu_calc----------------------------------------------------
    checkCudaErrors(cudaEventRecord(start)); // 0 默认流
    for (int run = 0; run < nIter; run++)
    {
        // printf("this is gemm_v8\n");
        Double_Buffer_RMem_SMem(M, K, N, d_A, d_B, d_C);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop)); // 确保事件完成
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf("My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[0],
           msecPerMatrixMul[0],
           flopsPerMatrixMul);
    //-----------------------------------------------------------------------------------------
    // dev2host
    checkCudaErrors(cudaMemcpy(h_C_gpu, d_C, C_mem_size, cudaMemcpyDeviceToHost));

    // cublas_calc 用来判断速度
    checkCudaErrors(cudaEventRecord(start)); // 0 默认流
    for (int run = 0; run < nIter; run++)
    {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop)); // 确保事件完成
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf("CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
           gigaFlops[1],
           msecPerMatrixMul[1],
           flopsPerMatrixMul);

    checkCudaErrors(cudaMemcpy(h_C_cublas, d_C_cublas, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    // cpu_calc 用来判断准确率
    // cpu_gemm(h_A,h_B,h_C_cpu,M,N,K);

    // compare res
    printf("mysgemm ans is ");
    compare_ans(h_C_cublas, h_C_gpu, M, N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_cublas);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_cublas);
    cublasDestroy(handle);
    return 0;
}
