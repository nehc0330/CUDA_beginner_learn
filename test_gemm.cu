#include <cstdio>
#include <cuda.h>
#include "utlis.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int num = 1024;
int nIter = 1;
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
        // test_v1;
        // printf("this is gemm_v1\n");
        // GlobalMemory(M,K,N,d_A,d_B,d_C);

        // test_v2;
        // printf("this is gemm_v2\n");
        // ShareMemory(M,K,N,d_A,d_B,d_C);

        // test_v3 21.7ms
        // printf("this is gemm_v3\n");
        // STRIDE_ShareMemory(M,K,N,d_A,d_B,d_C);

        // test_v4 7.4ms
        // printf("this is gemm_v4\n");
        // Float4_ShareMemory(M,K,N,d_A,d_B,d_C);

        // test_v5 4.9ms
        // printf("this is gemm_v5\n");
        // RMem_Float4_ShareMemory(M,K,N,d_A,d_B,d_C);

        // test_v6 4.4ms
        // printf("this is gemm_v6\n");
        // Transpose_RMem_Float4_ShareMemory(M,K,N,d_A,d_B,d_C);

        // test_v7 2.9ms
        // printf("this is gemm_v7\n");
        // Buffer_Transpose_RMem_Float4_ShareMemory(M, K, N, d_A, d_B, d_C);

        // test_v8 2.6
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
