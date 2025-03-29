#include <cuda_runtime.h>
#include <cstdio>
#include <cublas_v2.h>
#include "utlis.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>

int main()
{
    // �����С
    constexpr int M = 19;
    constexpr int N = 19;
    const int size = M * N;

    // cublas���transpose��Ҫ�õ�alpha��beta����
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // �����ڴ����
    float *h_input, *h_output;
    h_input = (float *)malloc(size * sizeof(float));
    h_output = (float *)malloc(size * sizeof(float));

    float *h_output_cublas;
    h_output_cublas = (float *)malloc(size * sizeof(float));

    // �����豸�ڴ�
    float *d_input, *d_output;
    // float *d_output_cublas;
    checkCudaErrors(cudaMalloc((void **)&d_input, size * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_output, size * sizeof(float)));

    float *d_output_cublas;
    checkCudaErrors(cudaMalloc((void **)&d_output_cublas, size * sizeof(float)));

    // ��ʼ���������
    initMatrix(h_input, M, N);
    // initIdentityMatrix(h_input_cublas, N, M);
    // �������ݵ��豸
    checkCudaErrors(cudaMemcpy(d_input, h_input, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // part 1: naive transpose
    // naive best is 32*8 64*4
    // naive_16_16(M, N, d_input, d_output);
    // naive_8_32(M, N, d_input, d_output);
    // naive_32_8(M, N, d_input, d_output);
    // naive_64_4(M,N,d_input, d_output);
    // naive_4_64(M,N,d_input, d_output);
    // naive_2_128(M,N,d_input, d_output);
    // naive_128_2(M, N, d_input, d_output);
    // naive_1_256(M,N,d_input, d_output);
    // naive_256_1(M, N, d_input, d_output);

    // part 2: SMem transpose
    SMem_16_16(M, N, d_input, d_output); // û���κ�����
    SMem_16_16_pad(M, N, d_input, d_output);

    // // part 3: Float2 transpose
    // Float2_2x2_32x8(M, N, d_input, d_output);
    // Float2_2x2_8x32(M, N, d_input, d_output);
    // Float2_2x2_16x16(M, N, d_input, d_output);

    // // part 4: Float4 1 * 4transpose
    // Float4_1x4_32x8(M, N, d_input, d_output);
    // Float4_1x4_8x32(M, N, d_input, d_output);
    // Float4_1x4_16x16(M, N, d_input, d_output);

    // // part 5: Float4 4 * 4transpose
    // Float4_4x4_32x8(M, N, d_input, d_output);
    // Float4_4x4_8x32(M, N, d_input, d_output);
    // Float4_4x4_16x16(M, N, d_input, d_output);

    // Float4_16_16_resize_SMem(M, N, d_input, d_output);
    checkCudaErrors(cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));

    // ����cublas���transpose
    // ʹ��cublasSgeam����ת�ò���
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, &alpha, d_input, M, &beta, NULL, N, d_output_cublas, N);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_output_cublas, d_output_cublas, size * sizeof(float), cudaMemcpyDeviceToHost));
    // �˺�����cublas��Ľ���Ա�
    bool correct = true;
    for (int i = 0; i < size; ++i)
    {
        if (fabs(h_output[i] - h_output_cublas[i]) > 1e-2)
        {
            correct = false;
            printf("Test failed at index %d: %.2f vs %.2f\n", i, h_output[i], h_output_cublas[i]);
            break;
        }
    }

    if (correct)
    {
        printf("Test passed!\n");
    }
    else
    {
        printf("Test failed!\n");
    }

    // �ͷ������ڴ�
    free(h_input);
    free(h_output_cublas);
    free(h_output);
    // free(h_output_cublas);

    // ����cublas��
    cublasDestroy(handle);

    // �ͷ��豸�ڴ�
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_cublas);
    cudaFree(d_output);

    return 0;
}
