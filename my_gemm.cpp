#include <torch/extension.h>
#include "utlis.cuh"

void torch_launch_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int K, int N)
{
    // 确保输入是 CUDA tensor，
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    // 确保输入是 float32
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(C.scalar_type() == torch::kFloat32, "C must be float32");

    // 获取数据指针
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    Double_Buffer_RMem_SMem(M, K, N, d_A, d_B, d_C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("torch_launch_gemm", &torch_launch_gemm, "torch_launch_gemm kernel warpper"); }
