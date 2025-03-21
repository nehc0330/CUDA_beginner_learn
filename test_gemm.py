import torch
import time
from torch.utils.cpp_extension import load

# 1. load C++/CUDA
gemm = load(
    name="double_buffer_gemm",
    sources=["my_gemm.cpp", "utlis.cu"],  # my C++ & CUDA
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# 2. input parameters
M, K, N = 1024, 1024, 1024

# 3. generate input tensors (float32, on CUDA)
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = torch.zeros(M, N, device="cuda", dtype=torch.float32)  # output matrix

# 4. warmup
gemm.torch_launch_gemm(A, B, C, M, N, K)
torch.cuda.synchronize()

# 5. test kernel
start = time.time()
gemm.torch_launch_gemm(A, B, C, M, N, K)
torch.cuda.synchronize()
custom_time = time.time() - start
print(f"[Custom GEMM] Time: {custom_time:.4f} s")

# 6. vs PyTorch matmul
start = time.time()
C_ref = torch.matmul(A, B)
torch.cuda.synchronize()
torch_time = time.time() - start
print(f"[Torch matmul] Time: {torch_time:.4f} s")

# 7. verify tol
max_abs_diff = (C - C_ref).abs().max()
print(f"[Correctness] Max Abs Diff: {max_abs_diff:.6f}")
