import torch
import time
from torch.utils.cpp_extension import load
import gemm  # setup.py should be run before running this code

# 1. load C++/CUDA
# gemm = load(
#     name="double_buffer_gemm",
#     sources=["my_gemm.cpp", "utlis.cu"],  # my C++ & CUDA
#     extra_cuda_cflags=["-O3"],
#     verbose=False,  # verbose 参数控制是否显示详细的编译过程。当设置为 True 时，编译过程中的详细信息（如编译命令、错误消息等）会显示在终端中。这对于调试编译问题很有帮助。
# )


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
