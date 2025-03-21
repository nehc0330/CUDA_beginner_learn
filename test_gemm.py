import torch
import time
from torch.utils.cpp_extension import load

# 1. 加载 C++/CUDA 扩展
gemm = load(
    name="double_buffer_gemm",
    sources=["my_gemm.cpp", "utlis.cu"],  # 你自己的 C++ 和 CUDA 文件
    extra_cuda_cflags=["-O3"],
    verbose=True,
)

# 2. 输入矩阵大小（你可以调大点看看性能变化）
M, K, N = 1024, 1024, 1024

# 3. 随机生成输入张量（float32，放到 CUDA）
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)
C = torch.zeros(M, N, device="cuda", dtype=torch.float32)  # 输出矩阵

# 4. 热身
gemm.torch_launch_gemm(A, B, C, M, N, K)
torch.cuda.synchronize()

# 5. 正式测试你的 kernel
start = time.time()
gemm.torch_launch_gemm(A, B, C, M, N, K)
torch.cuda.synchronize()
custom_time = time.time() - start
print(f"[Custom GEMM] Time: {custom_time:.4f} s")

# 6. 使用 PyTorch matmul 对比性能
start = time.time()
C_ref = torch.matmul(A, B)
torch.cuda.synchronize()
torch_time = time.time() - start
print(f"[Torch matmul] Time: {torch_time:.4f} s")

# 7. 验证精度
max_abs_diff = (C - C_ref).abs().max()
print(f"[Correctness] Max Abs Diff: {max_abs_diff:.6f}")
