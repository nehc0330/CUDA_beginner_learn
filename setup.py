from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_edouble_buffer_gemmxtension",
    ext_modules=[
        CUDAExtension(
            "gemm",
            ["my_gemm.cpp", "utlis.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
