import os
import sys
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=32",
    ],
}

setup(
    name="awq_backend",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="awq_backend",
            sources=[
                "csrc/pybind.cpp",
                "csrc/quantization/gemm_cuda_gen.cu",
                "csrc/quantization/gemv_cuda.cu",
                "csrc/quantization_new/gemv/gemv_cuda.cu",
                "csrc/quantization_new/gemm/gemm_cuda.cu",
                "csrc/w8a8/w8a8_gemm_cuda.cu",
                "csrc/w8a8/quantization.cu",
                "csrc/w8a8/act.cu",
                "csrc/w8a8/layernorm.cu",

                "csrc/dequant_kairos/deq_i4_to_f16.cu"

            ],
            extra_compile_args=extra_compile_args,
            include_dirs=[
                Path(os.path.dirname(os.path.abspath(__file__))) / "csrc" / "cutlass" / "include"
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
