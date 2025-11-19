import os
import sys
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

<<<<<<< HEAD

=======
>>>>>>> main
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
<<<<<<< HEAD
    name="kairos_cuda_accel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="kairos_cuda_accel",
            sources=[
                "csrc/pybind.cpp",
                "csrc/kairos_kernel/gemm/w8a8_wsas_gemm.cu",
                "csrc/kairos_kernel/gemm/w8a16_ws_gemm.cu",
                "csrc/kairos_kernel/gemm/w8a16_wa_gemm.cu",

                "csrc/kairos_kernel/gemm/w4a16_wa_gemm.cu",
                "csrc/kairos_kernel/gemv/w8a16_ws_gemv.cu",
                "csrc/kairos_kernel/gemv/w8a8_wsas_gemv.cu",
                "csrc/kairos_kernel/gemv/w4a16_wa_gemv.cu",

                "csrc/kairos_kernel/quant/quant_f16_to_i8.cu",

                "csrc/kairos_kernel/dequant/dequant_i4_to_i8.cu",
                "csrc/kairos_kernel/dequant/dequant_i4_to_f16.cu",
                "csrc/kairos_kernel/dequant_with_event/dequant_i4_to_i8.cu",
                "csrc/kairos_kernel/dequant_with_event/dequant_i4_to_f16.cu",
                
                "csrc/kairos_kernel/dequant_awq_format/dequant_i4_to_i8.cu",
                "csrc/kairos_kernel/dequant_awq_format/dequant_i4_to_f16.cu",
                
                "csrc/kairos_kernel/trans_layout/trans_layout_marlin_c16_to_c8.cu",
            ],
            extra_compile_args=extra_compile_args,
            include_dirs=[
                Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "thirdparty" / "cutlass" / "include"
            ],
=======
    name="kairos_cuda_accel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="kairos_cuda_accel",
            sources=[
                "csrc/pybind.cpp",
                "csrc/kairos_kernel/quant/quant_f16_to_i8.cu",

                "csrc/kairos_kernel/dequant/dequant_i4_to_i8.cu",
                "csrc/kairos_kernel/dequant/dequant_i4_to_f16.cu",
                
                "csrc/kairos_kernel/dequant_awq_format/dequant_i4_to_i8.cu",
                "csrc/kairos_kernel/dequant_awq_format/dequant_i4_to_f16.cu",
                
                "csrc/kairos_kernel/trans_layout/trans_layout_marlin_c16_to_c8.cu",
            ],
            extra_compile_args=extra_compile_args,
            # include_dirs=[
            #     Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "thirdparty" / "cutlass" / "include"
            # ],
>>>>>>> main
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
