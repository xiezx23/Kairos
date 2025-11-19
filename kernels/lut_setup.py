from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
}

ext_modules = [
    Pybind11Extension(
        "LUT_module",
        [
            "csrc/kairos_kernel/lut/lut_bindings.cpp",
        ],
        include_dirs=["csrc/kairos_kernel/lut/"],
        language='c++'
    ),
]

setup(
    name="LUT_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)