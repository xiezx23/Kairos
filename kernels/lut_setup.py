from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
}

ext_modules = [
    Pybind11Extension(
        "LUT_module",
        [
<<<<<<< HEAD
            "csrc/kairos_kernel/lut/lut_bindings.cpp",  # 包含上面绑定代码的文件
        ],
        include_dirs=["csrc/kairos_kernel/lut/"],  # 包含lut.h的目录
=======
            "csrc/kairos_kernel/lut/lut_bindings.cpp",  # 包含上面绑定代码的文件
        ],
        include_dirs=["csrc/kairos_kernel/lut/"],  # 包含lut.h的目录
>>>>>>> main
        language='c++'
    ),
]

setup(
    name="LUT_module",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)