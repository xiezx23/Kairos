#include <pybind11/pybind11.h>
#include <torch/extension.h>
// #include "attention/ft_attention.h"
// #include "layernorm/layernorm.h"
#include "quantization/gemm_cuda.h"
#include "quantization/gemv_cuda.h"
#include "quantization_new/gemm/gemm_cuda.h"
#include "quantization_new/gemv/gemv_cuda.h"
// #include "position_embedding/pos_encoding.h"
// #include "rope_new/fused_rope_with_pos.h"
#include "w8a8/w8a8_gemm_cuda.h"
#include "w8a8/quantization.h"
#include "w8a8/layernorm.h"
#include "w8a8/act.h"

<<<<<<< HEAD
#include "dequant_kairos/deq_i4_to_f16.h"
=======
#include "dequant_kairos/deq_i4_to_f16.h"
>>>>>>> main



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{   
    m.def("gemm_forward_cuda", &gemm_forward_cuda, "Quantized GEMM kernel.");
    m.def("gemv_forward_cuda", &gemv_forward_cuda, "Quantized GEMV kernel.");
    m.def("gemm_forward_cuda_new", &gemm_forward_cuda_new, "New quantized GEMM kernel.");
    m.def("gemv_forward_cuda_new", &gemv_forward_cuda_new, "New quantized GEMV kernel.");
    m.def("w8a8_gemm_forward_cuda", &w8a8_gemm_forward_cuda, "our w8a8 gemm kernel");
    m.def("w8a8_gemm_fuse_bias_forward_cuda", &w8a8_gemm_fuse_bias_forward_cuda, "our w8a8 gemm fused bias kernel");
    m.def("invoke_quant", &invoke_quant, "fp16->int8 quantization");

    m.def("dequantize_i4_to_f16_cuda", &dequantize_i4_to_f16_cuda, "dequantize_i4_to_f16_cuda");
}
