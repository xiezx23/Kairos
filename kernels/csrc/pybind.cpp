#include <pybind11/pybind11.h>
#include <torch/extension.h>

// QGEMM WORKSPACE
#include "edq_kernel/gemm/w8a8_wsas_gemm.h"
#include "edq_kernel/gemm/w8a16_ws_gemm.h"
#include "edq_kernel/gemm/w8a16_wa_gemm.h"
#include "edq_kernel/gemm/w4a16_wa_gemm.h"
// QGEMV WORKSPACE
#include "edq_kernel/gemv/w8a16_ws_gemv.h"
#include "edq_kernel/gemv/w8a8_wsas_gemv.h"
#include "edq_kernel/gemv/w4a16_wa_gemv.h"
// QUANTIZATION WORKSPACE
#include "edq_kernel/quant/quant_f16_to_i8.h"
// DEQUANTIZATION WORKSPACE
#include "edq_kernel/dequant/dequant_i4_to_i8.h"
#include "edq_kernel/dequant/dequant_i4_to_f16.h"
#include "edq_kernel/dequant_with_event/dequant_i4_to_i8.h"
#include "edq_kernel/dequant_with_event/dequant_i4_to_f16.h"

#include "edq_kernel/dequant_awq_format/dequant_i4_to_i8.h"
#include "edq_kernel/dequant_awq_format/dequant_i4_to_f16.h"

#include "edq_kernel/trans_layout/trans_layout_marlin_c16_to_c8.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("w8a8_wsas_gemm_cuda", &w8a8_wsas_gemm_cuda, "Edq w8a8 gemm kernel, support symmetry quantization and int8 dtype");
    m.def("w8a16_ws_gemm_cuda", &w8a16_ws_gemm_cuda, "Edq w8a16 gemm kernel, support symmetry quantization and int8 dtype");
    m.def("w8a16_wa_gemm_cuda", &w8a16_wa_gemm_cuda, "Edq w8a16 gemm kernel, support asymmetry quantization and int8/uint8 dtype");
    m.def("w4a16_wa_gemm_cuda", &w4a16_wa_gemm_cuda, "w4a16 gemm kernel, support symmetry quantization and int4 dtype");

    m.def("w8a16_ws_gemv_cuda", &w8a16_ws_gemv_cuda, "w8a16 gemv, w:sym.");
    m.def("w8a8_wsas_gemv_cuda", &w8a8_wsas_gemv_cuda, "w8a8 gemv, w:sym a:sym.");
    m.def("w4a16_wa_gemv_cuda", &w4a16_wa_gemv_cuda, "w4a16 gemv kernel, support symmetry quantization and int4 dtype");

    m.def("quant_fp16_to_int8", &quant_fp16_to_int8, "Quant_fp16_to_int8 function");

    m.def("dequant_int4_to_int8", &dequant_int4_to_int8, "Dequant_int4_to_int8 function");
    m.def("dequant_int4_to_fp16", &dequant_int4_to_fp16, "Dequant_int4_to_float16 function");
    m.def("dequant_int4_to_int8_stream", &dequant_int4_to_int8_stream, "Dequant_int4_to_int8 function");
    m.def("dequant_int4_to_fp16_stream", &dequant_int4_to_fp16_stream, "Dequant_int4_to_float16 function");
    m.def("dequant_int4_to_int8_new", &dequant_int4_to_int8_new, "Dequant_int4_to_int8 function");

    m.def("dequant_int4_to_int8_new_2", &dequant_int4_to_int8_new_2, "Dequant_int4_to_int8 function");

    m.def("dequant_int4_to_fp16_new", &dequant_int4_to_fp16_new, "Dequant_int4_to_float16 function");
    
    m.def("dequant_interleaved_int4_to_fp16", &dequant_interleaved_int4_to_fp16, "dequant_interleaved_int4_to_fp16");
    m.def("dequant_interleaved_int4_to_int8", &dequant_interleaved_int4_to_int8, "dequant_interleaved_int4_to_int8");
    
    m.def("trans_layout_c16_to_c8", &trans_layout_c16_to_c8, "transformate layout of weight from W4A16 format to W4A8 format");
}