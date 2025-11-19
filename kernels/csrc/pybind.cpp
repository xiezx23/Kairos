#include <pybind11/pybind11.h>
#include <torch/extension.h>

// QUANTIZATION WORKSPACE
#include "edq_kernel/quant/quant_f16_to_i8.h"
// DEQUANTIZATION WORKSPACE
#include "edq_kernel/dequant/dequant_i4_to_i8.h"
#include "edq_kernel/dequant/dequant_i4_to_f16.h"

#include "edq_kernel/dequant_awq_format/dequant_i4_to_i8.h"
#include "edq_kernel/dequant_awq_format/dequant_i4_to_f16.h"

#include "edq_kernel/trans_layout/trans_layout_marlin_c16_to_c8.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_fp16_to_int8", &quant_fp16_to_int8, "Quant_fp16_to_int8 function");
    m.def("dequant_int4_to_int8", &dequant_int4_to_int8, "Dequant_int4_to_int8 function");
    m.def("dequant_int4_to_fp16", &dequant_int4_to_fp16, "Dequant_int4_to_float16 function");
    m.def("dequant_interleaved_int4_to_fp16", &dequant_interleaved_int4_to_fp16, "dequant_interleaved_int4_to_fp16");
    m.def("dequant_interleaved_int4_to_int8", &dequant_interleaved_int4_to_int8, "dequant_interleaved_int4_to_int8");
    m.def("trans_layout_c16_to_c8", &trans_layout_c16_to_c8, "transformate layout of weight from W4A16 format to W4A8 format");
}