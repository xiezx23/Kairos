#pragma once
#include <torch/extension.h>

torch::Tensor dequantize_i4_to_f16_cuda(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int IC,
    int OC,
    int group_size);