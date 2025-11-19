#include <torch/extension.h>

void trans_layout_c16_to_c8(
        torch::Tensor &output,      // [K/16, N*16/8], dtype=torch.int32
        torch::Tensor &input        // [K/16, N*16/8], dtype=torch.int32
);