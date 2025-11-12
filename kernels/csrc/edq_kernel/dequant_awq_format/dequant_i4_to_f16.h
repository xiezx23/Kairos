#include <torch/extension.h>

void dequant_interleaved_int4_to_fp16(
        torch::Tensor &output,      // [N, K], dtype=torch.float16
        torch::Tensor &input,       // [N/interleave, K], dtype=torch.int16  
        // [N, K/group_size] or [K/group_size, N], dtype=torch.float16
        torch::Tensor &scale,
        torch::Tensor &scaled_zero,
        int group_size);