#include <torch/extension.h>

void dequant_int4_to_int8(torch::Tensor &out,   // [N, K]
                        torch::Tensor &input,   // [N, K/2]
                        torch::Tensor &scale,   // [N, K/group_size]
                        torch::Tensor &zero,    // [N, K/group_size]
                        int group_size);

void dequant_int4_to_int8_stream(torch::Tensor &out,   // [N, K]
                        torch::Tensor &input,   // [N, K/2]
                        torch::Tensor &scale,   // [N, K/group_size]
                        torch::Tensor &zero,    // [N, K/group_size]
                        int group_size,
                        uintptr_t stream_ptr);