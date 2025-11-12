#include <torch/extension.h>

void dequant_int4_to_fp16(torch::Tensor &out,   // [N, K]
                        torch::Tensor &input,   // [N, K/2]
                        torch::Tensor &scale,   // [N, K/group_size]
                        torch::Tensor &zero,    // [N, K/group_size]
                        int group_size);

void dequant_int4_to_fp16_stream(torch::Tensor &output,// [N, K], dtype=torch.float16
                        torch::Tensor &input,   // [N, K/2], dtype=torch.int8
                        torch::Tensor &scale,   // [N, K/group_size], dtype=torch.float16
                        torch::Tensor &zero,    // [N, K/group_size], dtype=torch.float16
                        int group_size,
                        uintptr_t stream_ptr);