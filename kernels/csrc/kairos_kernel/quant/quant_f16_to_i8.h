#include <torch/extension.h>
void quant_fp16_to_int8(torch::Tensor &out,   // [..., hidden_size]
                        torch::Tensor &input, // [..., hidden_size]
                        torch::Tensor &scale);  // [num_tokens]