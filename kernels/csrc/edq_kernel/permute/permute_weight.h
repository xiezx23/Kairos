#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

void permute_weight_cuda(torch::Tensor &d_input, torch::Tensor &d_output,
                               int in_features, int out_features);