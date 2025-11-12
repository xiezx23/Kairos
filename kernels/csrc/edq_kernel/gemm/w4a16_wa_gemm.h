#include <torch/extension.h>

// Marlin 4bit权重对称量化
void w4a16_wa_gemm_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C,
         const torch::Tensor &s, const torch::Tensor &sz, torch::Tensor &workspace, int thread_k = -1,
         int thread_n = -1, int sms = -1, int max_par = 16);