#include <torch/extension.h>

// 8bit权重对称量化
at::Tensor w8a16_ws_gemm_cuda(const at::Tensor &qact, const at::Tensor &qweight, const at::Tensor &w_scale, std::optional<at::Tensor> &out_);