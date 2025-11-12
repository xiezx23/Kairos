#include <torch/extension.h>

// 8bit权重对称量化
at::Tensor w8a16_wa_gemm_cuda(const at::Tensor &qact, const at::Tensor &qweight, const at::Tensor &w_scale, const at::Tensor &w_zp, std::optional<at::Tensor> &out_);