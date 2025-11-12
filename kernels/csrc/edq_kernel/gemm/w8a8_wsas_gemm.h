#include <torch/extension.h>

// 8bit权重对称量化，8bit激活对称量化，均不考虑分组量化
at::Tensor w8a8_wsas_gemm_cuda(const at::Tensor &qact, const at::Tensor &qweight, const at::Tensor &act_scale, const at::Tensor &w_scale, std::optional<at::Tensor> &out_);