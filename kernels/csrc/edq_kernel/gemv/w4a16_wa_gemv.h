#include <torch/extension.h>

// Marlin 4bit权重对称量化
at::Tensor w4a16_wa_gemv_cuda(const at::Tensor &qact, const at::Tensor &qweight, const at::Tensor &w_scale, const at::Tensor &w_scaled_zp, std::optional<at::Tensor> &out_);