# RUN: python -m test.test_w4a8_linear

import torch
from edq.quantization import quantize_tensor_int8
from thirdparty.Qserve.w4a8_linear import W4A8Linear

M = 16
K = 256
N = 64

group_size = 128

input_tensor = torch.randn((M, K), dtype=torch.float16, device='cuda')
input_int8, qconfig_input = quantize_tensor_int8(input_tensor)
output_tensor = torch.zeros((M,N), dtype=torch.float16, device = 'cuda')

weight_tensor = torch.randn((N, K), dtype=torch.float16, device='cuda')

q_linear = W4A8Linear.from_weight(weight_tensor, group_size)

q_linear.forward(input_int8, qconfig_input['scale'], 0, output_tensor)

print(output_tensor)
print(input_tensor @ weight_tensor.T)