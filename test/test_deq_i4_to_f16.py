

import torch
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from edq.w8a8_linear import W8A8Linear
from edq.w4a16_linear import W4A16Linear
from thirdparty.Qserve.w4a8_linear import W4A8Linear

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

MSELoss = torch.nn.MSELoss()

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

bias = False

def test(K, N):
    print(f'K: {K} N: {N}')
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    qLinear = W4A16Linear.from_module(torchLinear, 'cuda')

    for _ in range(20):
        dq_w = awq_backend.dequantize_i4_to_f16_cuda(qLinear.qweight, qLinear.scales, qLinear.scaled_zeros, K, N, 128)

    with timer(n = 100):
        for _ in range(100):
            dq_w = awq_backend.dequantize_i4_to_f16_cuda(qLinear.qweight, qLinear.scales, qLinear.scaled_zeros, K, N, 128)

    # print(torchLinear.weight.data)
    # print(dq_w)

if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])

exit(0)