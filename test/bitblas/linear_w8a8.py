# RUN: python3 -m test.bitblas.linear_w8a16

import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *

# enabling debug output
# bitblas.set_log_level("Debug")
torch.manual_seed(seed=10)
M = 16
K = 3584
N = 3584

model = bitblas.Linear(
    in_features=K,
    out_features=N,
    bias=False,
    A_dtype="int8",  # activation A dtype
    W_dtype="int8",  # weight W dtype
    accum_dtype="int32",  # accumulation dtype
    out_dtype="float32",  # output dtype
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
    # Target optimization var for dynamic symbolic.
    # For detailed information please checkout docs/PythonAPI.md
    # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
    opt_M=[1, 16, 32, 64, 128, 256, 512],
)

# Create input matrices
input_tensor  = torch.randn((M, K), dtype=torch.float16).cuda()
weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

weight_int8_edq, config_int8_edq = quantize_tensor_int8(weight_tensor)
weight_tensor = dequantize_tensor_int8(weight_int8_edq, config_int8_edq['scale'])

# Set the model to evaluation mode
model.eval()
# Load and transform weights into the BitBLAS linear module
model.load_and_transform_weight(weight_int8_edq)
model.cuda()

# Preheating Model
output = model(input_tensor)
refout = input_tensor.to(torch.float16) @ weight_tensor.T

n = 20
input_len = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
for m in input_len:
    print('input M: ', m)
    input_tensor  = torch.randint(-128, 127 + 1, (m, K), dtype=torch.int8).cuda()
    print(gre_prefix, end='')
    scales = config_int8_edq['scale'].T
    with timer('BitBLAS', n = n):
        for _ in range(n):
            output = model(input_tensor)
            output.mul_(scales)
    print(default_color, end='')
    # print("BitBLAS output:", output)
    input_tensor = input_tensor.to(torch.float16)
    with timer('RefFp16', n = n):
        for _ in range(n):
            refout = input_tensor @ weight_tensor.T
    loss = (refout-output).abs().mean()
    print('loss:', loss.item())
    print('---------------------')

# maxDif = (refout-output).abs().max()
# loss = (refout-output).abs().mean()
# print('max diff:', maxDif.item())
# print('loss:    ', loss.item())
