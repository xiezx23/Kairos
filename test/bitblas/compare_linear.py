# RUN: python3 -m test.bitblas.linear_w8a16

import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
import matplotlib.pyplot as plt

# enabling debug output
# bitblas.set_log_level("Debug")
torch.manual_seed(seed=10)
M = 16
K = 18944
N = 3584

model_w8a8 = bitblas.Linear(
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

model_w8a16 = bitblas.Linear(
    in_features=K,
    out_features=N,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="int8",  # weight W dtype
    accum_dtype="float32",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode='original',  # setting for how to calculating zeros
    # Target optimization var for dynamic symbolic.
    # For detailed information please checkout docs/PythonAPI.md
    # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
    opt_M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
)

# Create input matrices
input_tensor  = torch.randn((M, K), dtype=torch.float16).cuda()
# input_tensor  = torch.randint(-128, 127 + 1, (M, K), dtype=torch.int8).cuda()
# input_tensor   = torch.eye((K), dtype=torch.float16).cuda()
# input_tensor   = torch.randint(-64, 64 + 1, (M, K), dtype=torch.int8).cuda()
weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

input_int8, config_int8_x  = quantize_tensor_int8(input_tensor)
input_tensor = dequantize_tensor_int8(input_int8, config_int8_x['scale'])
weight_int8, config_int8_w = quantize_tensor_int8(weight_tensor)
weight_tensor = dequantize_tensor_int8(weight_int8, config_int8_w['scale'])

# Set the model to evaluation mode
model_w8a8.eval()
# Load and transform weights into the BitBLAS linear module
model_w8a8.load_and_transform_weight(weight_int8)
model_w8a8.cuda()

model_w8a16.eval()
# Load and transform weights into the BitBLAS linear module
model_w8a16.load_and_transform_weight(weight_int8, scales=config_int8_w['scale'])
# Set the model_w8a16 to evaluation mode
model_w8a16.cuda()

n = 20
recordList = [[],[],[]]
# input_len = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
input_len = [i for i in range(1, 2028, 16)]
for m in input_len:
    print('input M: ', m)
    input_tensor  = torch.randn((m, K), dtype=torch.float16).cuda()
    # input_tensor  = torch.randint(-128, 127 + 1, (m, K), dtype=torch.int8).cuda()
    input_int8, config_int8_x  = quantize_tensor_int8(input_tensor)
    input_tensor = dequantize_tensor_int8(input_int8, config_int8_x['scale'])
    print(gre_prefix, end='')
    scales = config_int8_w['scale'].T * config_int8_x['scale']
    for _ in range(5):
        input_int8, config_int8_x  = quantize_tensor_int8(input_tensor)
        output1 = model_w8a8(input_int8)
        output1.mul_(scales)
    with timer('BitBLAS W8A8 ', n = n, recordList=recordList[0]):
        for _ in range(n):
            input_int8, config_int8_x  = quantize_tensor_int8(input_tensor)
            output1 = model_w8a8(input_int8)
            output1.mul_(scales)
    print(default_color, end='')
    input_tensor = input_tensor.to(torch.float16)
    for _ in range(5):
        output2 = model_w8a16(input_tensor)
    with timer('BitBLAS W8A16', n = n, recordList=recordList[1]):
        for _ in range(n):
            output2 = model_w8a16(input_tensor)
    # print("BitBLAS output:", output)
    for _ in range(5):
        refout = input_tensor @ weight_tensor.T
    with timer('PyTorch Fp16 ', n = n, recordList=recordList[2]):
        for _ in range(n):
            refout = input_tensor @ weight_tensor.T
    # print("RefFp16 Output:", refout)
    # print(refout.shape)
    # print(output1.shape)
    # print(output2.shape)
    # exit(0)
    # loss = (refout-output1).abs().mean()
    # print('loss1:', loss.item())
    # loss = (refout-output2).abs().mean()
    # print('loss2:', loss.item())
    # print('---------------------')

plt.figure()
plt.plot(recordList[0], color = 'r')
plt.plot(recordList[1], color = 'b')
plt.plot(recordList[2], color = 'g')
# plt.xticks(input_len)
plt.grid(True)
plt.show()

# maxDif = (refout-output).abs().max()
# loss = (refout-output).abs().mean()
# print('max diff:', maxDif.item())
# print('loss:    ', loss.item())
