# RUN: python3 -m test.bitblas.linear_w8a16

import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend

# enabling debug output
# bitblas.set_log_level("Debug")
torch.manual_seed(seed=10)
M = 16
K = 18944
N = 3584

# Create input matrices
input  = torch.randn((M, K), dtype=torch.float16).cuda()
weight = torch.randn((N, K), dtype=torch.float16).cuda()

input_int8, config_int8_x  = quantize_tensor_int8(input)
input = dequantize_tensor_int8(input_int8, config_int8_x['scale'])
weight_int8, config_int8_w = quantize_tensor_int8(weight)
weight = dequantize_tensor_int8(weight_int8, config_int8_w['scale'])

n = 20
recordList = [[],[],[]]
# input_len = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
input_len = [i for i in range(1, 2028, 16)]
for m in input_len:
    print('input M: ', m)
    input  = torch.randn((m, K), dtype=torch.float16).cuda()
    # input  = torch.randint(-128, 127 + 1, (m, K), dtype=torch.int8).cuda()
    input_int8, config_int8_x  = quantize_tensor_int8(input)
    input = dequantize_tensor_int8(input_int8, config_int8_x['scale'])
    print(gre_prefix, end='')
    scale_w = config_int8_w['scale'].squeeze(-1)
    for _ in range(5):
        input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
        scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
        edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
        output1 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
        edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, output1)
    with timer('EDQ W8A8 ' , n = n, recordList=recordList[0]):
        for _ in range(n):
            input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
            output1 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, output1)
    print(default_color, end='')
    
    for _ in range(5):
        output2 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
        edq_cuda_accel.w8a16_ws_gemm_cuda(input, weight_int8, scale_w, output2)
    with timer('EDQ W8A16' , n = n, recordList=recordList[1]):
        for _ in range(n):
            output2 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.w8a16_ws_gemm_cuda(input, weight_int8, scale_w, output2)
    input = input.to(torch.float16)

    for _ in range(5):
        refout = input @ weight.T
    with timer('Pyto Fp16', n = n, recordList=recordList[2]):
        for _ in range(n):
            refout = input @ weight.T
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
