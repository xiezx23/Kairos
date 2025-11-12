import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend

torch.manual_seed(seed=10)
M = 1
K = 3584
N = 18944

def test(K, N):
    # Create input matrices
    input  = torch.randn((M, K), dtype=torch.float16).cuda()
    weight = torch.randn((N, K), dtype=torch.float16).cuda()

    input_int8, config_int8_x  = quantize_tensor_int8(input)
    input = dequantize_tensor_int8(input_int8, config_int8_x['scale'])
    weight_int8, config_int8_w = quantize_tensor_int8(weight)
    weight = dequantize_tensor_int8(weight_int8, config_int8_w['scale'])

    n = 50
    recordList = [[] for _ in range(5)]
    input_len = [i for i in range(1, 8, 1)]
    for m in input_len:
        print('input M: ', m)
        input  = torch.randn((m, K), dtype=torch.float16).cuda()
        input_int8, config_int8_x  = quantize_tensor_int8(input)
        input = dequantize_tensor_int8(input_int8, config_int8_x['scale'])

        print(gre_prefix, end='')
        scale_w = config_int8_w['scale'].squeeze(-1)
        for _ in range(5):
            input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
            output4 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, output4)
        with timer('EDQ W8A8 ' , n = n, recordList=recordList[3]):
            for _ in range(n):
                input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
                output4 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, output4)

        for _ in range(5):
            input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
            output3 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
            awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, output3)
        with timer('AWQ W8A8 ' , n = n, recordList=recordList[2]):
            for _ in range(n):
                input_int8 = torch.empty_like(input, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.quant_fp16_to_int8(input_int8, input, scale_x)
                output3 = torch.empty(input.shape[0], weight.shape[0], device='cuda', dtype=torch.float16)
                awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, output3)
                # edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, output3)

        print(default_color, end='')

        for _ in range(5):
            refout = input @ weight.T
        with timer('Pyto Fp16', n = n, recordList=recordList[4]):
            for _ in range(n):
                refout = input @ weight.T
        
        aver = refout.abs().mean()
        loss = (refout-output3).abs().mean()
        print('loss3:', loss.item()/aver.item())
        loss = (refout-output4).abs().mean()
        print('loss4:', loss.item()/aver.item())
        print('---------------------')
    plt.figure()
    plt.plot(recordList[2], color = 'r')
    plt.plot(recordList[3], color = 'g')
    plt.plot(recordList[4], color = 'k')
    plt.grid(True)

# w_shape = [(3584, 512), (3584, 3584)]
w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
for s in w_shape:
    test(s[0], s[1])
plt.show()