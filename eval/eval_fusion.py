
import torch
import numpy
import matplotlib.pyplot as plt
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ
from edq.w8a8_linear import W8A8Linear
from edq.w4a16_linear import W4A16Linear
from edq.dynamic_linear import DynamicLinear, shape_list
from edq.perf_profiler import Profiler

import marlin
import awq_backend
import edq_cuda_accel

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

kernel_name_list = ['Pytorch   FP16', 'AWQ      W4A16', 'Marlin   W4A16', 'DQ+FP16       ']

# input_len = [i for i in range(1, 32, 1)]
# input_len = [128, 512, 1024, 2048, 4096, 1024*8, 1024*16]
# input_len = [1024*8, 1024*16, 1024*32]
# input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024, 64*1024, 96*1024]
# input_len = [1, 4, 8, 16, 32, 64]
# input_len = [1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024, 64*1024, 96*1024]
input_len = [1, 1024, 2048, 4096, 8192]

DynamicLinear.prof = Profiler()
# DynamicLinear.prof.profiling()
# DynamicLinear.prof.save()
DynamicLinear.prof.load()

record = [[] for i in range(len(kernel_name_list))]

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(proj_name, K, N, bias):
    # M = 1024 + 1
    out_features = N; in_features = K; group_size = 128
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    weight = torchLinear.weight.data

    qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    qweight_awq = qlinear_w4a16_awq.qweight
    scales_awq = qlinear_w4a16_awq.scales
    scaled_zeros_awq = qlinear_w4a16_awq.scaled_zeros

    qlinear_w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    B = qlinear_w4a16_marlin.B
    s = qlinear_w4a16_marlin.s
    workspace = qlinear_w4a16_marlin.workspace



    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    qweight = dlinear.qweight
    shape = dlinear.shape
    scales = dlinear.scales
    scaled_zeros = dlinear.scaled_zeros

    print_flag = True
    linear_list = [torchLinear, qlinear_w4a16_awq, qlinear_w4a16_marlin]

    # Preheat Kernels
    preheat_time = 200
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for _ in range(preheat_time):
            out = input_tensor @ weight.T
                
        for _ in range(preheat_time):
            batch_token_len = input_tensor.numel() // input_tensor.shape[-1]
            if batch_token_len < 8:
                out = awq_backend.gemv_forward_cuda_new(
                    input_tensor, qweight_awq, scales_awq, scaled_zeros_awq,
                    batch_token_len, out_features, in_features, group_size)
            else:
                out = awq_backend.gemm_forward_cuda_new(
                    input_tensor, qweight_awq, scales_awq, scaled_zeros_awq)
    
        for _ in range(preheat_time):
            C = torch.empty(input_tensor.shape[:-1] + (s.shape[1],), dtype=torch.float16, device='cuda')
            marlin.mul(
                input_tensor.view((-1, input_tensor.shape[-1])),
                B,
                C.view((-1, C.shape[-1])),
                s,
                workspace, -1, -1, -1, 16
            )

        for _ in range(preheat_time):
            tmp_weight = torch.empty(shape, dtype=torch.float16, device='cuda')
            edq_cuda_accel.dequant_interleaved_int4_to_fp16(tmp_weight, qweight, 
                                        scales, scaled_zeros, 128)
            out = input_tensor @ tmp_weight.T

        for _ in range(preheat_time):
            tmp_weight = torch.empty(shape, dtype=torch.float16, device='cuda')
            edq_cuda_accel.dequant_interleaved_int4_to_fp16(tmp_weight, qweight, 
                                        scales, scaled_zeros, 128)
            
    n = 1000
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'N: {N} K: {K} Bias:{bias}')
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')

        torch.cuda.empty_cache()
        with timer(kernel_name_list[0], n=n, recordList=recordList[0], print_flag=print_flag):
            for _ in range(n):
                out = input_tensor @ weight.T
                    
        torch.cuda.empty_cache()
        with timer(kernel_name_list[1], n=n, recordList=recordList[1], print_flag=print_flag):
            for _ in range(n):
                batch_token_len = input_tensor.numel() // input_tensor.shape[-1]
                if batch_token_len < 8:
                    out = awq_backend.gemv_forward_cuda_new(
                        input_tensor, qweight_awq, scales_awq, scaled_zeros_awq,
                        batch_token_len, out_features, in_features, group_size)
                else:
                    out = awq_backend.gemm_forward_cuda_new(
                        input_tensor, qweight_awq, scales_awq, scaled_zeros_awq)
        
        torch.cuda.empty_cache()
        with timer(kernel_name_list[2], n=n, recordList=recordList[2], print_flag=print_flag):
            for _ in range(n):
                C = torch.empty(input_tensor.shape[:-1] + (s.shape[1],), dtype=torch.float16, device='cuda')
                marlin.mul(
                    input_tensor.view((-1, input_tensor.shape[-1])),
                    B,
                    C.view((-1, C.shape[-1])),
                    s,
                    workspace, -1, -1, -1, 16
                )

        torch.cuda.empty_cache()
        with timer(kernel_name_list[-1], n=n, recordList=recordList[-1], print_flag=print_flag):
            for _ in range(n):
                tmp_weight = torch.empty(shape, dtype=torch.float16, device='cuda')
                edq_cuda_accel.dequant_interleaved_int4_to_fp16(tmp_weight, qweight, 
                                            scales, scaled_zeros, 128)
                out = input_tensor @ tmp_weight.T

        torch.cuda.empty_cache()
        with timer('DQ', n=n,  print_flag=print_flag):
            for _ in range(n):
                tmp_weight = torch.empty(shape, dtype=torch.float16, device='cuda')
                edq_cuda_accel.dequant_interleaved_int4_to_fp16(tmp_weight, qweight, 
                                            scales, scaled_zeros, 128)
        # if print_flag:
        #     currecordList = [recordList[i][-1] for i in range(len(linear_list))]
        #     fastest_kid = numpy.argmin(currecordList)
        #     print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
        #     w4_fastest = min(currecordList[1], currecordList[2], currecordList[3], currecordList[4])
        #     accel_r = (w4_fastest-currecordList[6]) / w4_fastest
        #     print(color_text('gre', f'Accel Ratio to SOTA INT4: {accel_r:.2f}'))
        print('=' * 30)
    for i in range(len(kernel_name_list)):
        record[i].append(recordList[i])
    return

    plt.figure()
    # =============    ===============================
    # character        color
    # =============    ===============================
    # ``'b'``          blue
    # ``'g'``          green
    # ``'r'``          red
    # ``'c'``          cyan
    # ``'m'``          magenta
    # ``'y'``          yellow
    # ``'k'``          black
    # ``'w'``          white
    # =============    ===============================
    plt.title(f'{proj_name}(N:{N} K:{K} Bias:{bias})')
    plt.xlabel('m')
    if 0:
        plt.ylabel('Accelerate Ratio')
        for i in range(len(recordList[0])):
            recordList[1][i] = recordList[0][i] / recordList[1][i]
            recordList[2][i] = recordList[0][i] / recordList[2][i]
            recordList[3][i] = recordList[0][i] / recordList[3][i]
            recordList[4][i] = recordList[0][i] / recordList[4][i]
            recordList[0][i] = 1
    else: plt.ylabel('latency')
    # plt.xticks(input_len)
    plt.grid(True)
    plt.plot(input_len, recordList[0], color = 'y', label = 'FP16')
    plt.plot(input_len, recordList[1], color = 'c', label = 'W4A8')
    plt.plot(input_len, recordList[2], color = 'g', label = 'W4A16')
    plt.plot(input_len, recordList[3], color = 'k', label = 'W8A8')
    plt.plot(input_len, recordList[4], color = 'r', label = 'Dynamic')
    plt.legend()

if __name__ == '__main__':
    w_shape_proj = ['k_proj, v_proj', 'q_proj',
                'o_proj', 'gate_proj, up_proj', 'down_proj']
    bias_list = [True, True, False, False, False]
    # w_shape = [(3584, 512), (3584, 3584),
    #            (3584, 3584), (3584, 18944), (18944, 3584)]
    w_shape = shape_list
    w_shape.insert(1, w_shape[1])
    # w_shape = [(1024, 4096), (4096, 4096), 
    #            (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    # w_shape_proj = ['k_proj, v_proj', 'q_proj, o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    for i in range(len(w_shape)):
        print('-'*20)
        print(w_shape_proj[i])
        s = w_shape[i]
        test(w_shape_proj[i], s[1], s[0], bias_list[i])
        print('#' * 30)
        exit(0)
    exit(0)
    sum_lat_list = []
    for i in range(len(kernel_name_list)):
        sum_lat = []
        for j in range(len(record[i][0])):
            sum_lat.append(record[i][0][j] * 2 + record[i][1][j] + record[i][2][j] + record[i][3][j] * 2 + record[i][4][j])
        sum_lat_list.append(sum_lat)
    
    plt.xlabel('m')
    plt.title(f'LLM linear')
    plt.grid(True)
    plt.plot(input_len, sum_lat_list[0], color = 'y', label = 'FP16')
    plt.plot(input_len, sum_lat_list[1], color = 'b', label = 'W4A8_Qserve')
    plt.plot(input_len, sum_lat_list[2], color = 'c', label = 'W4A8_QQQ')
    plt.plot(input_len, sum_lat_list[3], color = 'g', label = 'W4A16_AWQ')
    plt.plot(input_len, sum_lat_list[4], color = 'm', label = 'W4A16_Marlin')
    plt.plot(input_len, sum_lat_list[5], color = 'k', label = 'W8A8')
    plt.plot(input_len, sum_lat_list[6], color = 'r', label = 'Dynamic')
    plt.legend()
    record_acc_marlin = []
    record_acc_awq = []
    record_acc_qserve = []
    record_acc_fp16 = []
    for i in range(len(input_len)):
        m = input_len[i]
        print(f'Input m: {m}')
        for j in range(len(kernel_name_list)):
            print(f'{kernel_name_list[j]} :  {sum_lat_list[j][i]:.4f} ms')
        currecordList = [sum_lat_list[x][i] for x in range(len(kernel_name_list))]
        fastest_kid = numpy.argmin(currecordList)
        print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
        # w4_fastest = min(currecordList[1], currecordList[2], currecordList[3], currecordList[4])
        # accel_r = (w4_fastest-currecordList[6]) / w4_fastest
        # print(color_text('gre', f'Accel Ratio to SOTA INT4   : {accel_r:.2f}'))
        acc_to_marlin = currecordList[4]/currecordList[6]; record_acc_marlin.append(acc_to_marlin) 
        acc_to_awq =    currecordList[3]/currecordList[6]; record_acc_awq.append(acc_to_awq) 
        acc_to_qserve = currecordList[1]/currecordList[6]; record_acc_qserve.append(acc_to_qserve) 
        acc_to_fp16 =   currecordList[0]/currecordList[6]; record_acc_fp16.append(acc_to_fp16) 
        print(color_text('gre', f'Accel Ratio to Marlin      : {acc_to_marlin:.2f}'))
        print(color_text('gre', f'Accel Ratio to AWQ(W4A16)  : {acc_to_awq:.2f}'))
        print(color_text('gre', f'Accel Ratio to Qserve(W4A8): {acc_to_qserve:.2f}'))
        print(color_text('gre', f'Accel Ratio to Full prec.  : {acc_to_fp16:.2f}'))
        print('=' * 30)
    print('=' * 30)
    print("FINAL REPORT")
    print('=' * 30)
    print(color_text('gre', f'Accel Ratio to Marlin      : {min(record_acc_marlin):.2f} {max(record_acc_marlin):.2f} {sum(record_acc_marlin)/len(record_acc_awq):.2f}'))
    print(color_text('gre', f'Accel Ratio to AWQ(W4A16)  : {min(record_acc_awq):.2f} {max(record_acc_awq):.2f} {sum(record_acc_awq)/len(record_acc_awq):.2f}'))
    print(color_text('gre', f'Accel Ratio to Qserve(W4A8): {min(record_acc_qserve):.2f} {max(record_acc_qserve):.2f} {sum(record_acc_qserve)/len(record_acc_awq):.2f}'))
    print(color_text('gre', f'Accel Ratio to Full prec.  : {min(record_acc_fp16):.2f} {max(record_acc_fp16):.2f} {sum(record_acc_fp16)/len(record_acc_awq):.2f}'))
    print('=' * 30)
    plt.show()