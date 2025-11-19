# python -m test.chronosQuant.qlinear_compare

import torch
import numpy
import matplotlib.pyplot as plt
import kairos_cuda_accel
import awq_backend
import omniserve_backend
from kairos.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from kairos.w8a8_linear import W8A8Linear
from kairos.w4a16_linear import W4A16Linear
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from kairos.dynamic_linear import DynamicLinear

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

sub_stream = torch.cuda.Stream()
enable_tuning=True
MSELoss = torch.nn.MSELoss()

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(K, N, bias):
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    qlinear_w4a8  = W4A8Linear.from_module(torchLinear, 'cuda')
    qlinear_w4a16_awq = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    qlinear_w4a16_kairos = W4A16Linear.from_module(torchLinear, 'cuda')
    qlinear_w8a8_awq  = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')
    qlinear_w8a8_kairos  = W8A8Linear.from_module(torchLinear, 'cuda', backend='kairos')
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    # input_len = [i for i in range(1, 32, 1)]
    # input_len = [128, 512, 1024, 2048, 4096, 1024*8, 1024*16]
    # input_len = [1024*8, 1024*16, 1024*32]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024]
    input_len = [1, 4, 8, 16, 32, 64]

    n = 20
    print_flag = True
    linear_list = [torchLinear, qlinear_w4a8, qlinear_w4a16_awq, qlinear_w4a16_kairos, qlinear_w8a8_awq, qlinear_w8a8_kairos]
    kernel_name_list = ['Pytorch  FP16','Qserve   W4A8', 'AWQ     W4A16', 'kairos     W4A16', 'SmoothQ  W8A8', 'kairos      W8A8']

    DL_type_list =  ['Dlinear  FP16', 'Dlinear W4A16', 'Dlinear  W8A8', 'Dlinear W8A16', 'Dlinear W8A16']
    comp_type_list = ['fp16', 'w4a16', 'w8a8', 'w8a16_gemm', 'w8a16_gemv']

    # Preheat Kernels
    preheat_time = 20
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for linear in linear_list:
            for _ in range(preheat_time):
                out = linear(input_tensor)
        for comp_type in comp_type_list:
            if m > 7 and 'gemv' in comp_type: continue
            for _ in range(preheat_time):
                out = dlinear._forward(input_tensor, comp_type)
            

    recordList = [[] for _ in range(len(kernel_name_list) + len(comp_type_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N} Bias:{bias}')
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for i in range(len(linear_list)):
            torch.cuda.empty_cache()
            with timer(kernel_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
                for _ in range(n):
                    out = linear_list[i](input_tensor)
        # TEST DYNAMIC LINEAR
        print('-' * 30)
        ori_kernel_num = len(kernel_name_list)
        for i in range(len(comp_type_list)):
            comp_type = comp_type_list[i]
            if m > 7 and 'gemv' in comp_type: continue
            torch.cuda.empty_cache()
            with timer(DL_type_list[i], n=n, recordList=recordList[i + ori_kernel_num], print_flag=print_flag):
                for _ in range(n):
                    out = dlinear._forward(input_tensor, comp_type)
        if print_flag:
            # fastest_kid = numpy.argmin(recordList)
            # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
            # print(gre_prefix, end='')
            # print(default_color, end='')
            print('=' * 30)
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
    def showAll():
        plt.plot(input_len, recordList[0], color = 'r')
        plt.plot(input_len, recordList[1], color = 'g')
        plt.plot(input_len, recordList[2], color = 'c')
        plt.plot(input_len, recordList[3], color = 'm')
        # plt.plot(input_len, recordList[4], color = 'y')
        # plt.plot(input_len, recordList[5], color = 'k')
    def showByQuantConfig():
        plt.title(f'K:{K}, N:{N}')
        w4a16_lat = min(recordList[2], recordList[3])
        w8a8_lat = min(recordList[4], recordList[5])
        plt.plot(input_len, recordList[0], color = 'r', label = 'fp16')
        plt.plot(input_len, recordList[1], color = 'g', label = 'w4a8')
        plt.plot(input_len, w4a16_lat, color = 'y', label = 'w4a16')
        plt.plot(input_len, w8a8_lat, color = 'k', label = 'w8a8')
        plt.legend()
    showAll()
    plt.xlabel('m')
    plt.ylabel('latency')
    # plt.xticks(input_len)
    plt.grid(True)
    
if __name__ == '__main__':
    # w_shape_proj = ['k_proj, v_proj', 'q_proj',
    #             'o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(3584, 512, True), (3584, 3584, True),
    #            (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    w_shape_proj = ['k_proj, v_proj', 'q_proj, o_proj', 'gate_proj, up_proj', 'down_proj']
    w_shape = [(3584, 512, False), (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    for i in range(len(w_shape)):
        print('-'*20)
        print(w_shape_proj[i])
        s = w_shape[i]
        test(s[0], s[1], s[2])
        print('#' * 30)
    # plt.show()


    # for M in input_len:
    #     print(f'M: {M} K: {K} N: {N} Bias:{bias}')
    #     n = 5
    #     print_flag = True
    #     linear_list = [torchLinear, qlinear_w4a8, qlinear_w4a16_awq, qlinear_w4a16_kairos, qlinear_w8a8_awq, qlinear_w8a8_kairos]
    #     kernel_name_list = ['Pytorch  FP16','Qserve   W4A8', 'AWQ     W4A16', 'kairos     W4A16', 'SmoothQ  W8A8', 'kairos      W8A8']
    #     input_tensor = torch.randn((M, K), dtype=torch.float16, device = 'cuda')
    #     ref_out = qlinear_w4a16_kairos(input_tensor)
    #     for _ in range(200):
    #         out = qlinear_w4a16_kairos(input_tensor)
    #         torch.cuda.synchronize()
    #         if (ref_out-out).abs().sum() > 0:
    #             print((ref_out-out).abs().sum())
    #             # return
    #             diff = ref_out-out
    #             for i in range(diff.shape[0]):
    #                 for j in range(diff.shape[1]):
    #                     if diff[i][j] != 0:
    #                         print(f'1 diff[{i}][{j}]={diff[i][j]}')
    #                         # return
    #     torch.save(qlinear_w4a16_kairos.cpu().state_dict(),  'marlin.pt')
    #     state_dict = torch.load('marlin.pt', map_location='cuda')
    #     qlinear_w4a16_kairos.load_state_dict(state_dict)
    #     qlinear_w4a16_kairos.to('cuda')
    #     for _ in range(200):
    #         out = qlinear_w4a16_kairos(input_tensor)
    #         torch.cuda.synchronize()
    #         if (ref_out-out).abs().sum() > 0:
    #             print((ref_out-out).abs().sum())
    #             # return
    #             diff = ref_out-out
    #             for i in range(diff.shape[0]):
    #                 for j in range(diff.shape[1]):
    #                     if diff[i][j] != 0:
    #                         print(f'2 diff[{i}][{j}]={diff[i][j]}')
    #                         # return