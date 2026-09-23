import torch
import numpy as np
import matplotlib.pyplot as plt
from kairos.quantization import *
from utils.command_parser import parser
from utils.perf_eval import timer
from utils.color_print import *

from thirdparty.SmoothQuant.w8a8_linear import W8A8Linear

from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin

from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ


torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

# input_len = [1, 4, 8, 1024*16, 1024*32, 1024*64]
# input_len_s = ['1', '4', '8', '16k', '32k', '64k']

input_len = [1, 4, 8, 128, 256, 512, 1024*16, 1024*32, 1024*64]
input_len_s = ['1', '4', '8', '128', '256', '512', '16k', '32k', '64k']

kernel_name_list = ['Pytorch   FP16','Qserve    W4A8','QQQ       W4A8', 'AWQ      W4A16', 'Marlin   W4A16', 'SmoothQ   W8A8']
    

args = parser.parse_args()
model_type = args.model_type
if model_type == 'qwen':
    save_path = 'qwen.pt'
    prof_path = './prof_qwen/'
    shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)] # Qwen2.5 7B
elif model_type == 'llama':
    save_path = 'llama.pt'
    prof_path = './prof_llama/'
    shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
elif model_type == 'conformer':
    save_path = 'conformer.pt'
    prof_path = './prof_conformer/'
    base_m = 50
    shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256), (5000, 256)] 
elif model_type == 'vit':
    save_path = 'vit.pt'
    prof_path = './prof_conformer/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
else: assert 0, print('Unknown model:', model_type)

record = [[] for i in range(len(kernel_name_list))]

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(proj_name, K, N, bias):
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    qlinear_w4a8_qserve  = W4A8Linear.from_module(torchLinear, 'cuda')
    qlinear_w4a8_qqq     = W4A8Linear_QQQ.from_module(torchLinear)
    qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    qlinear_w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    qlinear_w8a8_awq     = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')

    print_flag = True
    linear_list = [torchLinear, qlinear_w4a8_qserve, qlinear_w4a8_qqq, qlinear_w4a16_awq, qlinear_w4a16_marlin, qlinear_w8a8_awq]

    # Preheat Kernels
    preheat_time = 50
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for linear in linear_list:
            for _ in range(preheat_time):
                out = linear(input_tensor)
            
    n = 100
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'N: {N} K: {K} Bias:{bias}')
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for i in range(len(linear_list)):
            torch.cuda.empty_cache()
            with timer(kernel_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
                for _ in range(n):
                    out = linear_list[i](input_tensor)
    for i in range(len(kernel_name_list)):
        record[i].append(recordList[i])
    return

if __name__ == '__main__':
    if 1:
        # w_shape_proj = ['k_proj, v_proj', 'q_proj',
        #             'o_proj', 'gate_proj, up_proj', 'down_proj']
        # w_proj_count = [2, 1, 1, 2, 1]
        # w_shape = shape_list
        # w_shape.insert(1, w_shape[1])

        w_shape_proj = ['k_proj']
        w_proj_count = [1]
        w_shape = [(4096, 14336)]
        # shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B

        bias_list = [True, True, False, False, False]
    
        for i in range(len(w_shape)):
            print('-'*20)
            print(w_shape_proj[i])
            s = w_shape[i]
            test(w_shape_proj[i], s[1], s[0], bias_list[i])
            print('#' * 30)

        sum_lat_list = []
        for i in range(len(kernel_name_list)):
            sum_lat = []
            for j in range(len(record[i][0])):
                sl = 0
                for k in range(len(w_shape_proj)):
                    sl += record[i][k][j] * w_proj_count[k]
                sum_lat.append(sl)
            sum_lat_list.append(sum_lat)
        
        torch.save(sum_lat_list, save_path)
    else:
        sum_lat_list = torch.load(save_path)
    
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
        print('=' * 30)
    print('=' * 30)

    plt.figure(figsize=(5.7,5))
    plt.xlabel('Input dimension M')
    plt.ylabel('Speedup')
    def div_list(a, b):
        for idx in range(len(a)):
            a[idx] = b[idx] / a[idx]
    div_list(sum_lat_list[1], sum_lat_list[0]),
    div_list(sum_lat_list[2], sum_lat_list[0]),
    div_list(sum_lat_list[3], sum_lat_list[0]),
    div_list(sum_lat_list[4], sum_lat_list[0]),
    div_list(sum_lat_list[5], sum_lat_list[0]),
    # plt.title(f'LLM linear')
    font = {'style': 'normal', 'weight': 'bold', 'size': 18}
    if 0:   # Figure(b)
        # plt.plot(input_len, sum_lat_list[2], 's-c', label = 'W4A8_QQQ')
        plt.plot(input_len, sum_lat_list[3], '--ob', label = 'W4A16 by AWQ')
        plt.plot(input_len, sum_lat_list[1], '--oc', label = 'W4A8 by Qserve')
        plt.plot(input_len, [i + 0.0759 for i in sum_lat_list[0]], color = 'y', label = 'DQ + FP16')
        # plt.plot(input_len, sum_lat_list[4], 's-b', label = 'W4A16_Marlin')
        plt.plot(input_len, [i + 0.0310 for i in sum_lat_list[5]], '--ok', label = 'DQ + W8A8')
        plt.legend(prop=font)
        plt.show()
        exit(0)

    if 0:   # Figure(d)
        sub_w4a16_w8a8 = []
        deq_w4_w8 = []
        for i in range(len(sum_lat_list[0])):
            sub_w4a16_w8a8.append(sum_lat_list[3][i] - sum_lat_list[5][i])
            deq_w4_w8.append(0.0310)
        plt.plot(input_len, sub_w4a16_w8a8, color = 'r', label = 'W4A16 - W8A8')
        plt.plot(input_len, deq_w4_w8, color = 'b', label = 'DQ(W4->W8)')
        plt.legend()
        plt.show()
        exit(0)

    categories = input_len
    width = 0.12
    x = np.arange(len(categories))

    # plt.plot(input_len, sum_lat_list[0], color = '#EADB52', label = 'FP16 by cuBLAS')
    x = [i for i in range(len(input_len))]
    plt.plot(x, sum_lat_list[5], '--ok', linewidth=2, label = 'SmoothQuant W8A8')
    plt.plot(x, sum_lat_list[3], '--ob', linewidth=2, label = 'AWQ W4A16')
    plt.plot(x, sum_lat_list[4], 's-b', linewidth=2, label =  'MARLIN W4A16')
    plt.plot(x, sum_lat_list[1], '--oc', linewidth=2, label = 'Qserve W4A8')
    plt.plot(x, sum_lat_list[2], 's-c', linewidth=2, label =  'QQQ W4A8')

    plt.xticks(x, input_len_s)
    plt.legend()
    # plt.ylim(0.6, 2.6)
    # plt.savefig(f'./{model_type}_linear.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    # exit(0)

    exit(0)