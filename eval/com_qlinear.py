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

from kairos.dynamic_linear_v2 import DynamicLinear
from kairos.perf_profiler_v2 import Profiler

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

# input_len = [1, 4, 8, 1024*16, 1024*32, 1024*64]
# input_len_s = ['1', '4', '8', '16k', '32k', '64k']

input_len = [1, 2, 4, 8, 16, 4096, 8192, 16*1024, 32*1024]
input_len_s = ['1', '2', '4', '8', '16', '4k', '8k', '16k', '32k']

kernel_name_list = ['Pytorch   FP16','Qserve    W4A8','QQQ       W4A8', 'AWQ      W4A16', 'Marlin   W4A16', 'SmoothQ   W8A8', 'Dynamic Linear']

args = parser.parse_args()
model_type = args.model_type
if model_type == 'qwen':
    save_path = 'qwen_com_qlinear.pt'
    prof_path = './prof_qwen/'
    shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)] # Qwen2.5 7B
elif model_type == 'llama':
    save_path = 'llama_com_qlinear.pt'
    prof_path = './prof_llama/'
    shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
elif model_type == 'llama70':
    save_path = 'llama70_com_qlinear.pt'
    prof_path = './prof_llama70/'
    shape_list = [(1024 , 8192), (8192, 8192), (8192, 28672), (28672, 8192)] # Llama-3 70B
elif model_type == 'conformer':
    save_path = 'conformer_com_qlinear.pt'
    prof_path = './prof_conformer/'
    base_m = 50
    shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256)] 
    # shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256), (5000, 256)] 
elif model_type == 'vit':
    save_path = 'vit_com_qlinear.pt'
    prof_path = './prof_vit/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
elif model_type == 'bert':
    save_path = 'bert_com_qlinear.pt'
    prof_path = './prof_bert/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
else: assert 0, print('Unknown model:', model_type)

DynamicLinear.prof = Profiler(shape_list)
# DynamicLinear.prof.profiling()
# DynamicLinear.prof.save(prof_path)
DynamicLinear.prof.load(prof_path)
comp_type_list = ['W4A16 GEMM', 'W4A16 GEMV', 'W8A8 GEMM', 'w8a16_gemm', 'w8a16_gemv', 'FP16 GEMM']

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
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')

    print_flag = True
    linear_list = [torchLinear, qlinear_w4a8_qserve, qlinear_w4a8_qqq, qlinear_w4a16_awq, qlinear_w4a16_marlin, qlinear_w8a8_awq, dlinear]

    # Preheat Kernels
    preheat_time = 100
    # for m in input_len:
    #     # Create input matrices
    #     input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
    #     for linear in linear_list:
    #         for _ in range(preheat_time):
    #             out = linear(input_tensor)
            
    n = 300
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            comp_type = DynamicLinear.prof.get_comp_type((N,K), m)
            print(f'N: {N} K: {K} Bias:{bias} Use:{comp_type}')
        input_tensor = torch.randn((1,m, K), dtype=torch.float16, device = 'cuda')
        for i in range(len(linear_list)):
            torch.cuda.empty_cache()
            # Preheat Kernels
            for _ in range(preheat_time):
                out = linear_list[i](input_tensor)
            with timer(kernel_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
                for _ in range(n):
                    out = linear_list[i](input_tensor)
    for i in range(len(kernel_name_list)):
        record[i].append(recordList[i])
    return

if __name__ == '__main__':
    w_shape_proj = ['k_proj, v_proj', 'q_proj',
                'o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    w_shape = shape_list
    if 0:
        for i in range(len(w_shape)):
            print('-'*20)
            print(w_shape_proj[i])
            s = w_shape[i]
            test(w_shape_proj[i], s[1], s[0], False)
            print('#' * 30)
        torch.save(record, save_path)
    else:
        record = torch.load(save_path)

    sum_lat = []
    for i in range(len(kernel_name_list)):
        sum_latency = []
        for j in range(len(record[0][0])):
            sum_latency.append(record[i][0][j] * 2 + record[i][1][j]* 2 + record[i][2][j] * 2 + record[i][3][j])
        sum_lat.append(sum_latency)
    
    def cal_average_speedup(a,b):
        s = 0
        for idx in range(len(a)):
            s += a[idx] / b[idx]
        return s/len(a)
    sp2qserve = cal_average_speedup(sum_lat[1][1:5], sum_lat[6][1:5])
    sp2awq = cal_average_speedup(sum_lat[3][1:5], sum_lat[6][1:5])
    sp2marlin = cal_average_speedup(sum_lat[4][1:5], sum_lat[6][1:5])
    sp2smq = cal_average_speedup(sum_lat[5][1:5], sum_lat[6][1:5])
    print(f"Speedup to qserve {sp2qserve}")
    print(f"Speedup to awq {sp2awq}")
    print(f"Speedup to marlin {sp2marlin}")
    print(f"Speedup to smq {sp2smq}")
    print('--------------------------')
    sp2qserve = cal_average_speedup(sum_lat[1][5:9], sum_lat[6][5:9])
    sp2awq = cal_average_speedup(sum_lat[3][5:9], sum_lat[6][5:9])
    sp2marlin = cal_average_speedup(sum_lat[4][5:9], sum_lat[6][5:9])
    sp2smq = cal_average_speedup(sum_lat[5][5:9], sum_lat[6][5:9])
    print(f"Speedup to qserve {sp2qserve}")
    print(f"Speedup to awq {sp2awq}")
    print(f"Speedup to marlin {sp2marlin}")
    print(f"Speedup to smq {sp2smq}")
    print('--------------------------')
    
    plt.figure(figsize=(6, 2))
    font = {'style': 'normal', 'weight': 'bold', 'size': 18}
    x = [i for i in range(len(input_len))][0:4]

    p = np.arange(len(x))
    width = 0.17

    plt.subplot(1,2,1)
    plt.bar(p - 2.5*width, sum_lat[4][1:5], width, label='W4A16 (MARLIN)', color='#00ccff')
    plt.bar(p - 1.5*width, sum_lat[3][1:5], width, label='W4A16 (AWQ)', color='#2c7bb6')
    plt.bar(p - 0.5*width, sum_lat[1][1:5],  width, label='W4A8 (Qserve)',  color='#fdae61')
    plt.bar(p + 0.5*width, sum_lat[5][1:5],  width, label='W8A8 (SmoothQuant)',  color='#abd9e9')
    plt.bar(p + 1.5*width, sum_lat[6][1:5],       width, label='Kairos',         color='#d7191c')

    plt.ylabel('Latency', fontsize=12)
    # plt.legend(ncol=4, loc='upper left', fontsize=12, handletextpad=0.3, columnspacing=0.7, handlelength=1, bbox_to_anchor=(0.15, 1.15), frameon=False)
    plt.xlabel('Input dimension M', fontsize=12)
    plt.xticks(x, input_len_s[1:5])
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1,2,2)
    plt.bar(p - 2.5*width, sum_lat[4][5:9], width, label='W4A16 (MARLIN)', color='#00ccff')
    plt.bar(p - 1.5*width, sum_lat[3][5:9], width, label='W4A16 (AWQ)', color='#2c7bb6')
    plt.bar(p - 0.5*width, sum_lat[1][5:9],  width, label='W4A8 (Qserve)',  color='#fdae61')
    plt.bar(p + 0.5*width, sum_lat[5][5:9],  width, label='W8A8 (SmoothQuant)',  color='#abd9e9')
    plt.bar(p + 1.5*width, sum_lat[6][5:9],       width, label='Kairos',         color='#d7191c')

    plt.xlabel('Input dimension M', fontsize=12)
    plt.xticks(x, input_len_s[5:9])
    


    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig_compound_gemm.pdf', dpi=150)
    # plt.savefig('tmp.png', dpi=150)
    plt.savefig('fig_compound_gemm_'+model_type+'.png', dpi=150)
    # plt.show()
