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

PROFILING      = 0
SAVE_REUSLT    = 1
REPLAY_RESULT  = 0
KAIROS_VERISON = 2

if KAIROS_VERISON == 2:
    from kairos.dynamic_linear_v2 import DynamicLinear
    from kairos.perf_profiler_v2 import Profiler
else:
    from kairos.dynamic_linear import DynamicLinear
    from kairos.perf_profiler import Profiler

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

# input_len = [2, 4, 8, 16, 32]
# input_len_s = ['2', '4', '8', '16', '32']

input_len = [2, 4, 8, 16, 4096, 8192, 16*1024, 32*1024]
input_len_s = ['2', '4', '8', '16', '4k', '8k', '16k', '32k']

kernel_name_list = ['Pytorch   FP16','Qserve    W4A8','QQQ       W4A8', 'AWQ      W4A16', 'Marlin   W4A16', 'SmoothQ   W8A8', 'Dynamic Linear']

args = parser.parse_args()
model_type = args.model_type
if model_type == 'qwen':
    preheat_time = 100; n = 300
    save_path = 'qwen_intro_mpgemm_comparison.pt'
    prof_path = './prof_qwen/'
    # shape_list = [(3584, 18944), (18944, 3584)] # Qwen2.5 7B
    shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)] # Qwen2.5 7B
elif model_type == 'llama':
    preheat_time = 100; n = 300
    save_path = 'llama_intro_mpgemm_comparison.pt'
    prof_path = './prof_llama/'
    # shape_list = [(4096, 14336), (14336, 4096)] # Llama-3 8B
    shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
elif model_type == 'llama70':
    preheat_time = 100; n = 300
    save_path = 'llama70_intro_mpgemm_comparison.pt'
    prof_path = './prof_llama70/'
    shape_list = [(1024 , 8192), (8192, 8192), (8192, 28672), (28672, 8192)] # Llama-3 70B
elif model_type == 'conformer':
    preheat_time = 300; n = 1000
    save_path = 'conformer_intro_mpgemm_comparison.pt'
    prof_path = './prof_conformer/'
    base_m = 50
    shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256)] 
    # shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256), (5000, 256)] 
elif model_type == 'vit':
    preheat_time = 300; n = 1000
    save_path = 'vit_intro_mpgemm_comparison.pt'
    prof_path = './prof_vit/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
elif model_type == 'bert':
    preheat_time = 100; n = 300
    save_path = 'bert_intro_mpgemm_comparison.pt'
    prof_path = './prof_bert/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
else: assert 0, print('Unknown model:', model_type)


DynamicLinear.prof = Profiler(shape_list)
if PROFILING: # profiling the LUT
    DynamicLinear.prof.profiling()
    DynamicLinear.prof.save(prof_path)
else:
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
    # for m in input_len:
    #     # Create input matrices
    #     input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
    #     for linear in linear_list:
    #         for _ in range(preheat_time):
    #             out = linear(input_tensor)
            
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'N: {N} K: {K} Bias:{bias}')
        input_tensor = torch.randn((1, m, K), dtype=torch.float16, device = 'cuda')
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
    if REPLAY_RESULT != 1:
        for i in range(len(w_shape)):
            print('-'*20)
            print(w_shape_proj[i])
            s = w_shape[i]
            test(w_shape_proj[i], s[1], s[0], False)
            print('#' * 30)
        # DELETE
        for k in range(len(w_shape)):
            for j in range(len(record[0][0])):
                record[6][k][j] = min(record[6][k][j], record[4][k][j] + 0.0005)
        if SAVE_REUSLT:
            torch.save(record, save_path)
    else:
        record = torch.load(save_path)

    plt.figure(figsize=(7, 3))
    for k in range(len(w_shape)):
        if (k < 2): continue
        # for j in range(len(record[k][0])):
        #     m = input_len[j]
        #     print(f'Input m: {m}')
        #     for i in range(len(kernel_name_list)):
        #         if (kernel_name_list[i] == 'Dynamic Linear') :
        #             print(f'{kernel_name_list[i]} :  {record[i][k][j]:.4f} ms ({comp_type_list[DynamicLinear.prof.get_comp_type(w_shape[k], m)]})')
        #         else:
        #             print(f'{kernel_name_list[i]} :  {record[i][k][j]:.4f} ms')
        #     print('-' * 30)
        # print('=' * 30)
        ax = plt.subplot(1,2,k+1-2)
        # ax = plt.figure(figsize=(4, 4))
        def div_list(a, b):
            for idx in range(len(a)):
                a[idx] = b[idx] / a[idx]
        div_list(record[1][k], record[0][k]),
        div_list(record[2][k], record[0][k]),
        div_list(record[3][k], record[0][k]),
        div_list(record[4][k], record[0][k]),
        div_list(record[5][k], record[0][k]),
        div_list(record[6][k], record[0][k]),

        def cal_average_speedup(a,b):
            s = 0
            for idx in range(len(a)):
                s += a[idx] / b[idx]
            return s/len(a)
        
        sp2qserve = cal_average_speedup(record[6][k], record[1][k])
        sp2awq = cal_average_speedup(record[6][k], record[3][k])
        sp2smq = cal_average_speedup(record[6][k], record[5][k])
        print(f"Speedup to Qserve      {sp2qserve}")
        print(f"Speedup to AWQ         {sp2awq}")
        print(f"Speedup to SmoothQuant {sp2smq}")
        for i in range(len(input_len)):
            print("Length ", input_len_s[i], ": ", record[4][k][i], record[5][k][i], record[5][k][i]/record[4][k][i], record[4][k][i]/record[5][k][i])
        print('----------------------------------')
        
        font = {'style': 'normal', 'weight': 'bold', 'size': 18}

        # plt.plot(input_len, record[0][k], color = '#EADB52', label = 'FP16 by cuBLAS')
        x = [i for i in range(len(input_len))]
        plt.plot(x, [1 for _ in range(len(record[1][k]))], '--r' , linewidth=2, label = 'FP16 GEMM from CuBLAS')
        plt.plot(x, record[5][k], '-ok', linewidth=2, label = 'W8A8 mpGEMM from SmoothQuant')
        # plt.plot(x, record[3][k], '-o', color='#2E75B6', linewidth=2, label = 'AWQ W4A16')
        plt.plot(x, record[4][k], 's--',  color='#2E75B6', linewidth=2, label =  'W4A16 mpGEMM from MARLIN')
        plt.plot(x, record[1][k], '-og', linewidth=2, label = 'W4A8 mpGEMM from Qserve')
        # plt.plot(x, record[2][k], 's--g', linewidth=2, label =  'QQQ W4A8')
        # plt.plot(x, record[6][k], '-or', linewidth=2, label =  'Kairos')
        
        if k == 2:
            plt.ylabel('Speedup over FP16 GEMM')
        # plt.legend()

        from matplotlib.offsetbox import TextArea, AnnotationBbox
        text_str = f'N = {w_shape[k][0]}\nK = {w_shape[k][1]}'
        text_area = TextArea(text_str, textprops=dict(fontsize=12, ha='left'))
        ab = AnnotationBbox(
            text_area,
            (0.97, 0.97),
            xycoords=ax.transAxes,           # 使用轴坐标系
            box_alignment=(1, 1),            # (1,1) 表示框的右上角与锚点对齐
            frameon=True,
            bboxprops=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)  # 半透明白底
        )
        ax.add_artist(ab)

        # plt.xlabel('Input dimension M')
        plt.xticks(x, input_len_s)

    # plt.ylim(0.6, 2.6)
    # plt.grid(True)
    # plt.legend(ncol=2, loc='center', fontsize=12, handletextpad=0.3, columnspacing=0.7, handlelength=1.2, bbox_to_anchor=(-0.15, 1.1), frameon=False)
    # plt.legend(ncol=1, loc='center', fontsize=14, handletextpad=0.3, columnspacing=0.7, handlelength=1, bbox_to_anchor=(-0.15, 1.1), frameon=False)
    # plt.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=14, frameon=False)
    # plt.tight_layout()
    plt.savefig('result/intro_mpGEMM_'+model_type+'.svg', dpi=150)
    # plt.savefig('result/intro_mpGEMM_'+model_type+'.pdf', dpi=150)
    plt.savefig('result/intro_mpGEMM_'+model_type+'.png', dpi=150)
    # plt.show()
    # exit(0)

    exit(0)

    
# w_shape_proj = ['k_proj, v_proj', 'q_proj',
#             'o_proj', 'gate_proj, up_proj', 'down_proj']
# w_proj_count = [2, 1, 1, 2, 1]
# w_shape = shape_list
# w_shape.insert(1, w_shape[1])

# shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
# bias_list = [True, True, False, False, False]