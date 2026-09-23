import torch
from utils.command_parser import parser
import matplotlib.pyplot as plt
import kairos_cuda_accel
import awq_backend
import omniserve_backend
from kairos.quantization import *
from utils.perf_eval import timer
from utils.color_print import *

from thirdparty.SmoothQuant.w8a8_linear import W8A8Linear

from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin

from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ

from thirdparty.AWQ.awq_method import quant_weight_awq
from thirdparty.Qserve.w4a8_linear import W4A8Linear

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
elif model_type == 'llama70':
    save_path = 'llama70.pt'
    prof_path = './prof_llama70/'
    shape_list = [(1024 , 8192), (8192, 8192), (8192, 28672), (28672, 8192)] # Llama-3 70B
elif model_type == 'conformer':
    save_path = 'conformer.pt'
    prof_path = './prof_conformer/'
    base_m = 50
    shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256)] 
    # shape_list = [(2048, 256), (256, 2048), (256, 256), (512, 256), (5000, 256)] 
elif model_type == 'vit':
    save_path = 'vit.pt'
    prof_path = './prof_vit/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
elif model_type == 'bert':
    save_path = 'bert.pt'
    prof_path = './prof_bert/'
    base_m = 197
    shape_list = [(768, 768), (2304, 768), (3072, 768), (768, 3072)] 
else: assert 0, print('Unknown model:', model_type)

from kairos.dynamic_linear import DynamicLinear
from kairos.perf_profiler import Profiler

DynamicLinear.prof = Profiler(shape_list)
# DynamicLinear.prof.profiling()
# DynamicLinear.prof.save(prof_path)
DynamicLinear.prof.load(prof_path)
comp_type_list = ['W4A16 GEMM', 'W4A16 GEMV', 'W8A8 GEMM', 'w8a16_gemm', 'w8a16_gemv', 'FP16 GEMM']

torch.manual_seed(seed=10)

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})



input_len = [i for i in range(1, 8192, 256)]
# input_len = [i for i in range(1, 16, 2)]
# input_len = [1, 16, 32, 64, 128, 256, 512, 1024]
# input_len = [1, 4, 8, 16, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
# input_len_s = ['1', '4', '8', '16', '128', '256', '384', '512', '1k', '2k', '4k', '8k']

# input_len = [1, 4, 8, 16, 128, 256, 384, 512, 1024, 2048]
# input_len_s = ['1', '4', '8', '16', '128', '256', '384', '512', '1k', '2k']

# input_len = [1, 2, 4, 8, 16, 64, 128, 256, 1024, 2048, 4096, 8192, 16*1024, 32*1024]
# input_len_s = ['1', '2', '4', '8', '16', '64', '128', '256', '1k', '2k', '4k', '8k', '16k', '32k']

enable_tuning=True
MSELoss = torch.nn.MSELoss()

def awq_forward(input, qweight, scales, scaled_zeros, N, K, group_size):
    batch_token_len = input.numel() // input.shape[-1]
    if batch_token_len < 8:
        out = awq_backend.gemv_forward_cuda_new(
            input, qweight, scales, scaled_zeros,
            batch_token_len, N, K, group_size)
    else:
        out = awq_backend.gemm_forward_cuda_new(
            input, qweight, scales, scaled_zeros)
    return out

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(i, K, N):
    if 0:
        group_size = min(128, K)

        # Create input matrices
        weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()
        
        torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=False, dtype=torch.float16).cuda()
        qlinear_w4a8_qserve  = W4A8Linear.from_module(torchLinear, 'cuda')
        qlinear_w4a16_awq    = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
        qlinear_w8a8_awq     = W8A8Linear.from_module(torchLinear, 'cuda', backend='awq')
        dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')

        # Quantize weight tensor from edq
        weight_int4_edq, config_int4_edq = quantize_tensor_int4(weight_tensor, group_size=group_size)
        weight_int8, config_int8 = quantize_tensor_int8(weight_tensor)
        scale_w = config_int8['scale'].squeeze(-1)
        bitblas_args = {'scale':config_int4_edq['scale'], 'zeros':config_int4_edq['zero_pt']}
        qweight, scales, scaled_zeros = quant_weight_awq(weight_tensor.clone(), group_size)
        awq_args = (qweight, scales, scaled_zeros, N, K, group_size)
        # Prepare Qserve W4A8
        qserve_w4a8 = W4A8Linear.from_weight(
            weight_tensor, None, group_size)
        qserve_args = (qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales,)
        # Prepare Bitblas W4A16
        # bitblas_w4a16 = bitblasMatmulW4A16(K, N, group_size)
        # weight_int4_bitblas = bitblas_w4a16.transform_weight(weight_int4_edq)

        # Preheat Kernels
        preheat_time = 20
        for m in input_len:
            # Create input matrices
            input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
            for _ in range(preheat_time):
                ref_output = input_tensor @ weight_tensor.T
            for _ in range(preheat_time):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                    input_int8, *qserve_args, scale_x, qse_output,
                )
            for _ in range(preheat_time):
                awq_output = awq_forward(input_tensor, *awq_args)
            for _ in range(preheat_time):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
            # for _ in range(preheat_time):
            #     dlinear(input_tensor)

        n = 200
        print_flag = True
        kernel_name_list = ['Pytorch  FP16', 'Qserve   W4A8', 'BitBLAS W4A16', 'AWQ     W4A16', 'SmoothQ  W8A8', 'EDQ      W8A8']
        recordList = [[] for _ in range(len(kernel_name_list))]
        Conversion = []
        Quantization = []
        mpGEMM = []
        W4A8 = []
        W4A16 = []
        W8A8 = []
        for midx in range(len(input_len)):
            m = input_len[midx]
            kernel = comp_type_list[DynamicLinear.prof.get_comp_type((N,K), m)]
            print(f'M:{m}   {kernel}')

            # Create input matrices
            input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
            with timer('Pytorch  FP16', n = n, recordList=recordList[0], print_flag=print_flag):
                for _ in range(n):
                    ref_output = input_tensor @ weight_tensor.T

            input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
            kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
            with timer('Qserve   W4A8', n = n, recordList=recordList[1], print_flag=print_flag):
                for _ in range(n):
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )
            check_output(ref_output, qse_output)

            with timer('AWQ     W4A16', n = n, recordList=recordList[2], print_flag=print_flag):
                for _ in range(n):
                    awq_output = awq_forward(input_tensor, *awq_args)
            check_output(ref_output, awq_output)

            with timer('Q(Activation)', n = n, recordList=recordList[4], print_flag=print_flag):
                for _ in range(n):
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)

            with timer('SmoothQ  W8A8', n = n, recordList=recordList[3], print_flag=print_flag):
                for _ in range(n):
                    smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
            check_output(ref_output, smq_output)

            with timer('Converter  W4', n = n, recordList=recordList[5], print_flag=print_flag):
                for _ in range(n):
                    """Converter: W4 -> W8, A16 -> A8"""
                    tmp_weight = torch.empty(dlinear.shape, dtype=torch.int8, device='cuda')
                    kairos_cuda_accel.dequant_interleaved_int4_to_int8(tmp_weight, dlinear.qweight, 
                                                dlinear.scales_to8, dlinear.scaled_zeros_to8, 128)
            
            with timer('Qlinear W4A16', n = n, recordList=W4A16, print_flag=print_flag):
                for _ in range(n):
                    qlinear_w4a16_awq(input_tensor)
                    
            with timer('Qlinear W4A8 ', n = n, recordList=W4A8, print_flag=print_flag):
                for _ in range(n):
                    qlinear_w4a8_qserve(input_tensor)

            with timer('Qlinear W8A8 ', n = n, recordList=W8A8, print_flag=print_flag):
                for _ in range(n):
                    qlinear_w8a8_awq(input_tensor)


            if 'W4A16' in kernel:
                mpGEMM.append(recordList[2][-1])
                Conversion.append(0)
                Quantization.append(0)
            elif kernel == 'W8A8 GEMM':
                mpGEMM.append(recordList[3][-1])
                Quantization.append(recordList[4][-1])
                Conversion.append(recordList[5][-1])

            if print_flag:
                # fastest_kid = numpy.argmin(recordList)
                # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
                # print(gre_prefix, end='')
                # print(default_color, end='')
                print('-' * 30)
        record = [mpGEMM, Quantization, Conversion, W4A16, W4A8, W8A8]
        torch.save(record, save_path)
    else:
        record = torch.load(save_path)
        
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

    
    # plt.figure()
    # plt.subplot(1, 2, i)
    # showSpeedup()
    # plt.xticks(input_len)

    for i in range(len(input_len)):
        m = input_len[i]
        print(f'M:{m}')
        kairos_delay = record[0][i]+record[1][i]+record[2][i]
        print(f'Speedup to W4A8  {record[4][i]/(kairos_delay)}')
        print(f'Speedup to W4A16 {record[3][i]/(kairos_delay)}')
        print(f'Ratio of Conversion {record[2][i]/kairos_delay}')
        print(f'Ratio of Decision   {0.0009/kairos_delay}')
        print(f'Ratio of Quantizati {record[1][i]/kairos_delay}')
        print('--------------------------------')
    
    Decision = [0.0009 for _ in range(len(input_len))]

    def showStack():
        plt.plot(input_len, record[4], '-og', linewidth=2, label = 'W4A8  mpGEMM from Qserve')
        plt.plot(input_len, record[3], '-o', color='#2E75B6' ,linewidth=2, label = 'W4A16 mpGEMM from AWQ')
        # plt.plot(input_len, record[5], '-ok', linewidth=2, label = 'W8A8  mpGEMM from SmoothQuant')
        plt.stackplot(input_len, record[0], record[1], record[2], Decision, 
                      labels=["mpGEMM","Actvation Quantization","Weight Conversion", "Decision-making"], 
                      colors=['#abd9e9','#2c7bb6', '#d7191c', "#e6ab02"])
        plt.legend()

    plt.figure()
    showStack()
    plt.xlabel('M')
    plt.ylabel('latency')
    # plt.xticks(input_len)
    plt.grid(True)

    # print(bitblas_w4a16.get_source())
    
if __name__ == '__main__':
    # n = 200
    # with timer('Decision', n = n, print_flag=True):
    #     for _ in range(n):
    #         kernel = comp_type_list[DynamicLinear.prof.get_comp_type((4096,14336), n)]
    # exit(0)

    # w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    w_shape = [(14336, 4096)] # Llama-3 8B
    # w_shape = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    # w_shape = [(3584, 18944), (18944, 3584)]

    # plt.xlabel('M')
    # plt.ylabel('Speedup')
    # plt.figure(figsize=(16,8))
    font = {'style': 'normal', 'weight': 'bold', 'size': 18}
    for i in range(len(w_shape)):
        s = w_shape[i]
        test(i+1, s[0], s[1])
    # plt.legend(ncol=2, loc='center', fontsize=14, handletextpad=0.3, columnspacing=0.7, handlelength=1.2, bbox_to_anchor=(-0.15, 1.1), frameon=False)
    
    # plt.tight_layout()
    plt.show()