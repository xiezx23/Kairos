# python -m test.chronosQuant.qgemm_compare

# import bitblas
import torch
import numpy
import matplotlib.pyplot as plt
import kairos_cuda_accel
import awq_backend
import omniserve_backend
from kairos.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq
from thirdparty.Qserve.w4a8_linear import W4A8Linear
# from bitblas.cache import global_operator_cache, get_database_path

# BITBLAS_DATABASE_PATH = get_database_path()
# BITBLAS_TARGET = 'nvidia/nvidia-a100'
# global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
# bitblas.set_log_level("Debug")

torch.manual_seed(seed=10)

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})


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

# def bitblasMatmulW4A16(K, N, group_size):
#     matmul_config = bitblas.MatmulConfig(
#         M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],  # M dimension
#         # M=[4],  # M dimension
#         N=N,    # N dimension
#         K=K,    # K dimension
#         A_dtype="float16",      # activation A dtype
#         W_dtype="uint4",        # weight W dtype
#         accum_dtype="float32",  # accumulation dtype
#         out_dtype="float16",    # output dtype
#         layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
#         with_bias=False,  # bias
#         # configs for weight only quantization
#         group_size=group_size,  # setting for grouped quantization
#         with_scaling=True,  # setting for scaling factor
#         with_zeros=True,  # setting for zeros
#         zeros_mode='original',  # setting for how to calculating zeros
#     )
#     bitblas_w4a16 = global_operator_cache.get(matmul_config)
#     if bitblas_w4a16 is None:
#         # should disable tuning for the first time because we may require loading bitblas operator from database.
#         bitblas_w4a16 = bitblas.Matmul(matmul_config, target=BITBLAS_TARGET, enable_tuning=False)
#         if enable_tuning:
#             print('BitBLAS Tuning...')
#             bitblas_w4a16.hardware_aware_finetune(topk=20)
#             global_operator_cache.add(matmul_config, bitblas_w4a16)
#             global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
#             print("BitBLAS Tuning done, appended operator to global_operator_cache.")
#         else:
#             print("BitBLAS Operator created.")
#     # else:
#         # print("BitBLAS Operator found in global_operator_cache.")
#     return bitblas_w4a16


def test(i, K, N):
    group_size = min(128, K)

    # Create input matrices
    weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

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

    # input_len = [i for i in range(16, 256, 16)]
    # input_len = [i for i in range(1, 16, 2)]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024]
    # input_len = [1, 4, 8, 16, 128, 256, 384, 512, 1024, 2048, 4096, 8192]
    # input_len_s = ['1', '4', '8', '16', '128', '256', '384', '512', '1k', '2k', '4k', '8k']
    
    # input_len = [1, 4, 8, 16, 128, 256, 384, 512, 1024, 2048]
    # input_len_s = ['1', '4', '8', '16', '128', '256', '384', '512', '1k', '2k']

    input_len = [1, 2, 4, 8, 16, 64, 128, 256, 1024, 2048, 4096, 8192, 16*1024, 32*1024]
    input_len_s = ['1', '2', '4', '8', '16', '64', '128', '256', '1k', '2k', '4k', '8k', '16k', '32k']

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
        # for _ in range(preheat_time):
        #     model_output = bitblas_w4a16(input_tensor, weight_int4_bitblas, **bitblas_args)
        for _ in range(preheat_time):
            awq_output = awq_forward(input_tensor, *awq_args)
        for _ in range(preheat_time):
            input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
            kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
            smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
            awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)

    n = 200
    print_flag = True
    kernel_name_list = ['Pytorch  FP16', 'Qserve   W4A8', 'BitBLAS W4A16', 'AWQ     W4A16', 'SmoothQ  W8A8', 'EDQ      W8A8']
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N}')
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

        # with timer('BitBLAS W4A16', n = n, recordList=recordList[2], print_flag=print_flag):
        #     for _ in range(n):
        #         model_output = bitblas_w4a16(input_tensor, weight_int4_bitblas, **bitblas_args)
        # check_output(ref_output, model_output)
        recordList[2].append(1e10)
        with timer('AWQ     W4A16', n = n, recordList=recordList[3], print_flag=print_flag):
            for _ in range(n):
                awq_output = awq_forward(input_tensor, *awq_args)
        check_output(ref_output, awq_output)

        input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
        scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
        kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
        with timer('SmoothQ  W8A8', n = n, recordList=recordList[4], print_flag=print_flag):
            for _ in range(n):
                smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
        check_output(ref_output, smq_output)

        # input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
        # scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
        # kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
        # with timer('EDQ      W8A8', n = n, recordList=recordList[5], print_flag=print_flag):
        #     for _ in range(n):
        #         edq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
        #         kairos_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, edq_output)
        # check_output(ref_output, edq_output)

        if print_flag:
            # fastest_kid = numpy.argmin(recordList)
            # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
            # print(gre_prefix, end='')
            # print(default_color, end='')
            print('-' * 30)
        
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
        plt.plot(input_len, recordList[4], color = 'y')
        plt.plot(input_len, recordList[5], color = 'k')
        plt.xlabel('m')
        plt.ylabel('latency')
    def showByQuantConfig():
        plt.title(f'K:{K}, N:{N}')
        x = [i for i in range(len(input_len))]
        plt.plot(x, recordList[0], color = 'r', label = 'fp16')
        plt.plot(x, recordList[1], color = 'g', label = 'w4a8')
        plt.plot(x, recordList[3], color = 'y', label = 'w4a16')
        plt.plot(x, recordList[4], color = 'k', label = 'w8a8')
        plt.xticks(x, input_len)
        plt.legend()
        plt.xlabel('m')
        plt.ylabel('latency')
    def showBestToWorst():
        plt.title(f'Best To Worst of Linear(K:{K}, N:{N})')
        b2w = []
        for i in range(len(recordList[0])):
            best = min(min(recordList[0][i], recordList[1][i]), min(recordList[3][i], recordList[4][i]))
            worst = max(max(recordList[0][i], recordList[1][i]), max(recordList[3][i], recordList[4][i]))
            b2w.append(worst/best)
        print(max(b2w))
        plt.plot(input_len, b2w, color = 'r', label = 'BestToWorst')
        plt.legend()
        plt.xlabel('M')
        plt.ylabel('Speedup')
        
    def showSpeedup():
        def div_list(a, b):
            for idx in range(len(a)):
                a[idx] = b[idx] / a[idx]
        div_list(recordList[1], recordList[0]),
        div_list(recordList[3], recordList[0]),
        div_list(recordList[4], recordList[0]),
        div_list(recordList[0], recordList[0]),
        # plt.title(f'Linear (N:{N}, K:{K})')
        x = [i for i in range(len(input_len))]
        plt.plot(x, recordList[0], '-r' , linewidth=2, label = 'FP16    GEMM from CuBLAS')
        plt.plot(x, recordList[1], '-og', linewidth=2, label = 'W4A8  mpGEMM from Qserve')
        plt.plot(x, recordList[3], '-o', color='#2E75B6' ,linewidth=2, label = 'W4A16 mpGEMM from AWQ')
        plt.plot(x, recordList[4], '-ok', linewidth=2, label = 'W8A8  mpGEMM from SmoothQuant')
        plt.xticks(x, input_len_s)
        plt.xlabel('M')
        plt.ylabel('Speedup over FP16 GEMM')
        # plt.legend()

    # plt.figure()
    plt.subplot(1, 2, i)
    showSpeedup()
    # plt.xticks(input_len)
    plt.grid(True)

    # print(bitblas_w4a16.get_source())
    
if __name__ == '__main__':
    # w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    w_shape = [(4096, 4096), (14336, 4096)] # Llama-3 8B
    # w_shape = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B
    # w_shape = [(3584, 18944), (18944, 3584)]

    plt.figure()
    # plt.xlabel('M')
    # plt.ylabel('Speedup')
    # plt.figure(figsize=(16,8))
    font = {'style': 'normal', 'weight': 'bold', 'size': 18}
    for i in range(len(w_shape)):
        s = w_shape[i]
        test(i+1, s[0], s[1])
    plt.legend(ncol=2, loc='center', fontsize=14, handletextpad=0.3, columnspacing=0.7, handlelength=1.2, bbox_to_anchor=(-0.15, 1.1), frameon=False)
    
    # plt.tight_layout()
    plt.show()