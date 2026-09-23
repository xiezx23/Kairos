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

def test(K, N):
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

    # input_len = [i for i in range(1, 1024*32, 512)]
    input_len = [i for i in range(1, 32*1024, 1024)]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024]

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
        # with timer('Pytorch  FP16', n = n, recordList=recordList[0], print_flag=print_flag):
        #     for _ in range(n):
        #         ref_output = input_tensor @ weight_tensor.T

        input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
        scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
        kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
        with timer('Qserve   W4A8', n = n, recordList=recordList[1], print_flag=print_flag):
            for _ in range(n):
                qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                    input_int8, *qserve_args, scale_x, qse_output,
                )
        # check_output(ref_output, qse_output)

        input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
        scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
        kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)

        with timer('SmoothQ  W8A8', n = n, recordList=recordList[2], print_flag=print_flag):
            for _ in range(n):
                smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
        # check_output(ref_output, smq_output)


        if print_flag:
            # fastest_kid = numpy.argmin(recordList)
            # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
            # print(gre_prefix, end='')
            # print(default_color, end='')
            totaloverhead = recordList[1][-1]
            mainPart = recordList[2][-1]
            print(f'Ratio of Pre-Dequant: {(totaloverhead-mainPart)/totaloverhead:.2f}')
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
    def showByQuantConfig():
        # plt.title(f'K:{K}, N:{N}')
        plt.plot(input_len, recordList[1], color = 'g', label = 'w4a8')
        plt.plot(input_len, recordList[2], color = 'y', label = 'w8a8')
        plt.legend()
    def showStack():
        # plt.title(f'K:{K}, N:{N}')
        pregemm_dq = [recordList[1][i] - recordList[2][i] for i in range(len(recordList[1]))]
        gemm_postdq = recordList[2]
        plt.stackplot(input_len, gemm_postdq, pregemm_dq, labels=["Others","Prologue"], colors=['#abd9e9','#2c7bb6'])
        plt.legend()


    plt.figure()
    showStack()
    plt.xlabel('M')
    plt.ylabel('latency')
    # plt.xticks(input_len)
    # plt.grid(True)

    # print(bitblas_w4a16.get_source())
    
if __name__ == '__main__':
    w_shape = [(4096, 4096)]
    # w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(3584, 18944), (18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])
    plt.show()