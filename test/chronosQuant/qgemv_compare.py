import torch
import numpy
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
from thirdparty.Qserve.w4a8_linear import W4A8Linear
# from bitblas.cache import global_operator_cache, get_database_path

# BITBLAS_DATABASE_PATH = get_database_path()
# BITBLAS_TARGET = 'nvidia/nvidia-a100'
# global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
# bitblas.set_log_level("Debug")

torch.manual_seed(seed=10)

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

    # input_len = [i for i in range(1, 32, 1)]
    input_len = [1]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024]

    # Preheat Kernels
    preheat_time = 20
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        weight_tensor_t = weight_tensor.T
        for _ in range(preheat_time):
            ref_output = input_tensor @ weight_tensor_t
        for _ in range(preheat_time):
            input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
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
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
            smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
            awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
        for _ in range(preheat_time):
            input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
            scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
            edq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, edq_output)
        for _ in range(preheat_time):
            edq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.w8a16_ws_gemm_cuda(input_tensor, weight_int8, scale_w, edq_output)

    n = 100
    print_flag = True
    kernel_name_list = ['Pytorch  FP16','Qserve   W4A8', 'EDQ     W4A16', 'AWQ     W4A16', 'SmoothQ  W8A8', 'EDQ      W8A8', 'EDQ     W8A16']
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N}')
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        weight_tensor_t = weight_tensor.T
        with timer('Pytorch  FP16', n = n, recordList=recordList[0], print_flag=print_flag):
            for _ in range(n):
                ref_output = input_tensor @ weight_tensor_t

        with timer('Qserve   W4A8', n = n, recordList=recordList[1], print_flag=print_flag):
            for _ in range(n):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                    input_int8, *qserve_args, scale_x, qse_output,
                )
        check_output(ref_output, qse_output)

        # with timer('EDQ     W4A16', n = n, recordList=recordList[2], print_flag=print_flag):
        #     for _ in range(n):
        #         model_output = bitblas_w4a16(input_tensor, weight_int4_bitblas, **bitblas_args)
        # check_output(ref_output, model_output)
        recordList[2].append(1e10)

        with timer('AWQ     W4A16', n = n, recordList=recordList[3], print_flag=print_flag):
            for _ in range(n):
                awq_output = awq_forward(input_tensor, *awq_args)
        check_output(ref_output, awq_output)

        with timer('SmoothQ  W8A8', n = n, recordList=recordList[4], print_flag=print_flag):
            for _ in range(n):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                smq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8, scale_w, scale_x, smq_output)
        check_output(ref_output, smq_output)

        with timer('EDQ      W8A8', n = n, recordList=recordList[5], print_flag=print_flag):
            for _ in range(n):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                edq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8, scale_x, scale_w, edq_output)
        check_output(ref_output, edq_output)

        with timer('EDQ     W8A16', n = n, recordList=recordList[6], print_flag=print_flag):
            for _ in range(n):
                edq_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                edq_cuda_accel.w8a16_ws_gemm_cuda(input_tensor, weight_int8, scale_w, edq_output)
        check_output(ref_output, edq_output)

        if print_flag:
            # fastest_kid = numpy.argmin(recordList)
            # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
            # print(gre_prefix, end='')
            # print(default_color, end='')
            print('-' * 30)
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
        plt.plot(input_len, recordList[4], color = 'y')
        plt.plot(input_len, recordList[5], color = 'k')
    def showByQuantConfig():
        plt.title(f'K:{K}, N:{N}')
        w4a16_lat = min(recordList[2], recordList[3])
        w8a8_lat = min(recordList[4], recordList[5])
        plt.plot(input_len, recordList[0], color = 'r', label = 'fp16')
        plt.plot(input_len, recordList[1], color = 'g', label = 'w4a8')
        plt.plot(input_len, w4a16_lat, color = 'y', label = 'w4a16')
        plt.plot(input_len, w8a8_lat, color = 'k', label = 'w8a8')
        plt.legend()
    showByQuantConfig()
    plt.xlabel('m')
    plt.ylabel('latency')
    # plt.xticks(input_len)
    plt.grid(True)
    
if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(3584, 18944), (18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])
    plt.show()