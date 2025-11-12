import torch
import numpy
import torch.cuda.nvtx as nvtx
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from edq.dynamic_linear import edq_quantize_tensor_int4

sub_stream = torch.cuda.Stream(device='cuda:0', priority=0)
sub_stream_ptr = sub_stream.cuda_stream

# pri_stream = torch.cuda.Stream(device='cuda:0', priority=-1)
# pri_stream_ptr = pri_stream.cuda_stream

cur_stream = torch.cuda.current_stream()
cur_stream_ptr = cur_stream.cuda_stream

# torch.cuda.set_stream(pri_stream)

event_1 = torch.cuda.Event(enable_timing=False)
event_2 = torch.cuda.Event(enable_timing=False)

torch.manual_seed(seed=10)

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

def d_on_q(K, N):
    print(f'K: {K} N: {N}')
    m = 128
    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')

    group_size = min(128, K)
    weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

    qweight, scales, scaled_zeros = quant_weight_awq(weight_tensor.clone(), group_size)
    awq_args = (qweight, scales, scaled_zeros, N, K, group_size)

    weight_i8, q_config1 = quantize_tensor_int8(weight_tensor)
    scale_1 = q_config1['scale']

    weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=group_size)
    scale_2_f = q_config2['scale_f']
    scale_2_i = q_config2['scale_i']
    zero_2_f = q_config2['zero_pt_f']
    zero_2_i = q_config2['zero_pt_i']
    weight_int4 = pack_int4_data(weight_i4)

    # TEST DEQUANT KERNEL
    w_i8 = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
    edq_cuda_accel.dequant_int4_to_int8(w_i8, weight_int4, scale_2_i, zero_2_i, group_size)
    
    scale_f = scale_2_f.view(scale_1.shape[0], -1) * scale_1
    w_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)

    preheat_time = 20
    for _ in range(preheat_time):
        edq_cuda_accel.dequant_int4_to_fp16_stream(w_f16, weight_int4, 
                                            scale_f, zero_2_f, group_size, sub_stream_ptr)
        awq_output = awq_forward(input_tensor, *awq_args)
        input_tensor.add_(1)
        input_tensor.sub_(1)
        sub_stream.synchronize()
    
    torch.cuda.synchronize()
    nvtx.range_push(f"Test: ({m},{K},{N})")
    with timer('D on Q '):
        # awq_output = awq_forward(input_tensor, *awq_args)
        edq_cuda_accel.dequant_int4_to_fp16_stream(w_f16, weight_int4, 
                                            scale_f, zero_2_f, group_size, sub_stream_ptr)
        sub_stream.record_event(event_2)
        # input_tensor.add_(10)
        # input_tensor.sub_(10)
        awq_output = awq_forward(input_tensor, *awq_args)
        # input_tensor.add_(10)
        # input_tensor.sub_(10)
        # awq_output = awq_forward(input_tensor, *awq_args)

        # cur_stream.wait_event(event_2)
    nvtx.range_pop()

    with timer('Dequant'):
        edq_cuda_accel.dequant_int4_to_fp16_stream(w_f16, weight_int4, 
                                            scale_f, zero_2_f, group_size, cur_stream_ptr)
    
    with timer('QGEMM  '):
        awq_output = awq_forward(input_tensor, *awq_args)
        # input_tensor.add_(10)
        # input_tensor.sub_(10)
        # awq_output = awq_forward(input_tensor, *awq_args)
        # input_tensor.add_(10)
        # input_tensor.sub_(10)
        # awq_output = awq_forward(input_tensor, *awq_args)

    
if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(3584, 18944), (18944, 3584)]
    for s in w_shape:
        d_on_q(s[0], s[1])
    #     test(s[0], s[1])
    # plt.show()