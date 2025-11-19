import torch
import numpy
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from kairos.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width, pack_int_awq, dequantize_tensor_awq
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ
from kairos.dynamic_linear import edq_quantize_tensor_int4

sub_stream = torch.cuda.Stream(device='cuda:0')
sub_stream_ptr = sub_stream.cuda_stream

cur_stream = torch.cuda.current_stream()
cur_stream_ptr = cur_stream.cuda_stream

torch.manual_seed(seed=10)

@torch.compile()
@torch.no_grad()
def dequant_int4_to_fp16(w_int4, scale1, scale2, zero, group_size):
    res = unpack_int4_data(w_int4).to(torch.float16).view(-1, group_size)
    return res.sub_(zero).mul(scale2).view(w_int4.shape[0], -1).mul_(scale1)

# @torch.compile()
@torch.no_grad()
def dequant_int4_to_int8(weight_int4, scale_2_i, zero_2_i, group_size):
    w_8 = unpack_int4_data(weight_int4).view(-1, group_size)
    return w_8.sub_(zero_2_i).mul_(scale_2_i).view(weight_int4.shape[0], -1)

def str_format(backend, layout, deq_type):
    return f'{backend:6s}   {layout:6s}    {deq_type:6s}'

def test(K, N):
    print(f'K: {K} N: {N}')
    print(f"{'backend':6s}  {'layout':4s}  {'deq_type':6s}  time")
    group_size = min(128, K)
    weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()
    weight_i8, q_config1 = quantize_tensor_int8(weight_tensor, max_int=120)
    scale_1 = q_config1['scale']
    weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=group_size)
    scale_2_f = q_config2['scale_f']
    scale_2_i = q_config2['scale_i']
    zero_2_f = q_config2['zero_pt_f']
    zero_2_i = q_config2['zero_pt_i']
    scale_f = scale_2_f.view(scale_1.shape[0], -1) * scale_1
    weight_int4 = pack_int4_data(weight_i4.clone())
    weight_int4_awq = pack_int_awq(weight_i4.clone())

    # TEST DEQUANT KERNEL
    w_i8 = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
    edq_cuda_accel.dequant_int4_to_int8(w_i8, weight_int4, scale_2_i, zero_2_i, group_size)

    w_8  = dequant_int4_to_int8(weight_int4, scale_2_i,
                                zero_2_i, group_size)
    assert (w_8-w_i8).sum() == 0

    ########## DIRECT PACKED FORMAT ###############
    ref_w_16 = dequant_int4_to_fp16(weight_int4, scale_1, scale_2_f, zero_2_f, group_size)
    w_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)
    assert (ref_w_16-w_f16).abs().max() < 0.01, print((ref_w_16-w_f16).abs())
    # print(ref_w_16)

    ########## AWQ LAYOUT FORMAT ###############
    ########## UINT4 -> FLOAT16  ###############
    q_s = scale_f.T.contiguous()
    q_z = (-zero_2_f.reshape(scale_f.shape)*scale_f).T.contiguous()
    ref_dqw = dequantize_tensor_awq(weight_i4, scale_f, zero_2_f)
    dqw_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    edq_cuda_accel.dequant_interleaved_int4_to_fp16(dqw_f16, weight_int4_awq, q_s, q_z, 128)
    assert (ref_dqw-dqw_f16).abs().max() < 0.01, print((ref_dqw-dqw_f16).abs().max())
    
    ########## AWQ LAYOUT FORMAT ###############
    ##########  UINT4 -> INT8    ###############
    q_s_8 = scale_2_f.reshape(scale_f.shape).T.contiguous()
    q_z_8 = (-zero_2_f*scale_2_f).reshape(scale_f.shape).T.contiguous()
    ref_i8 = dequantize_tensor_awq(weight_i4, scale_2_f, zero_2_f).to(torch.int8)
    dqw_i8 = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
    edq_cuda_accel.dequant_interleaved_int4_to_int8(dqw_i8, weight_int4_awq, q_s_8, q_z_8, 128)
    # assert (ref_i8-weight_i8).abs().max() < 0.01, print((ref_i8-weight_i8).abs().max())
    assert (ref_i8-dqw_i8).abs().max() <= 1, print((ref_i8-dqw_i8).abs().max())
    
    w_i8_cpu = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
    edq_cuda_accel.dequant_int4_to_int8(w_i8, weight_int4, scale_2_i, zero_2_i, group_size)
    w_f16_cpu = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)
    w_i4_cpu = weight_int4.cpu()
    w_i8_cpu = w_i8_cpu.cpu()
    w_f16_cpu = w_f16_cpu.cpu()
    
    ########## TEST TRANS LAYOUT ##########
    torchLinear = torch.nn.Linear(in_features=K, out_features=N, bias=False, dtype=torch.float16).cuda()
    w4a16_marlin = W4A16Linear_Marlin.from_module(torchLinear)
    w4a8__marlin  = W4A8Linear_QQQ.from_module(torchLinear)
    w4a16_w = w4a16_marlin.B.cuda()
    w4a8__w = w4a8__marlin.B.cuda()
    tl_w = torch.empty(w4a16_w.shape, dtype=torch.int32, device='cuda')
    edq_cuda_accel.trans_layout_c16_to_c8(tl_w, w4a16_w)
    assert (tl_w - w4a8__w).abs().max() == 0, print((tl_w - w4a8__w).abs().max())

    preheat_time = 20
    for _ in range(preheat_time):
        w_8 = dequant_int4_to_int8(weight_int4, scale_2_i,
                                    zero_2_i, group_size)
        
        w_16 = dequant_int4_to_fp16(weight_int4, scale_1, scale_2_f,
                                    zero_2_f, group_size)
        
        w_i8 = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
        edq_cuda_accel.dequant_int4_to_int8(w_i8, weight_int4, scale_2_i, zero_2_i, group_size)
        # edq_cuda_accel.dequant_int4_to_int8_stream(w_i8, weight_int4, 
        #                                     scale_2_i, zero_2_i, group_size, cur_stream_ptr)

        w_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
        edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)
        # edq_cuda_accel.dequant_int4_to_fp16_stream(w_f16, weight_int4, 
        #                                     scale_f, zero_2_f, group_size, cur_stream_ptr)

        dqw_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
        edq_cuda_accel.dequant_interleaved_int4_to_fp16(dqw_f16, weight_int4_awq, q_s, q_z, 128)

        edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)

        tl_w = torch.empty(w4a16_w.shape, dtype=torch.int32, device='cuda')
        edq_cuda_accel.trans_layout_c16_to_c8(tl_w, w4a16_w)
        # w_i4 = w_i4_cpu.to('cuda')
        # w_i8 = w_i8_cpu.to('cuda')
        # w_f16 = w_f16_cpu.to('cuda')
    
    n = 100
    torch.cuda.empty_cache()
    desc = str_format('torch', 'DRPACK', '4->8')
    with timer(desc, n=n):
        for _ in range(n):
            w_8 = dequant_int4_to_int8(weight_int4, scale_2_i, zero_2_i, group_size)
                  
    torch.cuda.empty_cache() 
    w_i8 = torch.empty_like(weight_tensor, dtype=torch.int8, device='cuda')
    desc = str_format('cuda', 'DRPACK', '4->8')
    with timer(desc, n=n):
        for _ in range(n):
            edq_cuda_accel.dequant_int4_to_int8(w_i8, weight_int4, scale_2_i, zero_2_i, group_size)

    torch.cuda.empty_cache()
    dqw_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    desc = str_format('cuda', 'AWQ', '4->8')
    with timer(desc, n=n):
        for _ in range(n):
            edq_cuda_accel.dequant_interleaved_int4_to_int8(dqw_i8, weight_int4_awq, q_s_8, q_z_8, 128)

    torch.cuda.empty_cache()
    desc = str_format('torch', 'DRPACK', '4->16')
    with timer(desc, n=n):
        for _ in range(n):
            w_16 = dequant_int4_to_fp16(weight_int4, scale_1, scale_2_f,
                                    zero_2_f, group_size)
   
    torch.cuda.empty_cache()
    w_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    desc = str_format('cuda', 'DRPACK', '4->16')
    with timer(desc, n=n):
        for _ in range(n):
            edq_cuda_accel.dequant_int4_to_fp16(w_f16, weight_int4, scale_f, zero_2_f, group_size)

    torch.cuda.empty_cache()
    dqw_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    desc = str_format('cuda', 'AWQ', '4->16')
    with timer(desc, n=n):
        for _ in range(n):
            edq_cuda_accel.dequant_interleaved_int4_to_fp16(dqw_f16, weight_int4_awq, q_s, q_z, 128)
    
    torch.cuda.empty_cache()
    dqw_f16 = torch.empty_like(weight_tensor, dtype=torch.float16, device='cuda')
    desc = str_format('cuda', 'TL', '4->4')
    with timer(desc, n=n):
        for _ in range(n):
            # tl_w = torch.empty(w4a16_w.shape, dtype=torch.int32, device='cuda')
            edq_cuda_accel.trans_layout_c16_to_c8(tl_w, w4a16_w)
    # torch.cuda.empty_cache()
    # with timer('CPU->GPU get4   ', n=n):
    #     for _ in range(n):
    #         w_i4 = w_i4_cpu.to('cuda')

    # torch.cuda.empty_cache()
    # with timer('CPU->GPU get8   ', n=n):
    #     for _ in range(n):
    #         w_i8 = w_i8_cpu.to('cuda')

    # torch.cuda.empty_cache()
    # with timer('CPU->GPU get16  ', n=n):
    #     for _ in range(n):
    #         w_f16 = w_f16_cpu.to('cuda')
    print('-' * 30)

if __name__ == '__main__':
    # w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    w_shape = [(4096, 4096)]
    for s in w_shape:
        test(s[0], s[1])

exit(0)