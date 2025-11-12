import torch
import numpy
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width, pack_int_awq, dequantize_tensor_awq
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from edq.dynamic_linear import edq_quantize_tensor_int4

torch.manual_seed(seed=43)

def printLoss(desc, loss):
    print(f"{desc:10s} : {loss.item():2.4f}")

def countLoss(ref, cur, method = 'mse'):
    # method = 'mean'
    if method == 'mse':
        dif = (ref-cur).to(float).pow(2).sum()
    else:
        dif = (ref-cur).to(float).abs().sum()
    return dif / ref.numel()

print_detail = False
record = ([],[])

def print_record():
    print(f'Max Reduce Rate: {max(record[0]):.2f}%   {max(record[1]):.2f}%')
    print(f'Min Reduce Rate: {min(record[0]):.2f}%   {min(record[1]):.2f}%')
    rr1 = sum(record[0])/len(record[0])
    rr2 = sum(record[1])/len(record[1])
    print(f'Average Reduce Rate: {rr1:.2f}%   {rr2:.2f}%')

def test_TSQ(weight_tensor):
    N, K = weight_tensor.shape
    group_size = min(128, K)
    # group_size = min(4, K)
    int8_group = -1 # -1: per_channel  -2: per_tensor
    weight_i8, q_config1 = quantize_tensor_int8(weight_tensor, max_int=127, group_size=int8_group)
    scale_1 = q_config1['scale']
    dqw_from_i8 = dequantize_tensor_int8(weight_i8, scale_1, group_size=int8_group)
    if print_detail: printLoss('8bit Qloss', countLoss(weight_tensor,dqw_from_i8))

    weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=group_size)
    scale_2_f = q_config2['scale_f']
    zero_2_f  = q_config2['zero_pt_f']
    scale_f = scale_2_f.view(scale_1.shape[0], -1) * scale_1

    ref_4_16 = dequantize_tensor_awq(weight_i4, scale_f, zero_2_f, group_size=group_size)
    loss1 = countLoss(weight_tensor,ref_4_16)
    if print_detail: printLoss('I4 to F16', loss1)

    ref_4_8 = dequantize_tensor_awq(weight_i4, scale_2_f, zero_2_f, group_size=group_size).round_().to(torch.int8)
    # print('dq int8 w\n', ref_4_8)

    ref_8_16 = dequantize_tensor_int8(ref_4_8, scale_1, group_size=int8_group)
    # print('dq int16 w\n', ref_8_16)

    # print('diff i8 w\n', weight_i8 - ref_4_8)
    loss2 = countLoss(weight_tensor,ref_8_16)
    if print_detail: printLoss('I4->8->F16', loss2)
    # print('-'*40)

    weight_direct_i4, q_config3 = edq_quantize_tensor_int4(weight_tensor, group_size=group_size)
    scales = q_config3['scale_f']
    zeros  = q_config3['zero_pt_f']
    ref_4_16 = dequantize_tensor_awq(weight_direct_i4, scales, zeros, group_size=group_size)
    if print_detail: printLoss('4bit Qloss', countLoss(weight_tensor,ref_4_16))
    loss3 = countLoss(weight_tensor,ref_4_16)
    if print_detail: printLoss('I4 to F16', loss3)
    
    weight_direct_i8, q_config4 = quantize_tensor_int8(weight_tensor, group_size=int8_group)
    # print('q int8 w\n', weight_direct_i8)
    scales_8 = q_config4['scale']
    scales_4to8 = scales.view(N,-1) / scales_8
    # print('q int4 w\n', weight_direct_i4)
    # print('scales_4to8\n', scales_4to8)
    # print('zeros\n',zeros)
    ref_4_8 = dequantize_tensor_awq(weight_direct_i4, scales_4to8, zeros, group_size=group_size).round_().to(torch.int8)
    # print('dq int8 w\n', ref_4_8)
    ref_8_16 = dequantize_tensor_int8(ref_4_8, scales_8, group_size=int8_group)
    # print('diff i8 w\n', weight_direct_i8 - ref_4_8)
    loss4 = countLoss(weight_tensor,ref_8_16)
    if print_detail: printLoss('I4->8->F16', loss4)
    rr1 = (loss1-loss3)/loss1*100
    rr2 = (loss2-loss4)/loss2*100
    record[0].append(rr1)
    record[1].append(rr2)
    colored_print(f'Reduce Rate: {rr1:.2f}%   {rr2:.2f}%', 'green')
    print('-'*40)

    # ref_4_8 = dequantize_tensor_awq(weight_direct_i4, scale_2_f, zero_2_f, group_size=group_size).round_().to(torch.int8)
    # # print('dq int8 w\n', ref_4_8)
    # ref_8_16 = dequantize_tensor_int8(ref_4_8, scale_1, group_size=int8_group)
    # # print('diff i8 w\n', weight_direct_i8 - ref_4_8)
    # if print_detail: printLoss('loss 5', countLoss(weight_tensor,ref_8_16))

def test(K, N):
    print(f'K: {K} N: {N}')
    group_size = min(16, K)
    # group_size = min(4, K)
    int8_group = -1 # -1: per_channel  -2: per_tensor
    # weight_tensor = (torch.rand((N, K), dtype=torch.float16) - 0.5).mul_(20).cuda()
    weight_tensor = torch.randn((N, K), dtype=torch.float16).mul_(10).cuda()
    weight_tensor[:, 4].mul_(2)
    weight_tensor[:, 8].mul_(3)
    weight_i8, q_config1 = quantize_tensor_int8(weight_tensor, max_int=127, group_size=int8_group)
    scale_1 = q_config1['scale']
    dqw_from_i8 = dequantize_tensor_int8(weight_i8, scale_1, group_size=int8_group)
    printLoss('8bit Qloss', countLoss(weight_tensor,dqw_from_i8))

    weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=group_size)
    scale_2_f = q_config2['scale_f']
    zero_2_f  = q_config2['zero_pt_f']
    scale_f = scale_2_f.view(scale_1.shape[0], -1) * scale_1

    ref_4_16 = dequantize_tensor_awq(weight_i4, scale_f, zero_2_f, group_size=group_size)
    loss1 = countLoss(weight_tensor,ref_4_16)
    printLoss('I4 to F16', loss1)

    ref_4_8 = dequantize_tensor_awq(weight_i4, scale_2_f, zero_2_f, group_size=group_size).round_().to(torch.int8)
    # print('dq int8 w\n', ref_4_8)

    ref_8_16 = dequantize_tensor_int8(ref_4_8, scale_1, group_size=int8_group)
    # print('dq int16 w\n', ref_8_16)

    # print('diff i8 w\n', weight_i8 - ref_4_8)
    loss2 = countLoss(weight_tensor,ref_8_16)
    printLoss('I4->8->F16', loss2)
    print('-'*40)

    weight_direct_i4, q_config3 = edq_quantize_tensor_int4(weight_tensor, group_size=group_size)
    scales = q_config3['scale_f']
    zeros  = q_config3['zero_pt_f']
    ref_4_16 = dequantize_tensor_awq(weight_direct_i4, scales, zeros, group_size=group_size)
    printLoss('4bit Qloss', countLoss(weight_tensor,ref_4_16))
    loss3 = countLoss(weight_tensor,ref_4_16)
    printLoss('I4 to F16', loss3)
    
    weight_direct_i8, q_config4 = quantize_tensor_int8(weight_tensor, group_size=int8_group)
    # print('q int8 w\n', weight_direct_i8)
    scales_8 = q_config4['scale']
    scales_4to8 = scales.view(N,-1) / scales_8
    # print('q int4 w\n', weight_direct_i4)
    # print('scales_4to8\n', scales_4to8)
    # print('zeros\n',zeros)
    ref_4_8 = dequantize_tensor_awq(weight_direct_i4, scales_4to8, zeros, group_size=group_size).round_().to(torch.int8)
    # print('dq int8 w\n', ref_4_8)
    ref_8_16 = dequantize_tensor_int8(ref_4_8, scales_8, group_size=int8_group)
    # print('diff i8 w\n', weight_direct_i8 - ref_4_8)
    loss4 = countLoss(weight_tensor,ref_8_16)
    printLoss('I4->8->F16', loss4)
    colored_print(f'Reduce Rate: {(loss1-loss3)/loss1*100:.2f}%   {(loss2-loss4)/loss2*100:.2f}%', 'green')
    # print('-'*40)

    # ref_4_8 = dequantize_tensor_awq(weight_direct_i4, scale_2_f, zero_2_f, group_size=group_size).round_().to(torch.int8)
    # # print('dq int8 w\n', ref_4_8)
    # ref_8_16 = dequantize_tensor_int8(ref_4_8, scale_1, group_size=int8_group)
    # # print('diff i8 w\n', weight_direct_i8 - ref_4_8)
    # printLoss('loss 5', countLoss(weight_tensor,ref_8_16))

if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(8, 4)]
    for s in w_shape:
        test(s[0], s[1])
        print('='*40)