import torch
import time


@torch.no_grad()
def real_quantize_tensor_edq(
    w, n_bit=4, q_group_size=128):
    org_w_shape = w.shape
    if q_group_size > 0:
    #     assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    else:
        w = w.reshape(-1, w.shape[-1])
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    # NOTE: AWQ employs an integer zero value, but this is suboptimal.
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int)
    # zeros = (-(min_val / scales))
    # w.div_(scales).add_(zeros).round_().clamp_(0, max_int)
    # assert torch.isnan(scales).sum() == 0
    # assert torch.isnan(w).sum() == 0
    # assert torch.isnan(w).sum() == 0
    w = w.reshape(org_w_shape).to(dtype=torch.int32)
    return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)

@torch.no_grad()
def dequantize_tensor_edq(weight, scales, scaled_zeros, group_size=128):
    weight_shape = weight.shape
    # assert weight_shape[-1] % group_size == 0
    weight = weight.reshape(-1, group_size) 
    weight = weight.to(torch.float16)
    scales = scales.reshape(-1, 1)
    scaled_zeros = scaled_zeros.reshape(-1, 1)
    weight = weight.mul_(scales).add_(scaled_zeros).reshape(weight_shape)
    return weight

@torch.no_grad()    # per-tensor
def quantize_tensor(x, bit = 8, q_type = 'S'):
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    x_shape = x.shape    
    x = x.reshape(-1, x_shape[-1])
    max_int = 2**(bit-1) - 1
    x = x.to(torch.float32)
    if q_type == 'A':    # 非对称量化,注意是有符号量化,uint8在计算时会溢出
        max_val = x.amax()
        min_val = x.amin()
        scale   = (max_val-min_val).clamp(min=1e-7) / 2 / max_int
        zero_pt = -torch.round((max_val+min_val) / 2 / scale)
        # x = torch.round(x / scale) + zero_pt
        x.div_(scale).round_().add_(zero_pt)
        # assert torch.isnan(x).sum() == 0
        # Dequantize: px = (x - zero_pt) * scale
    else:
        max_val = x.abs().amax().clamp(min=1e-7)
        scale   = max_val / max_int
        zero_pt = 0
        # x = torch.round(x / scale)
        x.div_(scale).round_()
        # assert torch.isnan(x).sum() == 0
        # Dequantize: px = x * scale
    # Q_MAX = x.max()
    # Q_MIN = x.min()
    # print('qmax: {},  qmin: {}'.format(Q_MAX.item(), Q_MIN.item()))
    x.clamp_(min = -max_int, max = max_int)
    return x.reshape(x_shape).to(torch.int8), {'q_type':q_type, 'scale':scale, 'zero_pt':zero_pt}

@torch.no_grad()    # per-tensor
def dequantize_tensor(x, scale, zero_pt = None, q_type = 'S'):
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1]).to(torch.float16)
    if q_type == 'A':
        # x = (x - zero_pt) * scale
        x.sub_(zero_pt).mul_(scale)
    else:
        # px = x * scale
        x.mul_(scale)
    return x.reshape(x_shape)

@torch.no_grad()
def pack_int4_data(q_data):
    high_bits = q_data[:, 0::2] << 4     # 取偶数列(0,2,4...)左移4位
    low_bits = q_data[:, 1::2]           # 取奇数列(1,3,5...)
    pack_data = high_bits | low_bits     # 位或合并
    return pack_data.to(torch.int8)

@torch.compile()
@torch.no_grad()
def unpack_int4_data(pack_data):
    high_bits = ((pack_data >> 4) & 0x0F)
    low_bits = (pack_data & 0x0F)
    unpack_data = torch.stack((high_bits, low_bits), dim=-1)
    return unpack_data.flatten(start_dim=1)

@torch.no_grad()
def quantize_tensor_int4(x, q_type = 'A', group_size = -1):
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    x_shape = x.shape    
    if group_size < 0:
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    if q_type == 'A':   # 量化完是一个无符号数
        max_int = 15    # 2**4 - 1``
        max_val = x.amax(dim=1, keepdim=True)
        min_val = x.amin(dim=1, keepdim=True)
        scale   = (max_val-min_val).clamp(min=1e-7) / max_int
        zero_pt = -(min_val / scale)
        x.div_(scale).add_(zero_pt).round_()
        assert torch.isnan(x).sum() == 0
        x.clamp_(0, max_int)
    else:
        max_int = 7
        max_val = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        scale   = max_val / max_int
        # x = torch.round(x / scale)
        x.div_(scale).round_()
        assert torch.isnan(x).sum() == 0
        x.clamp_(-max_int, max_int)
        zero_pt = torch.tensor([0])
    return x.reshape(x_shape).to(torch.int8), {
        'q_type':q_type, 'scale':scale.to(torch.float16), 
        'zero_pt':zero_pt.to(torch.float16), 'group_size':group_size}

@torch.no_grad()
def dequantize_tensor_int4(x, scale, q_type = 'A', zero_pt = None, group_size = -1):
    x_shape = x.shape
    if group_size < 0:
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size) 
    x = x.to(torch.float16)
    zero_pt = zero_pt.reshape(-1,1)
    scale = scale.reshape(-1,1)
    if q_type == 'A':
        # x = (x - zero_pt) * scale
        x.sub_(zero_pt).mul_(scale)
    else:
        # x = x * scale
        x.mul_(scale)
    return x.reshape(x_shape)

@torch.no_grad()
def quantize_tensor_int8(x, q_type = 'S', group_size = -1, return_dtype = torch.float16, max_int = -1):
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    x_shape = x.shape    
    if group_size == -1:    # per_channel
        x = x.reshape(-1, x_shape[-1])
    elif group_size == -2:  # per_tensor
        x = x.reshape(1, -1)
    x = x.to(torch.float32)
    if q_type == 'A':
        if max_int == -1:
            max_int = 2**8 - 1
        max_val = x.amax(dim=1, keepdim=True)
        min_val = x.amin(dim=1, keepdim=True)
        scale   = (max_val-min_val).clamp(min=1e-7) / max_int
        zero_pt = -(min_val / scale)
        x.div_(scale).add_(zero_pt).round_()
        x = x.to(torch.uint8)
        assert torch.isnan(x).sum() == 0
    else:
        if max_int == -1:
            max_int = 127   # 2**(8-1) - 1
        max_val = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
        scale   = max_val / max_int
        zero_pt = torch.tensor([0])
        # x = torch.round(x / scale)
        x.div_(scale).round_()
        x = x.to(torch.int8)
        assert torch.isnan(x).sum() == 0
    return x.reshape(x_shape), {
        'q_type':q_type, 'scale':scale.to(return_dtype), 'zero_pt':zero_pt.to(return_dtype)}

@torch.no_grad()
def dequantize_tensor_int8(x, scale, q_type = 'S', zero_pt = None, 
                           group_size = -1, return_dtype = torch.float16):
    x_shape = x.shape
    if group_size == -1:    # per_channel
        x = x.reshape(-1, x_shape[-1])
    elif group_size == -2:  # per_tensor
        x = x.reshape(1, -1)
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size) 
    scale = scale.reshape(-1,1)
    x = x.to(return_dtype)
    if q_type == 'A':
        # x = (x - zero_pt) * scale
        x.sub_(zero_pt).mul_(scale)
    else:
        # x = x * scale
        x.mul_(scale)
    return x.reshape(x_shape)

@torch.no_grad()
def post_gemm_dequant(o, x, w, q_config_x, q_config_w):
    o = o.to(torch.float32)
    # assert q_config_w['q_type'] == 'S', \
        # 'Only support dequantize to symmetric quantization of weight.'
    # assert q_config_x['dim'] == 0 and q_config_w['dim'] == 1
    if (q_config_x['q_type'] == 'S'):
        o = o * q_config_x['scale'] * q_config_w['scale']
    else:
        d =  torch.cat(
            [
            w.to(torch.float32).sum(dim=0, keepdim=True) 
            for _ in range(x.shape[0])
            ], dim=0
        )
        o = (o - q_config_x['zero_pt'] * d) * q_config_x['scale'] * q_config_w['scale']
    return o

@torch.no_grad()
def post_gemm_dequant(o, x, w, x_scale, w_scale, x_zero_pt = None):
    # x_scale = x_scale.view(-1,1)
    # w_scale = w_scale.view(1,-1)
    w_scale = w_scale.T
    o = o.to(torch.float32)
    if 1:   # Symmetric Quantization
        # print(o.shape)
        # print(x_scale.shape)
        # print(w_scale.shape)
        o.mul_(x_scale).mul_(w_scale)
    else:
        d =  torch.cat(
            [
            w.to(torch.float32).sum(dim=0, keepdim=True) 
            for _ in range(x.shape[0])
            ], dim=0
        )
        o = (o - x_zero_pt * d) * x_scale * w_scale
    return o

@torch.no_grad()
def simu_quantize_tensor(x, bit = 8, q_type = 'S', group_size = -1, return_dtype = torch.float16):
    """
    dim         : 0->per_token / 1->per_channel
    group_size  : how many value will share one scale and zero point. -1: per-input_channel
    """
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    assert group_size < 0 or x.shape[-1] % group_size == 0, print(x.shape)
    dim = 0     # quantize by rows
    x_shape = x.shape
    if group_size < 0:  # per-input_channel
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    if q_type == 'A':
        max_int = 2**(bit) - 1
        max_val = x.amax(dim=(1^dim), keepdim=True)
        min_val = x.amin(dim=(1^dim), keepdim=True)
        scale   = (max_val-min_val).clamp(min=1e-7) / max_int
        zero_pt = -(min_val / scale)
        qx = torch.round(x / scale + zero_pt)
        qx = qx.clamp(min = 0, max = max_int)
        assert torch.isnan(qx).sum() == 0
        zero_pt = zero_pt.to(return_dtype)
        scale = scale.to(return_dtype)
        pqx = (qx - zero_pt) * scale
    else:
        max_int = 2**(bit-1) - 1
        max_val = x.abs().amax(dim=(1^dim), keepdim=True)
        scale   = (max_val / max_int).clamp(min=1e-7)
        qx = torch.round(input=x / scale)
        qx = qx.clamp(min = -max_int-1, max = max_int)
        assert torch.isnan(qx).sum() == 0
        scale = scale.to(return_dtype)
        pqx = qx * scale
    # Q_MAX = qx.max()
    # Q_MIN = qx.min()
    # print('qmax: {},  qmin: {},  groupSize: {}'.format(Q_MAX.item(), Q_MIN.item(), group_size))
    return pqx.reshape(x_shape).to(torch.float16)

@torch.no_grad()
def simu_quantize_tensor_autoscale(x, bit = 8, q_type = 'S', group_size = -1):
    """
    dim         : 0:->per_token / 1->per_channel
    group_size  : how many vector will share one scale and zero point.
    with_scale  : scale on the other dim to smooth the data distribution.
    """
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    dim = 0     # quantize by rows
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])
    # SCALE
    max_ori_val = x.abs().amax(dim=dim)
    total_max_ori_val = x.abs().amax().item()
    smooth_scale = total_max_ori_val / max_ori_val
    x = x * smooth_scale
    # RESHAPE FOR GROUP QUANTIZE
    assert group_size < 0 or x.shape[-1] % group_size == 0, print(x.shape)
    max_int = 2**(bit-1) - 1
    if group_size > 0:
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    # QUANTIZATION
    max_int = 2**(bit-1) - 1
    if q_type == 'A':
        max_val = x.amax(dim=(1^dim), keepdim=True)
        min_val = x.amin(dim=(1^dim), keepdim=True)
        scale   = (max_val-min_val).clamp(min=1e-7) / 2 / max_int
        zero_pt = -torch.round((max_val+min_val) / 2 / scale)
        qx = torch.round(x / scale) + zero_pt
        qx = qx.clamp(min = -max_int, max = max_int)
        assert torch.isnan(qx).sum() == 0
        pqx = (qx - zero_pt) * scale
    else:
        max_val = x.abs().amax(dim=(1^dim), keepdim=True)
        scale   = (max_val / max_int).clamp(min=1e-7)
        qx = torch.round(input=x / scale)
        qx = qx.clamp(min = -max_int, max = max_int)
        assert torch.isnan(qx).sum() == 0
        pqx = qx * scale
    # Q_MAX = qx.max()
    # Q_MIN = qx.min()
    # print('qmax: {},  qmin: {},  groupSize: {}'.format(Q_MAX.item(), Q_MIN.item(), group_size))
    return pqx.reshape(x_shape).to(torch.float16).div_(smooth_scale)

@torch.no_grad()
def simu_quantize_weight_mix_precsion(x, acti_max_row, bit = (4, 4), q_type = 'S', group_size = -1):
    """
    !注意: 对于 A(N,C) @ W(C,M), W在存储时按(M,C)的形状
    acti_max_row: the abs average value in column of Activation.
    dim         : 0:->per_token / 1->per_channel.
    group_size  : how many vector will share one scale and zero point.
    with_scale  : scale on the other dim to smooth the data distribution.
    """
    assert x.dim() == 2
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    dim = 0     # quantize by rows
    x_shape = x.shape
    # SCALE
    """以激活值在列上的平均作为权重每一行的重要性系数"""
    high_precision_rate = 0.01
    high_precision_num = max(round(x.shape[-1]*high_precision_rate), 1)
    selected_idx = acti_max_row.topk(high_precision_num).indices

    hpw = x[:,selected_idx].clone()
    x[:,selected_idx] = 0
    hpw = simu_quantize_tensor(hpw, bit = bit[0], q_type=q_type)
    pqx = simu_quantize_tensor(x, bit = bit[1], q_type=q_type, group_size=group_size)
    pqx[:,selected_idx] = hpw
    # Q_MAX = qx.max()
    # Q_MIN = qx.min()
    # print('qmax: {},  qmin: {},  groupSize: {}'.format(Q_MAX.item(), Q_MIN.item(), group_size))
    return pqx.reshape(x_shape).to(torch.float16)

# @torch.no_grad()
# def simu_quantize_weight_mix_precsion(x, acti_max_row, bit = (8, 4), q_type = 'S', group_size = -1):
#     """
#     !注意: 对于 A(N,C) @ W(C,M), W在存储时按(M,C)的形状
#     acti_max_row: the abs average value in column of Activation.
#     dim         : 0:->per_token / 1->per_channel.
#     group_size  : how many vector will share one scale and zero point.
#     with_scale  : scale on the other dim to smooth the data distribution.
#     """
#     assert x.dim() == 2
#     assert q_type in ('A', 'S'), 'Illegal quantification config.'
#     dim = 0     # quantize by rows
#     x_shape = x.shape
#     # SCALE
#     """以激活值在列上的平均作为权重每一行的重要性系数"""
#     high_precision_rate = 0.01
#     high_precision_num = max(round(x.shape[-1]*high_precision_rate), 1)
#     selected_idx = acti_max_row.topk(high_precision_num).indices

#     def split_tensor(w, idx):
#         # 直接索引获取第一部分
#         part1 = w.index_select(1, idx)
#         # 原位创建掩码减少内存峰值
#         mask = torch.ones(w.size(1), dtype=torch.bool, device=w.device)
#         mask[idx] = 0  # 原位操作避免复制
#         part2 = w[:, mask]
#         return part1, part2
    
#     def merge_tensors(part1, part2, i):
#         device = part1.device
#         m = part1.size(1) + part2.size(1)
#         w = torch.empty((part1.size(0), m), device=device, dtype=part1.dtype)
#         mask = torch.ones(m, dtype=torch.bool, device=device)
#         mask[i] = False
#         remaining = torch.where(mask)[0]       
#         w[:, i] = part1  # 放置part1
#         w[:, remaining] = part2  # 放置part2
#         return w
    
#     hpw, lpw = split_tensor(x, selected_idx)
#     # hpw = simu_quantize_tensor(hpw, bit = bit[0], q_type=q_type)
#     lpw = simu_quantize_tensor(lpw, bit = bit[1], q_type=q_type)
#     pqx = merge_tensors(hpw, lpw, selected_idx)
#     # Q_MAX = qx.max()
#     # Q_MIN = qx.min()
#     # print('qmax: {},  qmin: {},  groupSize: {}'.format(Q_MAX.item(), Q_MIN.item(), group_size))
#     return pqx.reshape(x_shape).to(torch.float16)


if __name__ == '__main__':
    torch.manual_seed(11)
    group_size = 4
    x = torch.rand((2,8), dtype=torch.float16, device = 'cuda')
    w = torch.rand((10, 8), dtype=torch.float16, device = 'cuda')

    print(x)

    x_int4, config_x_int4 = quantize_tensor_int4(x, q_type='A', group_size=group_size)
    x_deq = dequantize_tensor_int4(x_int4, **config_x_int4)
    print(x_deq)

    x_simu = simu_quantize_tensor(x, q_type='A', bit=4, group_size=group_size)
    print(x_simu)

    exit(0)

    

    x = simu_quantize_tensor(ori_x, bit=8, q_type='S', dim = 0, group_size=4)
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-token    Symmetric Quantize Loss: ", loss)

    x = simu_quantize_tensor(ori_x, bit=8, q_type='S', dim = 1, group_size=4)
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-channel  Symmetric Quantize Loss: ", loss)

    x = simu_quantize_tensor(ori_x, bit=8, q_type='S')
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-tensor   Symmetric Quantize Loss: ", loss)

    print('------------------------------------------------')

    x = simu_quantize_tensor(ori_x, bit=8, q_type='A', dim = 0, group_size=4)
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-token    Asymmetric Quantize Loss:", loss)
    
    x = simu_quantize_tensor(ori_x, bit=8, q_type='A', dim = 1, group_size=4)
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-channel  Asymmetric Quantize Loss:", loss)

    x = simu_quantize_tensor(ori_x, bit=8, q_type='A')
    loss = (ori_x - x).pow(2).mean().item()
    print("Per-tensor   Asymmetric Quantize Loss:", loss)

    exit(0)

    x = simu_quantize_tensor(ori_x, bit=8, q_type='A', dim=1, group_size=-1)
    loss = (ori_x - x).pow(2).mean().item()
    print("Loss: ", loss)
    x = simu_quantize_tensor(ori_x, bit=8, q_type='A', dim=1, group_size=2)
    loss = (ori_x - x).pow(2).mean().item()
    print("Loss: ", loss)
    x = simu_quantize_tensor(ori_x, bit=8, q_type='A', dim=1, group_size=4)
    loss = (ori_x - x).pow(2).mean().item()
    print("Loss: ", loss)
    x, config = quantize_tensor(ori_x, q_type='A')
    x = (x - config['zero_pt']) * config['scale']
    loss = (ori_x - x).pow(2).mean().item()
    print("Loss: ", loss)
    exit(0)

    x = ori_x
    w = ori_w
    tst = time.time()
    o1 = x @ w
    torch.cuda.synchronize()  # 等待 GPU 计算完成
    ted = time.time()
    print("GPU time (FP16): ", ted - tst, "s")
    print("o.max:", o1.max())

    x, q_config1 = quantize_tensor(ori_x, bit=8, q_type='A', dim=0)
    w, q_config2 = quantize_tensor(ori_w, bit=8, q_type='S', dim=1)
    tst = time.time()
    o = torch._int_mm(x, w).to(torch.float32)
    o = post_gemm_dequant(o, x, w, q_config1, q_config2)
    torch.cuda.synchronize()  # 等待 GPU 计算完成
    ted = time.time()
    print("GPU time (INT8): ", ted - tst, "s")
    print("o.max:", o.max())

    loss = (o - o1).pow(2).mean().item()
    print("Loss: ", loss)