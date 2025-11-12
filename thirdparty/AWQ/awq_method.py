import torch

@torch.no_grad()
@torch.compile()
def pack_int_awq(unpacked_qweight, interleave = 4, kstride = 64):
    # unpacked_qweight: [N, K]
    N, K = unpacked_qweight.shape

    packed_weight = unpacked_qweight.reshape(N, K // 32, 32).to(torch.int16)
    # np.arange(32).reshape(4, 4, 2).transpose(1, 0, 2) => [0, 1, 8, 9, 16, 17, 24, 25, ...]
    packed_weight = packed_weight.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 2, 4)
    packed_weight = packed_weight.reshape(N, K // 32, 32)

    # reorder each 8 weights for fast dequantization
    # [0, 1, 2, 3, 4, 5, 6, 7] => [0, 2, 4, 6, 1, 3, 5, 7]
    packed_weight = packed_weight.reshape(N, K // 32, 4, 8)
    packed_weight = packed_weight.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 2, 4, 3)
    packed_weight = packed_weight.reshape(N, K)

    # interleaving every four rows
    packed_weight = packed_weight.reshape(
        N // interleave, interleave, K // kstride, kstride
    )
    # N // 4, K // 64, 4, 64
    packed_weight = packed_weight.permute(0, 2, 1, 3)
    packed_weight = packed_weight.reshape(
        N // interleave, K // kstride, kstride, interleave
    )
    # Packing -> (N // 4, K // 64, 64)
    packed_weight = (
        packed_weight[..., 0]
        | (packed_weight[..., 1] << 4)
        | (packed_weight[..., 2] << 8)
        | (packed_weight[..., 3] << 12)
    )
    # reshape to (N // 4, K), Int16 format
    packed_weight = packed_weight.reshape(N // interleave, K)
    return packed_weight.to(torch.int16).contiguous()

@torch.no_grad()
def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    def make_divisible(c, divisor):
        return (c + divisor - 1) // divisor
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width

@torch.no_grad()
def trans_qfactor_layout_awq(scales, zeros, group_size, method = 'AWQ'):
    dtype = torch.float16
    pack_num = 8    # 32 // 4
    qscales = torch.zeros((
            scales.shape[0],
            calculate_zeros_width(scales.shape[1] * group_size, group_size) * pack_num,
        ), dtype=dtype,device=scales.device,)
    qscales[:, : scales.shape[1]] = scales
    # awq_linear.scales = scales.clone().half()
    q_scales = qscales.transpose(1, 0).contiguous()
    if method == 'AWQ': # Zexi: I cannt understand why AWQ converts zeros to Int32 here.
        zeros = zeros.to(dtype=torch.int32)
    scaled_zeros = torch.zeros_like(qscales)
    # scaled_zeros[:, :scales.shape[1]] = -(qscales[:, :scales.shape[1]] * (zeros.to(torch.float32) - 8.0)).to(torch.float16)
    scaled_zeros[:, : scales.shape[1]] = -(
        qscales[:, : scales.shape[1]] * (zeros.to(torch.float32))
    ).to(dtype)
    q_scaled_zeros = scaled_zeros.transpose(1, 0).contiguous()
    return q_scales, q_scaled_zeros

@torch.no_grad()
def quant_weight_awq(w, group_size):
    unpack_w, scales, zeros = real_quantize_tensor(w, n_bit=4, q_group_size=group_size)
    q_qweight = pack_int_awq(unpack_w.contiguous(), interleave=4, kstride=64)
    return q_qweight, *trans_qfactor_layout_awq(scales, zeros, group_size)

@torch.no_grad()
def real_quantize_tensor(w, n_bit=4, q_group_size=128):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    else:
        w = w.reshape(-1, w.shape[-1])
    assert w.dim() == 2
    max_val = w.amax(dim=1, keepdim=True)
    min_val = w.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0
    (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int)
    assert torch.isnan(w).sum() == 0
    w = w.reshape(org_w_shape).to(dtype=torch.int32)
    return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)

@torch.no_grad()
def dequantize_tensor_awq(weight, scales, zeros, group_size=128):
    weight_shape = weight.shape
    assert weight_shape[-1] % group_size == 0
    weight = weight.reshape(-1, group_size) 
    weight = weight.to(torch.float16)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    # weight = weight.sub_(zeros).mul_(scales).reshape(weight_shape)
    scaled_zeros = -(scales*zeros)
    weight = weight.mul_(scales).add_(scaled_zeros).reshape(weight_shape)
    return weight
