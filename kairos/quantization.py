import torch

@torch.no_grad()    # per-tensor
def quantize_tensor(x, bit = 8, q_type = 'S'):
    assert q_type in ('A', 'S'), 'Illegal quantification config.'
    x_shape = x.shape    
    x = x.reshape(-1, x_shape[-1])
    max_int = 2**(bit-1) - 1
    x = x.to(torch.float32)
    if q_type == 'A':
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
    high_bits = q_data[:, 0::2] << 4   
    low_bits = q_data[:, 1::2]         
    pack_data = high_bits | low_bits
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
    if q_type == 'A':   
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