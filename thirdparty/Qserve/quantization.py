import torch

@torch.no_grad()
def qserve_quantize_tensor_int8(x, return_dtype = torch.float16):
    x_shape = x.shape    
    x = x.reshape(-1, x_shape[-1])
    x = x.to(torch.float32)
    max_int = 119   # Qserve Max Int8
    max_val = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-7)
    scale   = max_val / max_int
    # x = torch.round(x / scale)
    x.div_(scale).round_()
    x = x.to(torch.int8)
    assert torch.isnan(x).sum() == 0
    return x.reshape(x_shape), {'scale':scale.to(return_dtype)}

@torch.no_grad()
def qserve_quantize_tensor_int4(x, group_size = -1):
    x_shape = x.shape    
    if group_size < 0:
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    max_int = 15    # 2**4 - 1``
    max_val = x.amax(dim=1, keepdim=True)
    min_val = x.amin(dim=1, keepdim=True)
    scale   = torch.round((max_val-min_val).clamp(min=1e-7) / max_int)
    scale[scale == 0] = 1
    zero_pt = torch.round(-(min_val / scale))
    x.div_(scale).add_(zero_pt).round_()
    assert torch.isnan(x).sum() == 0, print(scale.amin())
    x.clamp_(0, max_int)
    return x.reshape(x_shape).to(torch.int8), {
        'scale':scale.to(torch.int8), 
        'zero_pt':zero_pt.to(torch.int8), 'group_size':group_size}