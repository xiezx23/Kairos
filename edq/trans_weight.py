import torch

# Precompute permutations for weight
@torch.no_grad() 
def _get_perms_format():
    perm = []
    for i in range(32):
        _perm = []; col = i // 4
        for block in [0, 1]:
            for row in [(i % 4), (i % 4 + 4)]:
                _perm.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 128 * j for p in _perm])
    return torch.tensor(perm)

_perm_w = _get_perms_format()
mask = 0b00001111

# NOTE: use torch.compile here will have wrong result.
@torch.no_grad()
def permute_4int8_to_int32(t):
    shape = t.shape
    t = t.reshape(-1, 4)
    ha = (t[:,0] >> 4) & mask
    la = t[:,0] & mask
    lb = t[:,1] & mask
    t[:,0] = (t[:,1] & ~mask) | ha
    t[:,1] = la | (lb << 4)

    ha = (t[:,2] >> 4) & mask
    la = t[:,2] & mask
    lb = t[:,3] & mask
    t[:,2] = (t[:,3] & ~mask) | ha
    t[:,3] = la | (lb << 4) 
    # NOTE: view(int32) to 4int8(a,b,c,d) => (dcba)
    t[:, [1, 2]] = t[:, [2, 1]]
    return t.reshape(shape).view(torch.int32)

@torch.no_grad()
@torch.compile()
def _vectorized_4int8_to_int32(t):
    t0, t1, t2, t3 = t.unbind(-1)
    mask = 0x0F
    ha = (t0 >> 4) & mask
    la = t0 & mask
    lb = t1 & mask
    new_t0 = (t1 & ~mask) | ha
    new_t1 = la | (lb << 4)
    hc = (t2 >> 4) & mask
    lc = t2 & mask
    ld = t3 & mask
    new_t2 = (t3 & ~mask) | hc
    new_t3 = lc | (ld << 4)
    result = torch.cat([
        new_t0.unsqueeze(-1),
        new_t2.unsqueeze(-1), 
        new_t1.unsqueeze(-1),
        new_t3.unsqueeze(-1)
    ], dim=-1)
    return result
    return result.view(torch.int32)

""" Permute packed int8(2*int4) data to int32(8*int4) data for edq W4A16. """
@torch.no_grad()
@torch.compile()
def permute_weight(w):
    # w: (in_eat // 2, out_feat) 2*int4 store in int8.
    in_features = w.shape[0]; out_features = w.shape[1]
    tile = (8, 16)
    w = w.reshape((in_features // tile[0], tile[0], out_features // tile[1], tile[1]))\
            .permute((0, 2, 1, 3))\
            .reshape((-1, _perm_w.numel()))[:, _perm_w]\
            .reshape(in_features // tile[0], out_features * tile[0]//4, 4)
    w = _vectorized_4int8_to_int32(w).reshape(in_features // tile[0], -1)
    return w


