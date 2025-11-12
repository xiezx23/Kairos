import torch
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin, _perm as perm1
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ, _perm as perm2

import edq_cuda_accel
torch.manual_seed(seed=11)

# print(perm1.view(-1, 8))

# print(perm2.view(-1, 8))

# exit(0)


def depack_int32(data):
    mask = 0x0f
    a = data & mask; data >>= 4
    b = data & mask; data >>= 4
    c = data & mask; data >>= 4
    d = data & mask; data >>= 4
    e = data & mask; data >>= 4
    f = data & mask; data >>= 4
    g = data & mask; data >>= 4
    h = data & mask; data >>= 4
    print(f'{a.item():3d} {b.item():3d} {c.item():3d} {d.item():3d} {e.item():3d} {f.item():3d} {g.item():3d} {h.item():3d}')

K = 2048
N = 2048
bias = False
torchLinear = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
# print(torchLinear.weight.data)


w4a16 = W4A16Linear_Marlin.from_module(torchLinear)
w4a8  = W4A8Linear_QQQ.from_module(torchLinear)
# print(torchLinear.weight.data)

m_w = w4a16.B.cuda().contiguous()
q_w = w4a8.B.cuda().contiguous()

p_w = torch.empty(m_w.shape, dtype=torch.int32, device='cuda')
edq_cuda_accel.trans_layout_c16_to_c8(p_w, m_w)
torch.cuda.synchronize()

assert (p_w - q_w).abs().max() == 0, print((p_w - q_w).abs().max())

"""CHECK CORRECTNESS OF TRANS LAYOUT"""
# mask = 0x0f
# for i in range(p_w.shape[0]):
#     for j in range(p_w.shape[1]):
#         a = p_w[i][j]; b = q_w[i][j]
#         a1 = a & mask; a >>= 4
#         a2 = a & mask; a >>= 4
#         a3 = a & mask; a >>= 4
#         a4 = a & mask; a >>= 4
#         b1 = b & mask; b >>= 4
#         b2 = b & mask; b >>= 4
#         b3 = b & mask; b >>= 4
#         b4 = b & mask; b >>= 4
#         assert(abs(a1-b1) <= 1), print(i, j, a1, b1)
#         assert(abs(a2-b2) <= 1), print(i, j, a1, b1)
#         assert(abs(a3-b3) <= 1), print(i, j, a1, b1)
#         assert(abs(a4-b4) <= 1), print(i, j, a1, b1)

# print(m_w)

# print(q_w)

# print(p_w)
# print('='*32)
# for i in range(4, 5):
#     depack_int32(m_w[0][0 + 4*i])
#     depack_int32(m_w[0][1 + 4*i])
#     depack_int32(m_w[0][2 + 4*i])
#     depack_int32(m_w[0][3 + 4*i])
#     print('-' * 32)

# print('='*32)
# for i in range(8, 10):
#     depack_int32(q_w[0][0 + 4*i])
#     depack_int32(q_w[0][1 + 4*i])
#     depack_int32(q_w[0][2 + 4*i])
#     depack_int32(q_w[0][3 + 4*i])
#     print('-' * 32)

# print('='*32)
# for i in range(8, 10):
#     depack_int32(p_w[0][0 + 4*i])
#     depack_int32(p_w[0][1 + 4*i])
#     depack_int32(p_w[0][2 + 4*i])
#     depack_int32(p_w[0][3 + 4*i])
#     print('-' * 32)

m = 6
input = torch.randn((m, K), device='cuda', dtype=torch.float16)

ref_out = torchLinear(input)

out1 = w4a16(input)
out2 = w4a8(input)

print(ref_out)
print(out1)
print(out2)