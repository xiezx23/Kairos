import torch
import numpy as np
from utils.perf_eval import timer

def qserve_pack(w):
    N,K = w.shape
    W_unpack_reorder = (
        w.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
        .permute(0, 4, 3, 6, 1, 5, 2, 7)
        .contiguous())
    W_unpack_reorder = (
        W_unpack_reorder.permute(0, 1, 2, 3, 5, 6, 7, 4)
        .contiguous()
        # .to(torch.int8)
    )
    return W_unpack_reorder

@torch.no_grad
@torch.compile
def qserve_pack2(w):
    N,K = w.shape
    W_unpack_reorder = (
        w.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
        .permute(0, 4, 3, 6, 7, 1, 5, 2)
        .contiguous()
    )
    return W_unpack_reorder
    W_unpack_repacked = (W_unpack_reorder[..., 1] << 4) + W_unpack_reorder[..., 0]
    W_unpack_repacked = W_unpack_repacked.reshape(N, K // 2)

@torch.no_grad()
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
    # packed_weight = (
    #     packed_weight[..., 0]
    #     | (packed_weight[..., 1] << 4)
    #     | (packed_weight[..., 2] << 8)
    #     | (packed_weight[..., 3] << 12)
    # )
    # reshape to (N // 4, K), Int16 format
    packed_weight = packed_weight.reshape(N // interleave, -1)
    return packed_weight.contiguous()
    return packed_weight.to(torch.int16).contiguous()

def _get_perms_w4a8(group_size = 128):
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                4 * (i % 4),
                4 * (i % 4) + 1,
                4 * (i % 4) + 2,
                4 * (i % 4) + 3,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    if group_size == -1:
        interleave = np.array([4, 0, 5, 1, 6, 2, 7, 3])
    else:
        interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm.view(-1, 8)
    return perm

def _get_perms_w4a16():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm.view(-1, 8)
    return perm



# w = torch.rand((512, 3584//2), device='cuda').to(torch.int8)

# for _ in range(20):
#     qserve_pack2(w)

# n = 100
# with timer(n=n):
#     for _ in range(n):
#         qserve_pack2(w)
torch.set_printoptions(profile='full')
# w = torch.tensor([i for i in range(3584*3584)]).view(3584,3584)
# # print(pack_int_awq(w))
# w = qserve_pack2(w).view(-1,32)
# print(w[0:5, :])
# w = w[:,0].view(-1, 32)


perm1 = _get_perms_w4a8()
perm2 = _get_perms_w4a16()

print(perm1.view(-1, 16*8)[0].view(-1,8)[:, 0:2])
print('=' * 30)
print(perm2.view(-1, 16*8)[0].view(-1,8)[:, 0:2])

exit(0)

for i in range(8):
    perm1_s = perm1.view(-1, 16*8)[i].msort()
    perm2_s = perm2.view(-1, 16*8)[i].msort()
    assert (perm1_s-perm2_s).sum() == 0, print(perm1_s, perm2_s)
print('=' * 30)