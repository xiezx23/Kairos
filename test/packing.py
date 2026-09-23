import torch
import kairos_cuda_accel

group_size = 128

def pack_int_qserve(unpacked_qweight: torch.Tensor):
    N, K = unpacked_qweight.shape
    packed_weight = (
        unpacked_qweight.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
        .permute(0, 4, 3, 6, 5, 2, 7, 1)
        .contiguous())
    packed_weight = packed_weight.to(torch.int8)
    packed_weight = (packed_weight[..., 1] << 4) + packed_weight[..., 0]
    packed_weight = packed_weight.reshape(N, K // 2)
    return packed_weight

# def pack_int_qserve(unpacked_qweight: torch.tensor):
#     N, K = unpacked_qweight.shape
#     packed_weight = (
#         unpacked_qweight.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
#         .permute(0, 4, 3, 6, 5, 2, 7, 1)
#         .contiguous())
#     packed_weight = packed_weight.to(torch.int8)
#     packed_weight = (packed_weight[..., 1] << 4) + packed_weight[..., 0]
#     packed_weight = packed_weight.reshape(N, K // 2)
#     return packed_weight

# def pack_int_awq(unpacked_qweight: torch.Tensor):
#     N, K = unpacked_qweight.shape
#     packed_weight = (
#         unpacked_qweight.reshape(N // 4, 4, K // 64, 2, 4, 4, 2)
#         .permute(0, 2, 1, 3, 5, 6, 4)
#         .contiguous())
#     packed_weight = packed_weight.to(torch.int16)
#     packed_weight = (packed_weight[..., 0] | (packed_weight[..., 1] << 4)
#                 | (packed_weight[..., 2] << 8) | (packed_weight[..., 3] << 12))
#     packed_weight = packed_weight.reshape(N // 4, K)
#     return packed_weight

def pack_int_awq(unpacked_qweight):
    N, K = unpacked_qweight.shape
    packed_weight = (
        unpacked_qweight.reshape(N // 4, 4, K // 64, 2, 4, 4, 2)
        .permute(0, 2, 1, 3, 5, 6, 4)
        .contiguous())
    packed_weight = packed_weight.to(torch.int16)
    packed_weight = (packed_weight[..., 0] | (packed_weight[..., 1] << 4)
                | (packed_weight[..., 2] << 8) | (packed_weight[..., 3] << 12))
    packed_weight = packed_weight.reshape(N // 4, K)
    return packed_weight


import numpy as np
# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
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
    return perm


_perm = _get_perms()

def pack_int_marlin(unpacked_qweight):
    outfeatures, infeatures = unpacked_qweight.shape
    w = unpacked_qweight.T.to(torch.int32); tile = 16
    w = w.reshape((infeatures // tile, tile, outfeatures // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((infeatures // tile, outfeatures * tile))
    res = w
    res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32)).to(w.device)
    return q

@torch.compile
@torch.no_grad
def trans_layout(awq_packed_qweight: torch.Tensor) -> torch.Tensor:
    N_div4, K = awq_packed_qweight.shape
    N = N_div4 * 4
    # Step 1: 将 AWQ 的 int16 解包为 4 个 int4 nibble
    # AWQ 内部逻辑布局: (n0, k0, n1, k1, k3, k4) 每个位置存 4 个 k2 值
    awq_4bit = awq_packed_qweight.reshape(N // 4, K // 64, 4, 2, 4, 2)
    unpacked = torch.stack([
        awq_4bit & 0xF,
        (awq_4bit >> 4) & 0xF,
        (awq_4bit >> 8) & 0xF,
        (awq_4bit >> 12) & 0xF,
    ], dim=-1).to(torch.int8)          # shape: (N//4, K//64, 4, 2, 4, 2, 4)

    # Step 2: Permute 到逻辑维度顺序 (n0, n1, k0, k1, k2, k3, k4)
    unpacked = unpacked.permute(0, 2, 1, 3, 6, 4, 5)
    # shape: (N//4, 4, K//64, 2, 4, 4, 2)

    # Step 3: Reshape 为 QServe 的 tile 分解
    #   (N//4, 4)      -> (N//32, 2, 2, 8)   对应 QServe 的 (n0_q, n1_q, n2_q, n3_q)
    #   (K//64, 2)     -> (K//32)             对应 QServe 的 k0_q
    #   (4, 4, 2)      -> (2, 4, 4)           对应 QServe 的 (k1_q, k2_q, k3_q)
    unpacked = unpacked.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
    # shape: (N//32, n1_q, n2_q, n3_q, k0_q, k1_q, k2_q, k3_q)

    # Step 4: Permute 到 QServe 打包前的维度顺序
    # Target: (n0_q, k0_q, n3_q, k2_q, k1_q, n2_q, k3_q, n1_q)
    unpacked = unpacked.permute(0, 4, 3, 6, 5, 2, 7, 1)
    # shape: (N//32, K//32, 8, 4, 2, 2, 4, 2)

    # Step 5: 将最后两个维度 (n1_q=0/1) 打包成一个 int8
    # n1_q=1 放高位，n1_q=0 放低位，与 pack_int_qserve 一致
    packed = (unpacked[..., 1] << 4) + unpacked[..., 0]
    # shape: (N//32, K//32, 8, 4, 2, 2, 4)

    # Step 6: 展平为 QServe 最终输出形状
    return packed.reshape(N, K // 2)



from utils.perf_eval import timer

shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)] # Llama-3 8B

for w in shape_list:
    N = w[0]; K = w[1]
    print(f'N={N} K={K}')
    A = torch.randint(0, 16, (N, K), dtype=torch.int16, device='cuda')
    input = torch.randint(0,16, (K//16, N*2), dtype=torch.int32, device='cuda')
    output = torch.empty((K//16, N*2),  dtype=torch.int32, device='cuda')

    awq_packed = pack_int_awq(A)
    qserve_ref = pack_int_qserve(A)

    for _ in range(200):
        # qserve_out = trans_layout(awq_packed)
        _ = kairos_cuda_accel.trans_layout_c16_to_c8_awq(awq_packed, N, K)
        kairos_cuda_accel.trans_layout_c16_to_c8_marlin(output, input)

    n = 500
    # with timer('trans_layout awq->qserve (torch)', n=n):
    #     for _ in range(n):
    #         qserve_out = trans_layout(awq_packed)
            
    with timer('trans_layout awq->qserve (cuda)', n=n):
        for _ in range(n):
            _ = kairos_cuda_accel.trans_layout_c16_to_c8_awq(awq_packed, N, K)

    with timer('trans_layout marlin->qqq (cuda)', n=n):
        for _ in range(n):
            kairos_cuda_accel.trans_layout_c16_to_c8_marlin(output, input)

    print('-'*20)
    # print((qserve_out-qserve_ref).sum())
    # assert torch.equal(qserve_out, qserve_ref)  # 必为 True