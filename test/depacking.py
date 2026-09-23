import torch
import kairos_cuda_accel

N = 1024; K = 1024
group_size = 128


def pack_int_awq(unpacked_qweight):
    N, K = unpacked_qweight.shape
    packed_weight = (
        unpacked_qweight.reshape(N // 4, 4, K // 64, 2, 4, 4, 2)
        .permute(0, 2, 1, 3, 5, 6, 4)
        .contiguous())
    # return packed_weight.reshape(N,K)
    packed_weight = packed_weight.to(torch.int16)
    packed_weight = (packed_weight[..., 0] | (packed_weight[..., 1] << 4)
                | (packed_weight[..., 2] << 8) | (packed_weight[..., 3] << 12))
    packed_weight = packed_weight.reshape(N // 4, K)
    return packed_weight


def unpack_int_awq(packed_qweight: torch.Tensor) -> torch.Tensor:
    # assert packed_qweight.dim() == 2, "packed_qweight must be 2D"
    # assert packed_qweight.dtype == torch.int16, "packed_qweight must be int16"
    N_out, K = packed_qweight.shape
    N = N_out * 4
    # assert K % 64 == 0, "K must be divisible by 64"
    w0 = packed_qweight & 0xF
    w1 = (packed_qweight >> 4) & 0xF
    w2 = (packed_qweight >> 8) & 0xF
    w3 = (packed_qweight >> 12) & 0xF

    nibbles = torch.stack([w0, w1, w2, w3], dim=-1)
    nibbles = nibbles.reshape(N_out, K // 64, 4, 2, 4, 2, 4)
    inv_perm = [0, 2, 1, 3, 6, 4, 5]
    nibbles = nibbles.permute(*inv_perm)  # → [n0, n1, k0, d0, d1, d2, d3]
    unpacked = nibbles.reshape(N, K)
    return unpacked

orig = torch.randint(0, 16, (N, K), dtype=torch.int16)
packed = pack_int_awq(orig)
restored = unpack_int_awq(packed)
