import torch

# from thirdparty.AutoGPTQ.w4a16_linear import pack_int_marlin
# from thirdparty.AWQ.awq_method import pack_int_awq
# from thirdparty.Qserve.w4a8_linear import pack_int_qserve

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
    return packed_weight.reshape(N, K)

def pack_int_awq_2(unpacked_qweight, interleave = 4, kstride = 64):
    # unpacked_qweight: [N, K]
    N, K = unpacked_qweight.shape

    packed_weight = unpacked_qweight.reshape(N, K // 32, 32).to(torch.int16)
    packed_weight = packed_weight.reshape(N, K // 32, 4, 4, 2).permute(0, 1, 3, 4, 2)

    # interleaving every four rows
    # N // 4, K // 64, 4, 64
    packed_weight = packed_weight.reshape(
        N // interleave, interleave, K // kstride, kstride
    ).permute(0, 2, 1, 3)
    # Packing -> (N // 4, K // 64, 64)
    return packed_weight.reshape(N, K)

def pack_int_qserve(unpacked_qweight):
    N, K = unpacked_qweight.shape
    W_unpack_reorder = (
        unpacked_qweight.reshape(N // 32, 2,2, 8, K // 32, 2, 4, 4,)
        .permute(0, 4, 3, 6, 5, 2, 7, 1)
        .contiguous())
    return W_unpack_reorder.reshape(N,K)

# n = 64; k = 64
n = 128; k = 128
w = torch.tensor([i for i in range(n*k)]).view(n,k)

qw1 = pack_int_awq(w.clone()).reshape(-1)
qw1_2 = pack_int_awq_2(w.clone()).reshape(-1)
qw2 = pack_int_qserve(w.clone()).reshape(-1)

torch.set_printoptions(profile='full')

# print(qw1[0:4*64].msort())
# print(qw2[0:4*64].msort())

# print(qw1[0:64])
# print(qw2[0:64])

print(torch.equal(qw1, qw1_2))

qw1 = qw1.reshape(-1, 8)
qw1_2 = qw1_2.reshape(-1, 8)
qw2 = qw2.reshape(-1, 8).to(torch.int32)
# print((qw1[0:64]))
# print((qw1_2[0:64]))
print((qw2[0:64]))