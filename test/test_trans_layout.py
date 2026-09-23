import torch
import kairos_cuda_accel

# ------------------------------------------------------------------
# 2. 参考实现（PyTorch）
# ------------------------------------------------------------------
def pack_int_qserve(unpacked_qweight: torch.Tensor):
    N, K = unpacked_qweight.shape
    packed = (unpacked_qweight.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
              .permute(0, 4, 3, 6, 5, 2, 7, 1).contiguous())
    packed = packed.to(torch.int8)
    packed = (packed[..., 1] << 4) + packed[..., 0]
    return packed.reshape(N, K // 2)

def pack_int_awq(unpacked_qweight: torch.Tensor):
    N, K = unpacked_qweight.shape
    packed = (unpacked_qweight.reshape(N // 4, 4, K // 64, 2, 4, 4, 2)
              .permute(0, 2, 1, 3, 5, 6, 4).contiguous())
    packed = packed.to(torch.int16)
    packed = (packed[..., 0] | (packed[..., 1] << 4)
            | (packed[..., 2] << 8) | (packed[..., 3] << 12))
    return packed.reshape(N // 4, K)

def trans_layout_pytorch(awq_packed: torch.Tensor, N: int, K: int):
    """纯 PyTorch 实现（作为参考）"""
    awq_4bit = awq_packed.reshape(N // 4, K // 64, 4, 2, 4, 2)
    unpacked = torch.stack([
        awq_4bit & 0xF, (awq_4bit >> 4) & 0xF,
        (awq_4bit >> 8) & 0xF, (awq_4bit >> 12) & 0xF,
    ], dim=-1).to(torch.int8)
    unpacked = unpacked.permute(0, 2, 1, 3, 6, 4, 5)
    unpacked = unpacked.reshape(N // 32, 2, 2, 8, K // 32, 2, 4, 4)
    unpacked = unpacked.permute(0, 4, 3, 6, 5, 2, 7, 1)
    packed = (unpacked[..., 1] << 4) + unpacked[..., 0]
    return packed.reshape(N, K // 2)


# ------------------------------------------------------------------
# 3. 正确性验证
# ------------------------------------------------------------------
print("=" * 60)
print("Correctness Check")
print("=" * 60)

device = "cuda"
for N, K in [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)]:
    A = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    awq = pack_int_awq(A)
    ref = pack_int_qserve(A)
    out = kairos_cuda_accel.trans_layout_c16_to_c8_awq(awq.to(torch.int16), N, K)
    match = torch.equal(out, ref)
    print(f"  N={N:5} K={K:5}  ->  match={match}")

    # print()


    # ------------------------------------------------------------------
    # 4. A100 性能基准测试
    # ------------------------------------------------------------------
    # print("=" * 60)
    # print(f"Benchmark on {torch.cuda.get_device_name()}")
    # print("=" * 60)

    # N, K = 4096, 4096
    A = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)
    awq = pack_int_awq(A)
    ref = pack_int_qserve(A)

    # Warmup
    for _ in range(20):
        _ = kairos_cuda_accel.trans_layout_c16_to_c8_awq(awq, N, K)
    torch.cuda.synchronize()

    # CUDA kernel benchmark
    iters = 500
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        out_cuda = kairos_cuda_accel.trans_layout_c16_to_c8_awq(awq, N, K)
    end.record()
    torch.cuda.synchronize()
    t_cuda = start.elapsed_time(end) / iters

    # PyTorch reference benchmark
    start.record()
    for _ in range(iters):
        out_pt = trans_layout_pytorch(awq, N, K)
    end.record()
    torch.cuda.synchronize()
    t_pt = start.elapsed_time(end) / iters

    print(f"  CUDA/PTX : {t_cuda:>8.3f} ms/iter")
    print(f"  PyTorch  : {t_pt:>8.3f} ms/iter")
    print(f"  Speedup  : {t_pt / t_cuda:.2f}x")

# Memory bandwidth estimate
bytes_moved = awq.numel() * 2 + out_cuda.numel() * 1  # read int16 + write int8
bandwidth_gbps = (bytes_moved / 1e9) / (t_cuda / 1e3)
print(f"  Effective: {bandwidth_gbps:.1f} GB/s  (A100 peak ~2 TB/s)")