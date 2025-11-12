import sys
import numpy as np
import torch
import torch.nn as nn

from thirdparty.AWQ.awq_method import quant_weight_awq, real_quantize_tensor
from thirdparty.AWQ.w4a16_linear import W4A16Linear
from utils.perf_eval import timer

import kernels.edq_cuda_accel as edq_cuda_accel_test
import edq_cuda_accel
import awq_backend

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

M, N, K = 4096, 18944, 3584 # gate_proj, up_proj
M, N, K = 1024, 3584, 18944 # down_proj
M, N, K = 1024, 3584, 3584 # q_proj, o_proj
M, N, K = 1024, 512, 3584 # kv_proj
# M, N, K = 1, 256, 512
M = 1

# M = 1024 * 8
# M = 1024 * 16
# M = 256
# M = 64
def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def awq_forward(input, qweight, scales, scaled_zeros, N, K, group_size=128):
    batch_token_len = input.numel() // input.shape[-1]
    if batch_token_len < 8:
        out = awq_backend.gemv_forward_cuda_new(
            input, qweight, scales, scaled_zeros,
            batch_token_len, N, K, group_size)
    else:
        out = awq_backend.gemm_forward_cuda_new(
            input, qweight, scales, scaled_zeros)
    return out

@torch.no_grad()
def test_awq(A, B, N, K):
    C = torch.matmul(A, B.T)
    i4w, i4s, i4z = quant_weight_awq(B.clone(), 128)
    C_awq = awq_backend.gemm_forward_cuda_new(A, i4w, i4s, i4z)

    torch.cuda.synchronize()
    rmse_awq = relative_error(C_awq, C)
    print(f"rmse of AWQ: {rmse_awq}")

    torch.cuda.synchronize()

    for _ in range(50):
        C_awq = awq_forward(A, i4w, i4s, i4z, N, K)
    torch.cuda.synchronize()
    with timer("AWQ w4a16", 1000):
        for _ in range(1000):
            C_awq = awq_forward(A, i4w, i4s, i4z, N, K)

@torch.no_grad()
def test_gemv(A, B):
    C = torch.matmul(A, B.T)
    linear = nn.Linear(K, N)
    linear.weight.data = B.clone()
    linear.bias = None

    linear = W4A16Linear.from_module(linear, device, torch.float16)

    C_ml = torch.zeros(
        A.shape[:-1] + (linear.scales.shape[1],), dtype=linear.return_dtype, device=A.device
    )

    # print("A")
    # print(A[0,:129])
    # print("Quant Weight")
    # print(linear.qweight.shape)
    # print(linear.qweight[:9,:4])
    # print("Scale")
    # print(linear.scales[:2,:16])
    
    # print("C w0")
    # print(A[0,0:128] @ B[0,0:128])
    # print(A[0,128*1:128*2] @ B[0,128*1:128*2])
    # print(A[0,128*2:128*3] @ B[0,128*2:128*3])
    # print(A[0,128*3:128*4] @ B[0,128*3:128*4])

    edq_cuda_accel_test.w4a16_wa_gemv_cuda(
        A.view((-1, A.shape[-1])),
        linear.qweight,
        linear.scales,
        linear.scaled_zeros,
        C_ml.view((-1, C_ml.shape[-1])),
    )
    # C_ml = linear.forward(A)
    torch.cuda.synchronize()
    rmse_ml = relative_error(C_ml, C)
    print(f"rmse of C marlin: {rmse_ml}")
    # print("C marlin gemv")
    # print(C_ml[0,:128])
    # print("C pytorch")
    # print(C[0,:128])

    for _ in range(50):
        edq_cuda_accel_test.w4a16_wa_gemv_cuda(
            A.view((-1, A.shape[-1])),
            linear.qweight,
            linear.scales,
            linear.scaled_zeros,
            C_ml.view((-1, C_ml.shape[-1]))
        )
        # C_ml = linear.forward(A)
    torch.cuda.synchronize()
    with timer("Marlin Linear", 1000):
        for _ in range(1000):
            edq_cuda_accel_test.w4a16_wa_gemv_cuda(
                A.view((-1, A.shape[-1])),
                linear.qweight,
                linear.scales,
                linear.scaled_zeros,
                C_ml.view((-1, C_ml.shape[-1])),
            )
            # C_ml = linear.forward(A)


if __name__ == "__main__":
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)

    test_gemv(A, B)

