import sys
import numpy as np
import torch
import torch.nn as nn

from thirdparty.AWQ.awq_method import quant_weight_awq, real_quantize_tensor
from thirdparty.AWQ.w4a16_linear import W4A16Linear
from utils.perf_eval import timer

import kernels.edq_cuda_accel as edq_cuda_accel_marlin
import edq_cuda_accel
import awq_backend

sys.path.append("/home/chenjw/Learn/Quant/artifact/projects/")
import cuda as cuda
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

M, N, K = 4096, 18944, 3584 # gate_proj, up_proj
M, N, K = 1024, 3584, 18944 # down_proj
M, N, K = 1024, 3584, 3584 # q_proj, o_proj
M, N, K = 1024, 512, 3584 # kv_proj
# M, N, K = 1024, 256, 128

# M = 1024 * 10
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

# Marlin Symmetry Quantization, group_size=128
@torch.no_grad()
def quant_weight(B, groupsize=128):
    '''
    return:
        w: quantization weight, shape = (k, n), dtype = torch.int32
        s: scales, shape = (k // groupsize, n), dtype = torch.float16
    '''
    maxq = 2 ** 4 - 1
    w = B.t()
    k = w.shape[0]
    n = w.shape[1]
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        w = reshape(w).contiguous()
    s = s.reshape((-1, n)).contiguous()
    return w, s

# Marlin ASymmetry Quantization, refer to AWQ
@torch.no_grad()
def quant_weight_asym(B, groupsize=128):
    '''
    return:
        w: quantization weight, shape = (k, n), dtype = torch.int32
        s: scales, shape = (k // groupsize, n), dtype = torch.float16
        z: zero point, shape = (k // groupsize, n), dtype = torch.float16
    '''
    max_int = 2 ** 4 - 1
    min_int = 0
    w = B.t()
    k = w.shape[0]
    n = w.shape[1]
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))

    max_val = w.amax(dim=0, keepdim=True)
    min_val = w.amin(dim=0, keepdim=True)
    s = (max_val - min_val).clamp(min=1e-5) / max_int
    z = (-torch.round(min_val / s)).clamp_(min_int, max_int)
    assert torch.isnan(s).sum() == 0
    assert torch.isnan(w).sum() == 0
    (w.div_(s).round_().add_(z)).clamp_(min_int, max_int)
    assert torch.isnan(w).sum() == 0
    w = w.to(dtype=torch.int32)

    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n)).contiguous()
            return w
        w = reshape(w).contiguous()
    s = s.reshape((-1, n)).contiguous()
    z = z.reshape((-1, n)).contiguous()
    return w, s, z


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
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


class MarlinLinear(nn.Module):
    """PyTorch compatible Marlin Linear; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, device, groupsize=-1):
        """Create an empty Marlin Linear.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        tile = 16
        pack_num = 32 // 4
        self.register_buffer(
            "B", torch.empty((self.k // tile, self.n * tile // pack_num), dtype=torch.int, device=device)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.float16, device=device)
        )
        # scale zero point
        self.register_buffer(
            "sz", torch.empty((self.k // groupsize, self.n), dtype=torch.float16, device=device)
        )
        
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(self.n // 128 * 16, dtype=torch.int, device=device),
            persistent=False,
        )

    def forward(self, A):
        C = torch.zeros(
            A.shape[:-1] + (self.s.shape[1],), dtype=self.return_dtype, device=A.device
        )

        cuda.mul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
            self.workspace,
            -1, -1, -1, 16
        )
        return C
    
    def forward_asym(self, A):
        C = torch.zeros(
            A.shape[:-1] + (self.s.shape[1],), dtype=self.return_dtype, device=A.device
        )

        edq_cuda_accel_marlin.w4a16_wa_gemm_cuda(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
            self.sz,
            self.workspace,
            -1, -1, -1, 16
        )
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError('Only `torch.half` weights are supported.')
        tile = 16
        maxq = 2 ** 4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
    
    def pack_only(self, weight, scales):
        '''
            Pack Weight and Scale in Marlin format
            weight: (k, n) int4 store in int32
            scales: (k // groupsize, n)

        '''
        if weight.dtype != torch.int:
            raise ValueError('Only `torch.int` weights are supported.')
        tile = 16
        pack_num = 32 // 4
        s = scales.clone()
        w = weight.clone()

        if self.groupsize != -1:
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()

        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        w = w.reshape((-1, _perm.numel()))[:, _perm].reshape(w.shape)

        q = torch.zeros((w.shape[0], w.shape[1] // pack_num), dtype=torch.int, device=w.device)
        for i in range(8):
            q = torch.bitwise_or(q, w[:, i::8] << 4 * i)
        q = q.to(torch.int)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)

    def pack_only_asym(self, weight, scales, zero_point):
        '''
            Pack Weight and Scale in Marlin format
            weight: (k, n) int4 store in int32
            scales: (k // groupsize, n)
            zero_point: (k // groupsize, n)
        '''
        if weight.dtype != torch.int:
            raise ValueError('Only `torch.int` weights are supported.')
        tile = 16
        pack_num = 32 // 4
        s = scales.clone()
        z = zero_point.clone()
        w = weight.clone()

        if self.groupsize != -1:
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
            z = z.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
            z = z.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        
        s = s.reshape((-1, self.n)).contiguous()
        z = z.reshape((-1, self.n)).contiguous()

        sz = torch.zeros_like(z)
        sz[:, :] = -(s[:, :] * z.to(torch.int32).to(torch.float32)).to(s.dtype)
        

        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        w = w.reshape((-1, _perm.numel()))[:, _perm].reshape(w.shape)

        q = torch.zeros((w.shape[0], w.shape[1] // pack_num), dtype=torch.int, device=w.device)
        for i in range(8):
            q = torch.bitwise_or(q, w[:, i::8] << 4 * i)
        q = q.to(torch.int)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)
        self.sz[:, :] = sz.to(self.sz.device)
    
    @classmethod
    def from_module(cls, linear, device, return_dtype, groupsize=128):
        q_linear = cls(
            linear.in_features, linear.out_features,
            device, groupsize=groupsize
            )
        q_linear.return_dtype = return_dtype

        # Dequantization
        w, s = quant_weight(linear.weight.data.to(torch.float16), groupsize)
        
        # Pack
        q_linear.pack_only(w, s)

        return q_linear
    
    @classmethod
    def from_module_asym(cls, linear, device, return_dtype, groupsize=128):
        q_linear = cls(
            linear.in_features, linear.out_features,
            device, groupsize=groupsize
            )
        q_linear.return_dtype = return_dtype

        # Dequantization
        # w, s, z = quant_weight_asym(linear.weight.data.to(torch.float16), groupsize)
        w, s, z = real_quantize_tensor(linear.weight.data.to(torch.float16), 4, groupsize)
        
        # Pack
        # q_linear.pack_only_asym(w, s, z)
        q_linear.pack_only_asym(w.T, s.T, z.T)

        return q_linear



def gen_quant4(B, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = B.t()
    ref = B.clone()
    k = w.shape[0]
    n = w.shape[1]
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((k, n))
            return w
        w = reshape(w).contiguous()
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(k, n)
    linear.weight.data = ref
    # Workaround to test some special cases that are forbidden by the API
    layer = MarlinLinear(k, n, device=device, groupsize=groupsize)
    if groupsize == -1:
        groupsize = k
    layer.k = k
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((k // 16, n * 16 // 8), dtype=torch.int, device=device)
    layer.s = torch.empty((k // groupsize, n), dtype=torch.half, device=device)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return q, s



if __name__ == "__main__":
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C = torch.matmul(A, B.T)
    
    # Marlin
    B_m, s = gen_quant4(B, groupsize=128)

    C_m = torch.zeros((M, N), device=device, dtype=dtype)
    workspace = torch.zeros(N // 128 * 16, device=device, dtype=torch.int)

    cuda.mul(A, B_m, C_m, s, workspace, -1, -1, -1, 16)

    torch.cuda.synchronize()
    # mse_marlin = torch.nn.functional.mse_loss(C, C)
    # rmse_marlin = (C_m - C).pow(2).mean().sqrt().item()
    # print(f"rmse of marlin: {rmse_marlin}")
    # assert(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)) < 0.001)


    # AWQ
    # print(B_ref.shape)
    i4w, i4s, i4z = quant_weight_awq(B.clone(), 128)
    C_awq = awq_backend.gemm_forward_cuda_new(A, i4w, i4s, i4z)

    torch.cuda.synchronize()
    # mse_awq = torch.nn.functional.mse_loss(C_awq, C_ref)
    # rmse_awq = (C_awq - C).float().pow(2).mean().sqrt().item()
    rmse_awq = relative_error(C_awq, C)
    print(f"rmse of AWQ: {rmse_awq}")

    # Marlin + AWQ

    linear = nn.Linear(K, N)
    linear.weight.data = B.clone()

    linear = MarlinLinear.from_module_asym(linear, device, torch.float16, 128)

    # C_model = linear.forward_asym(A)
    # torch.cuda.synchronize()
    # rmse_model = (C_model - C).pow(2).mean().sqrt().item()
    # print(f"rmse of C model: {rmse_model}")

    # Marlin Linear
    linear = nn.Linear(K, N)
    linear.weight.data = B.clone()
    linear.bias = None

    linear_2 = W4A16Linear.from_module(linear, device, torch.float16)

    C_ml = linear_2.forward(A)
    torch.cuda.synchronize()
    # rmse_ml = (C_ml - C).float().pow(2).mean().sqrt().item()
    rmse_ml = relative_error(C_ml, C)
    print(f"rmse of C marlin: {rmse_ml}")


    # test speed

    # for _ in range(50):
    #     cuda.mul(A, B_m, C_m, s, workspace, -1, -1, -1, 16)
    # with timer("Marlin w4a16", 1000):
    #     for _ in range(1000):
    #         cuda.mul(A, B_m, C_m, s, workspace, -1, -1, -1, 16)

    for _ in range(50):
        C_awq = awq_forward(A, i4w, i4s, i4z, N, K)
        # C_awq = awq_backend.gemv_forward_cuda_new(A, i4w, i4s, i4z, M, N, K, 128)
    with timer("AWQ w4a16", 1000):
        for _ in range(1000):
            C_awq = awq_forward(A, i4w, i4s, i4z, N, K)
            # C_awq = awq_backend.gemv_forward_cuda_new(A, i4w, i4s, i4z, M, N, K, 128)

    # for _ in range(50):
    #     C_model = linear.forward_asym(A)
    # with timer("Marlin + AWQ", 1000):
    #     for _ in range(1000):
    #         C_model = linear.forward_asym(A)

    for _ in range(50):
        C_ml = linear_2.forward(A)
    with timer("Marlin Linear", 1000):
        for _ in range(1000):
            C_ml = linear_2.forward(A)



    # mw, ms, mz = quant_weight_asym(B.clone(), 128)
    # aww, aws, awz = real_quantize_tensor(B.clone(), 4, 128)
    # print(mw)
    # print(mw.shape)
    # print(aww)
    # print(aww.shape)
    # print(ms)
    # print(ms.shape)
    # print(aws)
    # print(aws.shape)
    # print(mz)
    # print(mz.shape)
    # print(awz)
    # print(awz.shape)
    # print((mw - aww.T).to(torch.float).pow(2).mean().sqrt().item())
    # print((ms - aws.T).pow(2).mean().sqrt().item())
    # print((mz - awz.T).pow(2).mean().sqrt().item())

    # 要先注释掉对布局的调整
    # asym_linear = MarlinLinear.from_module_asym(linear, device, torch.float16, 128)
    # print("s")
    # print(asym_linear.s)
    # print("sz")
    # print(asym_linear.sz)
    # print("i4s")
    # print(i4s[:28])
    # print("i4z")
    # print(i4z[:28])
    # print((asym_linear.s - i4s[:28]).pow(2).mean().sqrt().item())
    # print((asym_linear.sz - i4z[:28]).pow(2).mean().sqrt().item())

