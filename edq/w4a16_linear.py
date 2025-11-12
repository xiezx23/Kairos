# Inspired by Marlin.
# Modified by Junwei Chen.

import torch
import numpy as np
from edq.quantization import *

from utils.perf_eval import timer
from utils.global_config import * #device, enable_accel

# Precompute permutations for weight and scale shuffling
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

class W4A16Linear(torch.nn.Module):
    """
        activation  FP16 \\
        weight      FP16 -> quantize -> INT4\\
        O = A @ deq(W) (FP16 GEMM) -> FP16\\
    """
    @torch.no_grad
    def __init__(self, in_features, out_features, bias, device, group_size = 128):
        super().__init__()
        if group_size != 128:
            raise ValueError("Only support group_size 128.")
        if in_features % 128 != 0 or out_features % 256 != 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        self.enable_accel = enable_accel
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size = group_size
        buffer_kwargs = {"device" : device, "dtype": torch.float16}
        tile = 16
        pack_num = 32 // 4
        self.max_par = 16
        self.register_buffer(
            "qweight",  torch.empty(
                (self.in_features // tile, self.out_features * tile // pack_num),
                dtype=torch.int32,
                device=device))
        self.register_buffer(
            "scales",    torch.empty(
                (self.in_features // group_size, self.out_features),
                **buffer_kwargs))
        self.register_buffer(
            "scaled_zeros",  torch.empty(
                (self.in_features // group_size, self.out_features),
                **buffer_kwargs))
        # 128 is currently the minimum `tile_n`, hence it gives the maximum workspace size; 16 is the default `max_par`
        self.register_buffer(
            "workspace",
            torch.zeros(self.out_features // 128 * self.max_par, dtype=torch.int, device=device),
            # persistent=False,
        )
        if bias:
            self.register_buffer(
                "bias", torch.empty((self.out_features), **buffer_kwargs))
        else:
            self.bias = None
    
    def pack(self, weight, scales, zero_point):
        '''
        Pack quantization weight, scales and scaled_zeros in Marlin format\\
        Input:\\
            weight: (in_feat, out_feat) int4 store in int32\\
            scales: (in_feat // group_size, out_feat) in fp16\\
            zero_point: (in_feat // group_size, out_feat) not scaled in fp16\\
        '''
        if weight.dtype != torch.int:
            raise ValueError('Only `torch.int` weights are supported.')
        tile = 16
        s = scales
        z = zero_point
        w = weight

        s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        z = z.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        
        s = s.reshape((-1, self.out_features)).contiguous()
        z = z.reshape((-1, self.out_features)).contiguous()

        sz = torch.zeros_like(z)
        sz[:, :] = -(s[:, :] * z.to(torch.int32).to(torch.float32)).to(s.dtype)
        
        w = w.reshape((self.in_features // tile, tile, self.out_features // tile, tile))
        w = w.permute((0, 2, 1, 3))
        # w = w.reshape((self.in_features // tile, self.out_features * tile))
        # print(w.shape)
        w = w.reshape((-1, _perm.numel()))[:, _perm].reshape((self.in_features // tile, self.out_features * tile))
        # print(w.shape)

        q = torch.zeros((w.shape[0], w.shape[1] // 8), dtype=torch.int, device=w.device)
        for i in range(8):
            q = torch.bitwise_or(q, w[:, i::8] << 4 * i)
        q = q.to(torch.int)
        self.qweight[:, :] = q.to(self.qweight.device)
        self.scales[:, :] = s.to(self.scales.device)
        self.scaled_zeros[:, :] = sz.to(self.scaled_zeros.device)
    

    @torch.compile
    def unpack(self, group_size=128):
        """
        Unpack weight, scales, scaled_zeros in Marlin format\\
        
        Returns:\\
            weight: (out_feat, in_feat) int4 in int32\\
            scales: (out_feat, in_feat // groupsize)\\
            scaled_zeros: (out_feat, in_feat // group_size)\\
        """
        tile = 16
        # unpack int32->int4
        # qweight shape: (self.in_features//16, self.out_features*16//8)
        q = self.qweight
        shifts = torch.arange(0, 32, 4, device=q.device, dtype=torch.int32)
        extracted = (q.unsqueeze(-1) >> shifts) & 0xF
        res = extracted.reshape(q.size(0), -1)
        
        # 创建逆向排列索引
        inv_perm = torch.zeros_like(_perm)
        inv_perm[_perm] = torch.arange(_perm.numel())
        
        # 应用逆向排列
        res = res.reshape((-1, _perm.numel()))[:, inv_perm].reshape(res.shape)
        
        # (self.in_features//16, self.out_features*16) -> (self.in_features//16, self.out_features//16, 16, 16) -> (self.in_features//16, 16, self.out_features//16, 16)
        # -> (self.in_features, self.out_features) -> (self.out_features, self.in_features)
        res = res.reshape((self.in_features // tile, self.out_features // tile, tile, tile))
        res = res.permute((0, 2, 1, 3))  # 逆向之前的 permute(0, 2, 1, 3)
        res = res.reshape((self.in_features, self.out_features))
        weight = res.t()  # 从 (K, N) 转回 (N, K)
        
        # 创建逆向scale和zp的排列索引
        inv_scale_perm = [0] * len(_scale_perm)
        for i, p in enumerate(_scale_perm):
            inv_scale_perm[p] = i
        
        # unpack scales
        s = self.scales.clone()
        s = s.reshape((-1, len(_scale_perm)))[:, inv_scale_perm]
        s = s.reshape((self.in_features // group_size, self.out_features)).t()
        
        # unpack scale_zero
        z = self.scaled_zeros.clone()
        z = z.reshape((-1, len(_scale_perm)))[:, inv_scale_perm]
        z = z.reshape((self.in_features // group_size, self.out_features)).t()
        
        return weight, s, z

    @classmethod
    def from_module(cls, linear, device, return_dtype=torch.float16, init_only=False, name = ''):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device, group_size=128)
        # assert not(init_only and not q_linear.enable_accel)
        q_linear.return_dtype = return_dtype
        if init_only: 
            return q_linear
        
        qw, qs, qzp = real_quantize_tensor_edq(linear.weight.data.clone().to(torch.float16), 4, 128)
        
        q_linear.pack(qw.T, qs.T, qzp.T)

        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half().contiguous()
        return q_linear

    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        input = input.to(torch.float16)
        if self.enable_accel:
            batch_token_len = input.numel() // input.shape[-1]
            out = torch.empty(
                input.shape[:-1] + (self.scales.shape[1],), dtype=self.return_dtype, device=input.device
            )
            # edq_cuda_accel.w4a16_wa_gemm_cuda(
            #     input.view((-1, input.shape[-1])),
            #     self.qweight,
            #     out.view((-1, out.shape[-1])),
            #     self.scales,
            #     self.scaled_zeros,
            #     self.workspace,
            #     -1, -1, -1, self.max_par
            # )
            # print(out.shape)
            if batch_token_len < 4:
                edq_cuda_accel.w4a16_wa_gemv_cuda(
                    input.view((-1, input.shape[-1])),
                    self.qweight,
                    self.scales,
                    self.scaled_zeros,
                    out.view((-1, out.shape[-1])),
                )
            else:
                edq_cuda_accel.w4a16_wa_gemm_cuda(
                    input.view((-1, input.shape[-1])),
                    self.qweight,
                    out.view((-1, out.shape[-1])),
                    self.scales,
                    self.scaled_zeros,
                    self.workspace,
                    -1, -1, -1, 16
                )
        else:
            input = input.reshape(-1, input_shape[-1]).to(torch.float16)
            unpack_weight, unpack_scales, unpack_scaled_zeros = self.unpack(self.group_size)
            unpack_weight = dequantize_tensor_edq(unpack_weight, unpack_scales, unpack_scaled_zeros, self.group_size)
            out = input @ unpack_weight.T
        if len(input_shape) == 3:
            out = out.view(input_shape[0], input_shape[1], -1)
        if self.bias is not None:
            out.add_(self.bias)
        return out.to(self.return_dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"



##################################################
#                    TEST CODE                   #    
##################################################    
@torch.no_grad()
@torch.compile()
def pack_test(w):
    tile = 16
    in_features = w.shape[0]
    out_features = w.shape[1]

    w = w.reshape((in_features // tile, tile, out_features // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((-1, _perm.numel()))[:, _perm].reshape((in_features // tile, out_features * tile))

    q = torch.zeros((w.shape[0], w.shape[1] // 8), dtype=torch.int, device=w.device)
    for i in range(8):
        q = torch.bitwise_or(q, w[:, i::8] << 4 * i)
    q = q.to(torch.int)
  
if __name__ == "__main__":
    def relative_error(target, current):
        return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    M, N, K = 30, 18944, 3584 # gate_proj, up_proj
    # M, N, K = 1024, 3584, 18944 # down_proj
    # M, N, K = 1024, 3584, 3584 # q_proj, o_proj
    M, N, K = 1024, 512, 3584 # kv_proj
    M = 32 + 63

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    # C = torch.matmul(A, B.T)

    linear = torch.nn.Linear(K, N, dtype=torch.float16).cuda()
    linear.weight.data = B.clone()
    linear.bias = None

    linear = W4A16Linear.from_module(linear, device, torch.float16)


    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    C = torch.matmul(A, B.T)
    torch.cuda.synchronize()
    C_ref = linear.forward(A)
    torch.cuda.synchronize()
    loss_ref = relative_error(C, C_ref).item()
    # print(C_ref)
    for _ in range(20):
        torch.cuda.synchronize()
        C_ml = linear.forward(A)
        torch.cuda.synchronize()
        # print(torch.nn.functional.mse_loss(C_ml, C).item())
        loss = relative_error(C, C_ml).item()
        if abs(loss_ref - loss) < 1e-7:
            print(loss)
            # print(C_ml)
            break

    # cnt = 0
    # for m in range(1, M + 1):
    #     A = torch.randn(m, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    #     C = torch.matmul(A, B.T)
    #     C_ml = linear.forward(A)
    #     torch.cuda.synchronize()
    #     # loss_first = relative_error(C, C_ml)
    #     loss_first = torch.nn.functional.mse_loss(C_ml, C).item()
        
    #     for _ in range(10):
    #         C_ml = linear.forward(A)
    #         torch.cuda.synchronize()
    #         # loss = relative_error(C, C_ml)
    #         loss = torch.nn.functional.mse_loss(C_ml, C).item()
    #         diff = (loss_first - loss).abs()
    #         if diff > 1e-7:
    #             print(f"rmse of C marlin first : {loss_first}")
    #             print(f"rmse of C marlin repeat: {loss}")
    #             print(f"M={m}, diff is {diff}")
    #             cnt += 1
    #             break
    # print(f"rate = {cnt} / {M}")
    # print("C")
    # print(C)
    # print("C marlin")
    # print(C_ml)

    for _ in range(50):
        C_ml = linear.forward(A)
    with timer("Marlin Linear Forward", 1000):
        for _ in range(1000):
            C_ml = linear.forward(A)

    # B_t = B.T.to(torch.int32).clone()
    # for _ in range(50):
    #     pack_test(B_t)
    # with timer("Marlin Linear Pack w torch", 1000):
    #     for _ in range(1000):
    #         pack_test(B_t)

    # linear2 = torch.nn.Linear(K, N)
    # linear2.weight.data = B.clone()
    # linear2.bias = None
    # for _ in range(50):
    #     C_ml = linear.from_module(linear2, device, torch.float16)
    # with timer("Marlin Linear Pack no pytorch", 1000):
    #     for _ in range(1000):
    #         C_ml = linear.from_module(linear2, device, torch.float16)