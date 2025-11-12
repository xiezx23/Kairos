import torch
from utils.color_print import *
from utils.global_config import * #device, enable_accel
from edq.quantization import quantize_tensor_int8, post_gemm_dequant, dequantize_tensor_int8

# @torch.compile
def MIX_PREC_GEMM(f_x, i_w, w_scale, bias):
    i_x, config_x = quantize_tensor_int8(f_x)
    if (i_w.shape[0] <= i_w.shape[1]):
        if f_x.shape[0] <= 16:
            z = torch.zeros(32-f_x.shape[0], f_x.shape[1], dtype=torch.int8, device='cuda')
            i_x = torch.cat([i_x, z], dim=0)
            o = torch._int_mm(i_x, i_w)
            o = o[0:f_x.shape[0], :]
        else:
            o = torch._int_mm(i_x, i_w)
        o = post_gemm_dequant(o, i_x, i_w, config_x['scale'], w_scale)
    else:
        f_w = dequantize_tensor_int8(i_w, w_scale).to(torch.float16)
        o = torch.mm(f_x, f_w)
    if bias is not None:
        o.add_(bias)
    return o

# @torch.compile
def Q_GEMM_DQ(f_x, i_w, w_scale, bias):
    i_x, config_x = quantize_tensor_int8(f_x)
    if hasattr(torch, "npu") and torch.npu.is_available():
        scale = torch.ones(1, device = 'npu', dtype = torch.float16)
        o = torch_npu.npu_quant_matmul(i_x, i_w, w_scale.to(torch.float32).npu(), offset=None, bias=None, output_dtype = torch.float16)
        o = post_gemm_dequant(o, i_x, i_w, config_x['scale'], scale)
    else:
        if f_x.shape[0] <= 16:
            z = torch.zeros(32-f_x.shape[0], f_x.shape[1], dtype=torch.int8, device=device)
            i_x = torch.cat([i_x, z], dim=0)
            o = torch._int_mm(i_x, i_w)
            o = o[0:f_x.shape[0], :]
        else:
            o = torch._int_mm(i_x, i_w)
        o = post_gemm_dequant(o, i_x, i_w, config_x['scale'], w_scale)
    if bias is not None:
        o.add_(bias)
    return o

class W8A8Linear(torch.nn.Module):
    """
        activation  FP16 -> quantize -> INT8\\
        weight      FP16 -> quantize -> INT8\\
        O = A @ W (INT8 GEMM) -> INT32\\
        dequant(O) -> FP16
    """
    def __init__(self, in_features, out_features, bias, device, bit = 8, backend = 'edq'):
        super().__init__()
        self.enable_accel = enable_accel
        self.return_dtype = None
        self.in_features  = in_features
        self.out_features = out_features
        self.bit          = bit
        self.backend      = backend # 'edq' / 'awq'
        buffer_kwargs = {"device" : device, "dtype": torch.float16}
        self.register_buffer(
            "weight",   torch.empty(
                (out_features, in_features),
                device=device, dtype=torch.int8),
        )
        self.register_buffer(
            "scale",    torch.empty(
                (out_features),
                **buffer_kwargs),
        )
        if bias:
            self.register_buffer(
                "bias", torch.empty((out_features), **buffer_kwargs))
        else:
            self.bias = None
    
    @classmethod
    def from_module(cls, linear, device, return_dtype = torch.float16, init_only=False, backend='edq', name = ''):
        backend='awq'
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device, backend=backend)
        q_linear.return_dtype = return_dtype
        if init_only: 
            return q_linear
        if q_linear.enable_accel:
            edq_cuda_accel.quant_fp16_to_int8(
                q_linear.weight, linear.weight.data.to(torch.float16), q_linear.scale)
        else:
            q_linear.weight, q_config = quantize_tensor_int8(linear.weight.data)
            q_linear.scale = q_config['scale'].reshape(-1)
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)
        return q_linear

    @torch.no_grad
    def update_weight(self, weight):
        if self.enable_accel:
            edq_cuda_accel.quant_fp16_to_int8(self.weight, weight.to(torch.float16), self.scale)
        else:
            self.weight, q_config = quantize_tensor_int8(weight)
            self.scale = q_config['scale'].reshape(-1)

    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        intput_shape = input.shape
        if len(intput_shape) > 2:
            input = input.reshape(-1, intput_shape[-1])
        if self.enable_accel:
            int_x = torch.empty_like(input, device=device, dtype=torch.int8)
            out = torch.empty((input.shape[0], self.weight.shape[0]), 
                              device=device, dtype=torch.float16)
            scale_x = torch.empty(input.shape[0], device=device, dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(int_x, input, scale_x)
            if self.backend == 'edq':
                if input.shape[-2] < 8:
                    edq_cuda_accel.w8a8_wsas_gemv_cuda(int_x, self.weight, scale_x, self.scale, out)
                else:
                    edq_cuda_accel.w8a8_wsas_gemm_cuda(int_x, self.weight, scale_x, self.scale, out)
            else:
                awq_backend.w8a8_gemm_forward_cuda(int_x, self.weight, self.scale, scale_x, out)
            if self.bias is not None:
                out.add_(self.bias)
        else:   # Pytorch backend
            out = Q_GEMM_DQ(input, self.weight.T, self.scale, self.bias)
        return out
        return out.reshape(intput_shape[0], intput_shape[1], -1)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

# BACKEND:
#   fbgemm:     x86 CPU推理
#   qnnpack:    ARM移动端推理
# class Qlinear(torch.nn.Module):
#     """
#         activation  FP16 -> quantize -> INT8\\
#         weight      FP16 -> quantize -> INT8\\
#         O = A @ W (INT8 GEMM) -> INT32\\
#         dequant(O) -> FP16
#     """
#     def __init__(self, in_features, out_features, bias, device, bit = 8, group_size = -1):
#         super().__init__()
#         self.quant = torch.ao.quantization.QuantStub()
#         self.dequant = torch.ao.quantization.DeQuantStub()
#         self.in_features  = in_features
#         self.out_features = out_features
#         self.register_buffer(
#             "weight",
#             torch.zeros(
#                 (in_features, out_features),
#                 dtype=torch.int8,
#                 device=device
#             ),
#         )
#         if bias:
#             self.register_buffer(
#                 "bias", torch.zeros((out_features), dtype=torch.float16, device=device)
#             )
#         else:
#             self.bias = None
#     @classmethod
#     def from_module(cls, linear, device, return_dtype):
#         q_linear = cls(
#             linear.in_features, linear.out_features, 
#             linear.bias is not None, device)
#         q_linear.weight = q_linear.quant(linear.weight.T)
#         if linear.bias is not None:
#             q_linear.bias = linear.bias.clone().to(torch.float16)
#         q_linear.quant.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
#         ptq_linear = torch.ao.quantization.prepare(q_linear)
#         ptq_linear = torch.ao.quantization.convert(ptq_linear)
#         return ptq_linear

#     def forward(self, input:torch.Tensor) -> torch.Tensor:
#         input = self.quant(input.to(torch.float))
#         pad_size = max(0, 17 - input.size(0))
#         if pad_size > 0:
#             input_pad = torch.nn.functional.pad(input, (0, 0, 0, pad_size))
#         else:
#             input_pad = input
#         # 执行INT8矩阵乘法
#         out = torch._int_mm(input_pad, self.weight)
#         # 截断到原始维度
#         out = out[:input.size(0)]
#         out = self.dequant(out)
#         out = out + self.bias if self.bias is not None else out
#         return out
