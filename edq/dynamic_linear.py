import torch
import edq_cuda_accel
import awq_backend
import torch.cuda.nvtx as nvtx
from utils.color_print import *
from utils.perf_eval import timer
from edq.quantization import (quantize_tensor_int8, pack_int4_data, dequantize_tensor_int4, real_quantize_tensor_edq)
from thirdparty.AWQ.awq_method import pack_int_awq, trans_qfactor_layout_awq, dequantize_tensor_awq, quant_weight_awq

COMP_TYPE_NAME = {'w4a16_gemm':0, 'w4a16_gemv':1, 'w8a8':2, 'w8a16_gemm':3, 'w8a16_gemv':4, 'fp16':5, 'w4a16':6}

@torch.compile()
@torch.no_grad()
def edq_quantize_tensor_int4(x, group_size = -1):
    x_shape = x.shape    
    if group_size < 0:
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    max_int = 15    # 2**4 - 1``
    max_val = x.amax(dim=1, keepdim=True)
    min_val = x.amin(dim=1, keepdim=True)
    scale_f   = ((max_val-min_val).clamp(min=1e-5) / max_int)
    """NOTE: AWQ employs an integer zero value, but this is suboptimal."""
    # zero_pt_f = (-(min_val / scale_f).round_()).clamp_(0, max_int)
    # x.div_(scale_f).round_().add_(zero_pt_f).clamp_(0, max_int)
    zero_pt_f = (-(min_val / scale_f))
    x.div_(scale_f).add_(zero_pt_f).round_().clamp_(0, max_int)
    assert torch.isnan(x).sum() == 0, print(scale_f.amin())
    # NOTE: Use floor instead of Round to prevent overflow during quantization.
    scale_i   = torch.floor(scale_f).to(torch.int8)
    zero_pt_i = torch.round(zero_pt_f).to(torch.int8)
    return x.reshape(x_shape).to(torch.int8), {
        'scale_f' : scale_f.to(torch.float16),      'scale_i' : scale_i,
        'zero_pt_f' : zero_pt_f.to(torch.float16),  'zero_pt_i' : zero_pt_i, 
        'group_size':group_size}
        
class DynamicLinear(torch.nn.Module):
    """
        Weight   FP16 -> INT8 -> INT4
        Dynamically select Wq and Aq at runtime.
    """
    prof = None
    @torch.no_grad
    def __init__(self, in_features, out_features, bias, device, group_size):
        super().__init__()
        self.enable_accel = False
        self.return_dtype = torch.float16
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        i8_buffer_kwargs  = {"device" : device, "dtype": torch.int8}
        fp_buffer_kwargs  = {"device" : device, "dtype": torch.float16}
        # self.register_buffer(
        #     "qweight",   torch.empty((out_features, in_features // 2), **i8_buffer_kwargs))
        # self.register_buffer(   # Used for int8 -> fp16.
        #     "scale_1",    torch.empty((out_features), **fp_buffer_kwargs))
        # self.register_buffer(
        #     "scale_2_f",        # Used for int4 -> int8 in dynamic transformation.
        #     torch.empty((out_features * (in_features//group_size)), **fp_buffer_kwargs))
        # self.register_buffer(
        #     "zero_2_f",         # Used for int4 -> int8 in dynamic transformation.
        #     torch.empty((out_features * (in_features//group_size)), **fp_buffer_kwargs))
        # self.register_buffer(
        #     "scale_2_i",        # Used for int4 -> int8 in W4A8 kernel.
        #     torch.empty((out_features * (in_features//group_size)), **i8_buffer_kwargs))
        # self.register_buffer(
        #     "zero_2_i",         # Used for int4 -> int8 in W4A8 kernel.
        #     torch.empty((out_features * (in_features//group_size)), **i8_buffer_kwargs))
        if bias:
            self.register_buffer(
                "bias", torch.empty((out_features), **fp_buffer_kwargs))
        else:
            self.bias = None
        self.shape  = (out_features, in_features)
        self.pre_m = 0
        self.pre_comp_type = ''
    
    def init(self):
        """ Initial work for quantized model loaded from memory. """
        self.scale_f = self.scale_2_f.view(self.scale_1.shape[0], -1)*self.scale_1.view(-1,1)
    
    @classmethod
    def from_module(cls, linear, device, return_dtype=torch.float16, init_only=False, name = ''):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device, group_size = 128)
        # q_linear.return_dtype = return_dtype
        # q_linear.name = name
        # q_linear.ori_weight = linear.weight.data.clone().T
        if init_only: 
            return q_linear
        """ Step1: Quantize weight from fp16 to int8.   """
        weight_tensor = linear.weight.data.to(torch.float16)
        N, K = weight_tensor.shape
        weight_i8, q_config1 = quantize_tensor_int8(weight_tensor, max_int=127)
        scale_1 = q_config1['scale']
        q_linear.scales_1 = scale_1.view(-1)
        """ Step2: Quantize weight from int8 to uint4.  """
        # weight_i4, scale_2_f, zero_2_f = real_quantize_tensor_edq(weight_i8.clone())
        weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=128)
        scale_2_f = q_config2['scale_f']
        zero_2_f = q_config2['zero_pt_f']
        q_linear.scales_to8, q_linear.scaled_zeros_to8 = trans_qfactor_layout_awq(
            scale_2_f.reshape(N,-1), zero_2_f.reshape(N,-1), 128, method='EDQ')
        """ Step3: Calculate the dequantization factor for uint4 to fp16. """
        scale_f = scale_2_f.view(scale_1.shape[0], -1) * scale_1
        q_linear.qweight = pack_int_awq(weight_i4.clone())
        q_linear.scales, q_linear.scaled_zeros = trans_qfactor_layout_awq(
            scale_f.reshape(N,-1), zero_2_f.reshape(N,-1), 128, method='EDQ')
        
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half()
        return q_linear
    
    
    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        m = input.shape[-2]
        # m = input.numel() // input.shape[-1]
        if m != self.pre_m:
            self.pre_m = m
            self.comp_type = DynamicLinear.prof.get_comp_type(self.shape, m)
            # print(f'{m}  {self.comp_type}')
        # self.comp_type = COMP_TYPE_NAME['w4a16']
        return self._forward(input, m, self.comp_type)

    @torch.no_grad
    # @torch.compile
    def _forward(self, input:torch.Tensor, batch_token_len, comp_type) -> torch.Tensor:
        # w4a16 kernel
        if comp_type == COMP_TYPE_NAME['w4a16_gemv']:
            out = awq_backend.gemv_forward_cuda_new(
                input, self.qweight, self.scales, self.scaled_zeros,
                batch_token_len, self.out_features, self.in_features, self.group_size)
        elif comp_type == COMP_TYPE_NAME['w4a16_gemm']:
            out = awq_backend.gemm_forward_cuda_new(
                input, self.qweight, self.scales, self.scaled_zeros)
        # w8a8 kernel
        elif comp_type == COMP_TYPE_NAME['w8a8']:
            tmp_weight = torch.empty(self.shape, dtype=torch.int8, device='cuda')
            edq_cuda_accel.dequant_interleaved_int4_to_int8(tmp_weight, self.qweight, 
                                        self.scales_to8, self.scaled_zeros_to8, 128)
            intput_shape = input.shape
            input = input.view(-1, intput_shape[-1])
            int_x = torch.empty_like(input, device='cuda', dtype=torch.int8)
            scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
            edq_cuda_accel.quant_fp16_to_int8(int_x, input, scale_x)
            out = torch.empty((input.shape[0], self.shape[0]), device='cuda', dtype=torch.float16)
            awq_backend.w8a8_gemm_forward_cuda(int_x, tmp_weight, self.scales_1, scale_x, out)
        # full precision kernel
        elif comp_type == COMP_TYPE_NAME['fp16']:
            tmp_weight = torch.empty(self.shape, dtype=torch.float16, device='cuda')
            edq_cuda_accel.dequant_interleaved_int4_to_fp16(tmp_weight, self.qweight, 
                                        self.scales, self.scaled_zeros, 128)
            out = input @ tmp_weight.T
        # # w8a16 kernel
        # elif comp_type == COMP_TYPE_NAME['w8a16_gemm']:
        #     intput_shape = input.shape
        #     input = input.view(-1, intput_shape[-1])
        #     tmp_weight = torch.empty(self.shape, dtype=torch.int8, device='cuda')
        #     edq_cuda_accel.dequant_interleaved_int4_to_int8(tmp_weight, self.qweight, 
        #                                 self.scales_to8, self.scaled_zeros_to8, 128)
        #     out = torch.empty((input.shape[0], self.shape[0]), device='cuda', dtype=torch.float16)
        #     edq_cuda_accel.w8a16_ws_gemm_cuda(input, tmp_weight, self.scales_1, out)
        # elif comp_type == COMP_TYPE_NAME['w8a16_gemv']:
        #     intput_shape = input.shape
        #     input = input.view(-1, intput_shape[-1])
        #     tmp_weight = torch.empty(self.shape, dtype=torch.int8, device='cuda')
        #     edq_cuda_accel.dequant_interleaved_int4_to_int8(tmp_weight, self.qweight, 
        #                                 self.scales_to8, self.scaled_zeros_to8, 128)
        #     out = torch.empty((input.shape[0], self.shape[0]), device='cuda', dtype=torch.float16)
        #     edq_cuda_accel.w8a16_ws_gemv_cuda(input, tmp_weight, self.scales_1, out)
        #     """Beta Version"""
        # else: assert 0, print(f"Unkown comp_type_id: {comp_type}")
        if self.bias is not None:
            out += self.bias
        """ CHECK CORRECTNESS REGION
        if torch.isnan(out).sum() > 0 or torch.isinf(out).sum() > 0:
            print(self.name, comp_type, torch.isnan(out).sum(), out)
            exit(0)
        """
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"