import torch
from utils.color_print import *
from utils.global_config import device
import awq_backend

class W8A8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias, device, bit = 8):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.bit          = bit
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
    def from_module(cls, linear, device, return_dtype = torch.float16, init_only=False, backend='kairos', name = ''):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device)
        if init_only: 
            return q_linear
        awq_backend.invoke_quant(
            q_linear.weight, linear.weight.data.to(torch.float16), q_linear.scale)
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)
        return q_linear

    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        intput_shape = input.shape
        if len(intput_shape) > 2:
            input = input.reshape(-1, intput_shape[-1])
        int_x = torch.empty_like(input, device=device, dtype=torch.int8)
        out = torch.empty((input.shape[0], self.weight.shape[0]), 
                            device=device, dtype=torch.float16)
        scale_x = torch.empty(input.shape[0], device=device, dtype=torch.float16)
        awq_backend.invoke_quant(int_x, input, scale_x)
        awq_backend.w8a8_gemm_forward_cuda(int_x, self.weight, self.scale, scale_x, out)
        if self.bias is not None:
            out.add_(self.bias)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"