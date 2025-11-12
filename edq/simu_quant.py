import torch
from edq.quantization import (
    simu_quantize_tensor, quantize_tensor, dequantize_tensor
)
from utils.analyse_tensor import analyse_tensor
from edq.quant_config import LinearQuantConfig

class SimuQuantLinear(torch.nn.Module):
    @torch.no_grad
    def __init__(self, in_features, out_features, q_config):
        super().__init__()
        self.q_config     = q_config
        self.in_features  = in_features
        self.out_features = out_features
        self.bias         = None

    @classmethod
    def from_module(cls, linear, q_config:LinearQuantConfig, scales = 1):
        q_linear = cls(linear.in_features, linear.out_features, q_config)
        q_linear.scales = scales
        # qw, config = quantize_tensor(linear.weight.data)
        # q_linear.weight = dequantize_tensor(qw, config['scale'])
        q_linear.weight = simu_quantize_tensor(linear.weight.data * scales, 
            q_type=q_config.w.qtype, bit=q_config.w.bit, group_size=q_config.w.group_size)
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)
        if q_config.x.bit != 16:
            def x_quantize_func(x):
                return simu_quantize_tensor(x, bit=q_config.x.bit, q_type=q_config.x.qtype, 
                                            group_size=q_config.x.group_size)
        else:
            def x_quantize_func(x):
                return x
        q_linear.x_quantize = x_quantize_func
        return q_linear
    
    @torch.no_grad
    def update_scale(self, weight, scales):
        self.scales = scales
        self.weight.copy_(weight)
        self.weight.mul_(scales)
        # Simu Quantize Inplace.
        bit = self.q_config.w.bit
        weight_shape = self.weight.shape
        if self.q_config.w.group_size != -1:
            self.weight = self.weight.reshape(-1, self.q_config.w.group_size)
        if self.q_config.w.qtype == 'A':
            max_int = 2**(bit) - 1
            max_val = self.weight.amax(dim=(1), keepdim=True)
            min_val = self.weight.amin(dim=(1), keepdim=True)
            scale   = (max_val-min_val).clamp(min=1e-7) / max_int
            zero_pt = -(min_val / scale)
            self.weight.div_(scale).add_(zero_pt).round_().clamp_(min = 0, max = max_int)
            assert torch.isnan(self.weight).sum() == 0
            self.weight.sub_(zero_pt).mul_(scale)
        else:
            max_int = 2**(bit-1) - 1
            max_val = self.weight.abs().amax(dim=(1), keepdim=True)
            scale   = (max_val / max_int).clamp(min=1e-7)
            self.weight.div_(scale).round_().clamp_(min = -max_int-1, max = max_int)
            assert torch.isnan(self.weight).sum() == 0
            self.weight.mul_(scale)
        self.weight = self.weight.reshape(weight_shape)
    
    @torch.no_grad
    def forward(self, input):
        out = self.x_quantize(input / self.scales)
        out = out @ self.weight.T
        if self.bias is not None:
            out += self.bias
        return out

class RealQuantLinearWithScale(torch.nn.Module):
    @torch.no_grad
    def __init__(self, q_config):
        super().__init__()
        self.q_config = q_config
        self.scales = 1

    @classmethod
    def from_module(cls, linear, q_config:LinearQuantConfig, scales = 1):
        q_linear = cls(q_config)
        q_linear.scales = scales
        ori_weight = linear.weight.data.clone()
        linear.weight.data.mul_(scales)
        q_linear.proj = q_config.from_module(linear, 'cuda')
        linear.weight.data = (ori_weight)
        return q_linear
    
    @torch.no_grad
    def update_scale(self, weight, scales):
        self.scales = scales
        self.proj.update_weight(weight * scales)
    
    @torch.no_grad
    def forward(self, input):
        return self.proj(input / self.scales)