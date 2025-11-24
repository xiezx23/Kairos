import torch
from kairos.quant_config import LinearQuantConfig

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