import torch
from edq.w4a16_linear import W4A16Linear
from edq.w8a8_linear import W8A8Linear
from edq.dynamic_linear import DynamicLinear
from edq.dynamic_linear_syn import DynamicLinearSyn
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ

class TensorQuantConfig:
    @torch.no_grad
    def __init__(self, q_bit, qtype = None, group_size = -1):
        self.bit = q_bit
        self.qtype = qtype  # 'A' or 'S'
        self.group_size = group_size

class LinearQuantConfig:
    @torch.no_grad
    def __init__(self, quant_type, group_size = -1):
        self.backend = 'awq'
        if quant_type == 'W8A8Linear':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(8, 'S')
            self.from_module = W8A8Linear.from_module

        elif quant_type == 'W4A16Linear':
            self.x = TensorQuantConfig(16)
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            if self.backend == 'edq':
                self.from_module = W4A16Linear.from_module
            else:
                self.from_module = W4A16Linear_AWQ.from_module

        elif quant_type == 'W4A8Linear':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = W4A8Linear.from_module

        elif quant_type == 'DynamicLinear':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = DynamicLinear.from_module

        elif quant_type == 'DynamicLinearSyn':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = DynamicLinearSyn.from_module
        else:
            assert 0, print(f'No Supported Linear Type: {quant_type}.')