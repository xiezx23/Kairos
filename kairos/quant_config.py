import torch
from kairos.dynamic_linear import DynamicLinear
from thirdparty.SmoothQuant.w8a8_linear import W8A8Linear
from thirdparty.Qserve.w4a8_linear import W4A8Linear
from thirdparty.QQQ.w4a8_linear import W4A8Linear_QQQ
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AutoGPTQ.w4a16_linear import W4A16Linear_Marlin

class TensorQuantConfig:
    @torch.no_grad
    def __init__(self, q_bit, qtype = None, group_size = -1):
        self.bit = q_bit
        self.qtype = qtype  # 'A' or 'S'
        self.group_size = group_size

class LinearQuantConfig:
    @torch.no_grad
    def __init__(self, quant_type):
        if quant_type == 'W8A8Linear':      # W8A8Linear_SmoothQuant
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(8, 'S')
            self.from_module = W8A8Linear.from_module

        elif quant_type == 'W4A16Linear':   # W4A16Linear_AWQ
            self.x = TensorQuantConfig(16)
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = W4A16Linear_AWQ.from_module
            
        elif quant_type == 'W4A16Linear_Marlin':
            self.x = TensorQuantConfig(16)
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = W4A16Linear_Marlin.from_module

        elif quant_type == 'W4A8Linear':   # W4A8Linear_Qserve
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = W4A8Linear.from_module

        elif quant_type == 'W4A8Linear_QQQ':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = W4A8Linear_QQQ.from_module

        elif quant_type == 'DynamicLinear':
            self.x = TensorQuantConfig(8, 'S')
            self.w = TensorQuantConfig(4, 'A', group_size=128)
            self.from_module = DynamicLinear.from_module
            
        else:
            assert 0, print(f'No Supported Linear Type: {quant_type}.')