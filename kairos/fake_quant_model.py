import torch
import tqdm
import gc
import os
from torch import nn
from kairos.quantization import (
    simu_quantize_tensor, quantize_tensor, dequantize_tensor, 
    simu_quantize_tensor_autoscale, simu_quantize_weight_mix_precsion
)
from utils.analyse_tensor import analyse_tensor

class QLinear(nn.Module):
    @torch.no_grad
    def __init__(self, in_features, out_features, bias, q_bit, device):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.q_bit        = q_bit

    @classmethod
    def from_module(cls, linear, q_bit = (4, 8), device = 'cuda', acti_max_row = None):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, q_bit, device)
        gv.update_max('max', linear.weight.data.amax().item())
        gv.update_min('min', linear.weight.data.amin().item())
        
        if acti_max_row is None:
            q_linear.weight = simu_quantize_tensor(
                linear.weight.data, q_type='A', bit=q_bit[0], group_size=128).T
        else:
            q_linear.weight = simu_quantize_weight_mix_precsion(
                linear.weight.data, acti_max_row=acti_max_row, q_type='A', group_size=128).T
        # qw, config = quantize_tensor(linear.weight.data)
        # q_linear.weight = dequantize_tensor(qw, config['scale']).T
        # q_linear.weight += w.T
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)
        return q_linear
    
    @classmethod    # AWQ method
    def from_module_auto_scale(cls, linear, inputs, q_bit = (8, 8), device = 'cuda'):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, q_bit, device)
        inputs = inputs.to(linear.weight.device)
        gv.update_max('max', linear.weight.data.amax().item())
        gv.update_min('min', linear.weight.data.amin().item())

        def get_act_magnitude(inputs):
            return inputs.abs().view(-1, inputs.shape[-1]).mean(0)
        
        act_magnitude = get_act_magnitude(inputs)
        best_loss = float("inf")
        best_ratio = -1
        best_scales = None
        ori_out = inputs @ linear.weight.T
        for ratio in range(20):
            r = ratio*2.0 / 20
            scales = act_magnitude.pow(r).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            cur_weight = linear.weight * scales
            cur_weight = simu_quantize_tensor(cur_weight, bit=q_bit[0], q_type='A', group_size=-1)
            # cur_weight = simu_quantize_tensor_autoscale(cur_weight, bit=q_bit[0], q_type='A', group_size=-1)
            cur_weight.div_(scales)
            cur_out = inputs @ cur_weight.T

            loss = (ori_out - cur_out).float().pow(2).mean().item()
            if loss < best_loss:
                best_scales = scales
                best_loss = loss
                best_ratio = r
                q_linear.weight = cur_weight.clone().T
        assert best_ratio != -1 
        # qw, config = quantize_tensor(linear.weight.data)
        # q_linear.weight = dequantize_tensor(qw, config['scale']).T
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().to(torch.float16)
        return q_linear

    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        # gv.update_max('max', input.amax().item())
        # gv.update_min('min', input.amin().item())
        input = simu_quantize_tensor(input, bit=self.q_bit[1], q_type='A', group_size=128)
        out = input @ self.weight
        if self.bias is not None:
            out.add_(self.bias)
        return out

@torch.no_grad()
def fake_quantize_model(model, bit = 8, enable_scales = False, activations = None):
    if (enable_scales):
        if (os.path.exists(gv.SAVE_PATH + gv.ACT_INFO)):
            acti_mean_distri = torch.load(gv.SAVE_PATH+gv.ACT_INFO)
        else: assert('Need to run catch_activation first to get activation info.')
    decoderLayers = model.model.layers  # Qwen2是28层堆叠的Qwen2DecoderLayer
    # for layer_idx in tqdm.tqdm(range(0, 4), desc="Fake Quantize Model"):
    for layer_idx in tqdm.tqdm(range(0, len(decoderLayers)), desc="Fake Quantize Model"):
        layer = decoderLayers[layer_idx]
        # 获取模型的线性层，建立名称->线性层的字典
        def get_name_linears(module):  
            return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}
        name2linears = get_name_linears(layer)
        for name, module in name2linears.items():
            if (enable_scales):
                # q_linear = QLinear.from_module_auto_scale(
                #     module, inputs=activations[name][layer_idx]
                # )
                q_linear = QLinear.from_module(
                    module, acti_max_row=acti_mean_distri[name+str(layer_idx)]
                )
            else:
                # if layer_idx < 14:
                #     qbit = 4
                # else:
                #     qbit = 8
                qbit = 6
                q_linear = QLinear.from_module(module, q_bit=(qbit, 8))
            module.cpu()
            def set_op_by_name(layer, name, new_module):
                levels = name.split(".")
                if len(levels) > 1:
                    mod_ = layer
                    for l_idx in range(len(levels) - 1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], new_module)
                else:
                    setattr(layer, name, new_module)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()


# BACKEND:
#   fbgemm:     x86 CPU推理
#   qnnpack:    ARM移动端推理
# class Qlinear(nn.Module):
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
