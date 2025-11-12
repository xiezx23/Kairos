import torch
import tqdm
import gc
from utils.color_print import *
from edq.quant_config import LinearQuantConfig

from test.chronosQuant.two_stage_quant import test_TSQ

@torch.no_grad()
def get_name_linears(module):  
    return {name: m 
        for name, m in module.named_modules() 
            if isinstance(m, torch.nn.Linear)}

@torch.no_grad()
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

@torch.no_grad()
def quantize_model(model, init_only=False, _quant_config=None, _quant_type=None):
    quant_config = []   # List of dict.
    decoderLayers = model.model.layers  # Qwen2___5-7B 是28层堆叠的Qwen2DecoderLayer
    for layer_idx in tqdm.tqdm(range(len(decoderLayers)), desc="Quantize Model"+
                gre_prefix+(" (Initial Only)" if init_only else " (Real Quantize)")+default_color):
        quant_config.append({})
        layer = decoderLayers[layer_idx]
        # 获取模型的线性层，建立名称->线性层的字典
        name2linears = get_name_linears(layer)
        for name, module in name2linears.items():
            return_dtype = module.weight.dtype  # NOTE: Nvidia Devices support bf16 but Ascend 310 does not.
            if _quant_config is not None:
                quant_type = _quant_config[layer_idx][name]
            else:   # Select quant linear type here.
                # if 'self_attn' in name:     quant_type = "W8A8Linear"
                # else:                       quant_type = "W4A16Linear"
                # 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear', 'DynamicLinearSyn'
                if _quant_type is not None:
                    quant_type = _quant_type
                else:
                    quant_type = "DynamicLinear"
                quant_config[layer_idx][name] = quant_type
            # print(f'layer{layer_idx:2d} : {name}')
            # test_TSQ(module.weight.data)
            # continue
            quant_func = LinearQuantConfig(quant_type).from_module
            q_linear = quant_func(module, model.device, init_only=init_only, name='layer'+str(layer_idx)+'.'+name)
            if init_only is False:
                module.cpu()
            set_op_by_name(layer, name, q_linear)
        torch.cuda.empty_cache()
        gc.collect()
    return quant_config

# @torch.no_grad()
# def test_linear(model, layer_idx, name):
#     decoderLayers = model.model.layers
#     layer = decoderLayers[layer_idx]
#     name2linears = get_name_linears(layer)
#     qlinear = name2linears[name]
#     inputs = torch.randn((1, 64, 3584), device='cuda', dtype=torch.float16)
#     try:
#         qlinear.to('cuda')
#         out = qlinear(inputs)
#     except RuntimeError as e:
#         print(red_prefix, 'error:', default_color)
#         print(e)
#     print(color_text('gre', 'Pass Compare Test'))