import torch
import tqdm
import gc
import functools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from  utils.color_print import *
from utils.global_config import quant_strategy
from edq.simu_quant import SimuQuantLinear, RealQuantLinearWithScale
from edq.quant_config import LinearQuantConfig
from edq.quant_model import set_op_by_name, get_name_linears
from edq.catch_tensor import catch_embedding_output

@torch.no_grad()
def get_op_by_name(module, op_name):
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")

@torch.no_grad()
def get_op_name(module, op):
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")

@torch.no_grad()
def _apply_scale_block(layer, prev_op, child_name2quant:list, best_scales):
    best_scales = best_scales.cuda()
    prev_op = get_op_by_name(layer, prev_op)
    name2linears = get_name_linears(layer)
    # Apply Best Scale in Weight = Weight * scales.
    for name in child_name2quant:
        linear = name2linears[name].cuda()
        linear.weight.mul_(best_scales)
        assert torch.isnan(linear.weight).sum() == 0
    # Apply Best Scale in Prev_op = Prev / scales.
    if isinstance(prev_op, torch.nn.Linear):
        prev_op.weight.div_(best_scales.view(-1, 1))
        if prev_op.bias is not None:
            prev_op.bias.div_(best_scales.view(-1))
    elif isinstance(prev_op, (torch.nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm)):
        prev_op.weight.div_(best_scales.view(-1))
        if hasattr(prev_op, "bias") and prev_op.bias is not None:
            prev_op.bias.div_(best_scales.view(-1))
    else:
        raise NotImplementedError(f"Prev_op {type(prev_op)} not supported yet.")
    for w in prev_op.parameters():
        assert torch.isnan(w).sum() == 0

@torch.no_grad()
def apply_scale(model, scales_meta_data):
    decoderLayers = model.model.layers
    for layer_idx in tqdm.tqdm(range(len(decoderLayers)), 
                               desc="Apply Scales Meta Data"):
        layer = decoderLayers[layer_idx]
        for scales_args in scales_meta_data[layer_idx]:
            _apply_scale_block(layer, *scales_args)

@torch.no_grad()
def _smooth_scale_block(layer, prev_op, child_name2quant:list, input, 
                        detected_module, detected_input = None, kwargs = {}):
    def get_act_magnitude(input):
        return input.abs().view(-1, input.shape[-1]).mean(0)
    act_magnitude = get_act_magnitude(input)

    if detected_input is None: detected_input = input
    ref_out = detected_module(detected_input, **kwargs)
    if isinstance(ref_out, tuple): ref_out = ref_out[0]

    # Search the best scales
    best_loss = float("inf"); best_scales = None; best_r = None
    n_grid = 20
    name2linears = get_name_linears(layer)
    for ratio in range(n_grid):
        r = ratio*1.0 / n_grid
        scales = act_magnitude.pow(r).clamp(min=1e-4).view(-1)
        scales = (scales / (scales.max() * scales.min()).sqrt()).view(1,-1)
        for name in child_name2quant:
            linear = name2linears[name]
            # NOTE: The q_config here must be the same as quant_model.
            q_config = LinearQuantConfig(quant_strategy)
            q_linear = RealQuantLinearWithScale.from_module(linear, q_config, scales=scales)
            set_op_by_name(layer, name, q_linear)
        cur_out = detected_module(detected_input, **kwargs)
        if isinstance(cur_out, tuple): cur_out = cur_out[0]
        loss = (ref_out.reshape(-1)-cur_out.reshape(-1)).float().pow(2).mean().item()
        if ratio == 0: ori_loss = loss
        if loss < best_loss:
            best_loss = loss; best_scales = scales; best_r = ratio
    # print(f'Loss Reduce: {(ori_loss-best_loss)/ori_loss*100:.2f}%, best r: {best_r}')

    # Recover Original Linear
    for name in child_name2quant:
        set_op_by_name(layer, name, name2linears[name])
    return (get_op_name(layer, prev_op), child_name2quant, best_scales)

@torch.no_grad()
def smooth_scale_layer(layer, layer_kwargs, activations):
    assert isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)),\
        print('Only support Llama or Qwen now.')
    scales_args_list = []
    # attention
    scales_args_list.append(
        _smooth_scale_block(layer, layer.input_layernorm,
        [ 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        activations['self_attn.q_proj'],
        layer.self_attn,
        kwargs=layer_kwargs))
    # fc1
    scales_args_list.append(
        _smooth_scale_block(layer, layer.post_attention_layernorm,
        ['mlp.gate_proj', 'mlp.up_proj'],
        activations['mlp.gate_proj'],
        layer.mlp))
    # fc2
    scales_args_list.append(
        _smooth_scale_block(layer, layer.mlp.up_proj,
        ['mlp.down_proj'],
        activations['mlp.down_proj'],
        layer.mlp,
        detected_input = activations['mlp.gate_proj']))
    return scales_args_list

@torch.no_grad()
def smooth_scale(model, inputs):
    decoderLayers = model.model.layers  # Qwen2.5 7B 是28层堆叠的Qwen2DecoderLayer
    # 捕获embedding层的输出
    embed_output, layer_kwargs = catch_embedding_output(model, inputs)
    
    if "use_cache" in layer_kwargs:
        # layer_kwargs.pop("use_cache")
        layer_kwargs["use_cache"] = False
        layer_kwargs["past_key_value"] = None

    inputs = embed_output[0]
    scales_meta_data = []
    for layer_idx in tqdm.tqdm(range(len(decoderLayers)), 
                               desc="Search Smooth Scales  "):
        layer = decoderLayers[layer_idx].cuda()
        name2linears = get_name_linears(layer)

        activations = {}; handles = []

        def get_input_hook(module, input, output, linear_name, activations):
            x = input[0]    # x.shape = (batch_size, ids_len, hidden_size)
            activations[linear_name] = (x)

        for linear_name in name2linears:
            handles.append(
                name2linears[linear_name].register_forward_hook(
                    functools.partial(get_input_hook, linear_name=linear_name,
                                        activations = activations)))
        inputs = layer(inputs, **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        scales = smooth_scale_layer(layer, layer_kwargs, activations)
        scales_meta_data.append(scales)
        # torch.cuda.empty_cache()
    return scales_meta_data