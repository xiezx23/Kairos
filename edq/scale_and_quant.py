import torch
import gc
import functools
import math
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm

from  utils.color_print import *
from edq.simu_quant import SimuQuantLinear, RealQuantLinearWithScale
from edq.quant_config import LinearQuantConfig
from edq.quant_model import set_op_by_name, get_name_linears
from edq.smooth_scale import get_op_by_name, get_op_name, _apply_scale_block
from edq.catch_tensor import catch_embedding_output


@torch.no_grad()
def _smooth_scale_block(layer, prev_op, child_name2quant:list, input, input_q,
            detected_module, detected_input = None, detected_input_q = None, kwargs = {}):
    def get_act_magnitude(input):
        return input.abs().view(-1, input.shape[-1]).mean(0)
    act_magnitude = get_act_magnitude(input_q)

    if detected_input is None: detected_input = input
    if detected_input_q is None: detected_input_q = input_q
    ref_out = detected_module(detected_input, **kwargs)
    if isinstance(ref_out, tuple): ref_out = ref_out[0]

    # Replace Linear with SimuQuantlinear.
    name2linears = get_name_linears(layer)
    for name in child_name2quant:
        linear = name2linears[name]
        # NOTE: The q_config here must be the same as quant_model.
        if linear.in_features <= 4096 and linear.out_features <= 4096:
            q_config = LinearQuantConfig("W8A8Linear")
        else:
            q_config = LinearQuantConfig("W4A16Linear", group_size=128)
        # q_config = LinearQuantConfig("W8A8Linear")
        # q_config = LinearQuantConfig("W4A16Linear", group_size=128)
        q_linear = SimuQuantLinear.from_module(linear, q_config)
        # q_linear = SimuQuantLinear.from_module(linear, q_config)
        set_op_by_name(layer, name, q_linear)

    # Search the best scales.
    best_loss = float("inf"); best_scales = None; best_r = None
    n_grid = 20
    for ratio in range(n_grid):
        if ratio > 0:
            r = ratio*1.0 / n_grid
            scales = act_magnitude.pow(r).clamp(min=1e-4).view(-1)
            # scales = scales.div_(scales.mean()).view(1,-1)
            scales = (scales / (scales.max() * scales.min()).sqrt()).view(1,-1)
            for name in child_name2quant:
                q_linear = get_op_by_name(layer, name)
                q_linear.update_scale(name2linears[name].weight.data, scales)
        else: scales = torch.tensor(1, device = 'cuda', dtype = torch.float16)
        cur_out = detected_module(detected_input_q, **kwargs)[0]
        # if isinstance(cur_out, tuple): cur_out = cur_out[0].
        loss = (ref_out-cur_out).float().pow(2).mean().item()
        if loss < best_loss:
            best_loss = loss; best_scales = scales; best_r = ratio
    # Recover Original Linear.
    for name in child_name2quant:
        set_op_by_name(layer, name, name2linears[name])
    return (get_op_name(layer, prev_op), child_name2quant, best_scales)

@torch.no_grad()
def _quantize_layer(layer, quant_config_layer):
    name2linears = get_name_linears(layer)
    for name, module in name2linears.items():
        if 'self_attn' in name:     quant_type = "W8A8Linear"
        else:                       quant_type = "W4A16Linear"
        # 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear', 'DynamicLinearSyn'
        # quant_type = "DynamicLinear"
        quant_config_layer[name] = quant_type
        quant_func = LinearQuantConfig(quant_type).from_module
        q_linear = quant_func(module, 'cuda')
        module.cpu()
        set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def scale_quant_layer(layer, quant_config_layer, layer_kwargs, activations, activations_q):
    assert isinstance(layer, (LlamaDecoderLayer, Qwen2DecoderLayer)),\
        print('Only support Llama or Qwen now.')
    scales_args_list = []
    # attention
    scales_args_list.append(
        _smooth_scale_block(layer, layer.input_layernorm,
        [ 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
        activations['self_attn.q_proj'], activations_q['self_attn.q_proj'],
        layer.self_attn,
        kwargs=layer_kwargs))
    # fc1
    scales_args_list.append(
        _smooth_scale_block(layer, layer.post_attention_layernorm,
        ['mlp.gate_proj', 'mlp.up_proj'],
        activations['mlp.gate_proj'], activations_q['mlp.gate_proj'],
        layer.mlp))
    # fc2
    scales_args_list.append(
        _smooth_scale_block(layer, layer.mlp.up_proj,
        ['mlp.down_proj'],
        activations['mlp.down_proj'], activations_q['mlp.down_proj'],
        layer.mlp,
        detected_input = activations['mlp.gate_proj'], 
        detected_input_q=activations_q['mlp.gate_proj']))
    # Apply Scales
    for scales_args in scales_args_list:
        _apply_scale_block(layer, *scales_args)
    # Quantize the layer
    _quantize_layer(layer, quant_config_layer)
    return scales_args_list

@torch.no_grad()
def scale_and_quant(model, inputs):
    decoderLayers = model.model.layers  # Qwen2.5 7B 是28层堆叠的Qwen2DecoderLayer
    # 捕获embedding层的输出
    embed_output, layer_kwargs = catch_embedding_output(model, inputs)

    if "use_cache" in layer_kwargs:
        # layer_kwargs.pop("use_cache")
        layer_kwargs["use_cache"] = False
        layer_kwargs["past_key_value"] = None

    inputs = embed_output[0]
    inputs_q = inputs
    scales_meta_data = []
    quant_config = []   # List of dict.
    for layer_idx in tqdm(range(len(decoderLayers)), 
                               desc="Scale and Quant Model"):
        layer = decoderLayers[layer_idx]
        layer = layer.to('cuda')
        name2linears = get_name_linears(layer)
        activations = {}; handles = []
        quant_config.append({})
        def get_input_hook(module, input, output, linear_name, activations):
            x = input[0]    # x.shape = (batch_size, ids_len, hidden_size)
            activations[linear_name] = (x)

        for linear_name in name2linears:
            handles.append(
                name2linears[linear_name].register_forward_hook(
                    functools.partial(get_input_hook, linear_name=linear_name,
                                        activations = activations)))
        next_inputs = layer(inputs, **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        # Get activation in with inputs that affected by quantization.
        activations_q = {}
        for linear_name in name2linears:
            handles.append(
                name2linears[linear_name].register_forward_hook(
                    functools.partial(get_input_hook, linear_name=linear_name,
                                        activations = activations_q)))
        layer(inputs_q, **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        scales = scale_quant_layer(layer, quant_config[layer_idx],
                        layer_kwargs, activations, activations_q)
        scales_meta_data.append(scales)
        layer = layer.cuda()
        inputs_q = layer(inputs, **layer_kwargs)[0]
        inputs = next_inputs
    return scales_meta_data, quant_config