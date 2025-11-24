import torch
import tqdm
import gc
import functools
from collections import defaultdict
from  utils.color_print import *
from kairos.quant_model import get_name_linears

@torch.no_grad()
def catch_embedding_output(model, inputs):
    model.model.rotary_emb   = model.model.rotary_emb.to('cuda')
    model.model.embed_tokens = model.model.embed_tokens.to('cuda')
    decoderLayers = model.model.layers  # Qwen2.5 7B 是28层堆叠的Qwen2DecoderLayer
    embed_output = []
    layer_kwargs = {}
    def pre_forward_hook(module, args, kwargs): # 在执行forward之前先截取数据
        embed_output.append(args[0])
        layer_kwargs.update(kwargs)
        raise ValueError
    handle = decoderLayers[0].register_forward_pre_hook(pre_forward_hook, with_kwargs=True)
    try:
        model(**inputs)
    except ValueError: print('Already catch the output of embedding.')
    handle.remove()
    del inputs
    model.model.rotary_emb   = model.model.rotary_emb.to('cpu')
    model.model.embed_tokens = model.model.embed_tokens.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    return embed_output, layer_kwargs

@torch.no_grad()
def catch_activation(model, tokenizer, inputs):
    decoderLayers = model.model.layers  # Qwen2.5 7B 是28层堆叠的Qwen2DecoderLayer
    model.cuda()
    # 捕获embedding层的输出
    embed_output, layer_kwargs = catch_embedding_output(model, inputs)
    # 捕获每个线性层的输入激活
    activations = defaultdict(list)
    acti_max_distri = {}
    inputs = embed_output[0]
    for layer_idx in tqdm.tqdm(range(len(decoderLayers)), 
                               desc="Collecting Layer Activation:"):
        layer = decoderLayers[layer_idx]
        name2linears = get_name_linears(layer)

        def get_input_hook(module, input, output, linear_name, layer_idx, acti_dict, max_dist):
            x = input[0]
            # assert x.dim() == 3 # shape(batch_size, ids_len, hidden_size)
            acti_dict[linear_name].append(x.cpu())
            # 获取列上的最值，并在batch间取mean
            max_in_col = x.abs().max(dim=1).mean(dim=0, keepdim=False)
            max_dist[linear_name+str(layer_idx)] = (max_in_col)

        handles = []
        for linear_name in name2linears:
            handles.append(
                name2linears[linear_name].register_forward_hook(
                    functools.partial(
                        get_input_hook, linear_name=linear_name, layer_idx = layer_idx,
                        acti_dict=activations, max_dist = acti_max_distri
                    )
                )
            )
        inputs = layer(inputs, **layer_kwargs)[0]
        for handle in handles:
            handle.remove()
        torch.cuda.empty_cache()
    # for k, v in acti_max_distri.items():
    #     print(k, ":", len(v))
    return activations, acti_max_distri