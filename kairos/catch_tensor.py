from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import tqdm
import gc
import functools
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from  utils.color_print import *
from utils.analyse_tensor import analyse_tensor
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
    
    def showValInToken(array, token_idx):
        plt.figure(figsize=(12, 6))
        plt.plot(array[token_idx])
        plt.title(f"Input Features for Token {token_idx}")
        plt.xlabel("Feature Dimension")
        plt.ylabel("Activation Value")
        plt.grid(True)
    
    def showValInChannel(array, channel_idx):
        plt.figure(figsize=(12, 6))
        plt.plot(array.T[channel_idx])
        plt.title(f"Input Features {channel_idx}")
        plt.xlabel("Input Tokens")
        plt.ylabel("Activation Value")
        plt.grid(True)
    
    def showValDensityInToken(array, token_idx):
        print(array[token_idx].mean().item())
        sns.displot(array[token_idx])
        
    analyse_tensor(activations['self_attn.o_proj'][0])
    analyse_tensor(activations['self_attn.o_proj'][13])
    analyse_tensor(activations['self_attn.o_proj'][27])
    plt.show()
    exit(0)

    ori_x = activations['self_attn.o_proj'][15]
    print('------------------------------------------------------')
    mean_data = ori_x.abs().mean().item()
    print("Shape: ({},{})".format(ori_x.shape[-2], ori_x.shape[-1]))
    print("Mean:    {}{:.2f}{} Max:    {}{:.2f}{} Min:    {}{:.2f}{}".format(
        color_text('gre', ori_x.mean().item()),
        color_text('gre', ori_x.max().item()), 
        color_text('gre', ori_x.min().item())))
    print("Mean|x|: {}{:.2f}{} Max|x|: {}{:.2f}{} Min|x|: {}{:.2f}{}".format(
        color_text('gre', mean_data),
        color_text('gre', ori_x.abs().max().item()), 
        color_text('gre', ori_x.abs().min().item())))
    for gs in [1,2,4,8,16,32,64,128,256,512]:
        print('------------------------------------------------------')
        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='S', dim = 0, group_size=gs)
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-token Symmetric Quantize L2 Loss rate:    {:.4f}%".format(loss / mean_data * 100))

        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='S', dim = 1, group_size=gs)
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-channel Symmetric Quantize L2 Loss rate:  {:.4f}%".format(loss / mean_data * 100))

        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='S')
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-tensor Symmetric Quantize L2 Loss rate:   {:.4f}%".format(loss / mean_data * 100))

        print('------------------------------------------------------')

        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='A', dim = 0, group_size=gs)
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-token Asymmetric Quantize L2 Loss rate:   {:.4f}%".format(loss / mean_data * 100))
        
        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='A', dim = 1, group_size=gs)
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-channel Asymmetric Quantize L2 Loss rate: {:.4f}%".format(loss / mean_data * 100))

        x = quantization.simu_quantize_tensor(ori_x, bit=8, q_type='A')
        loss = (ori_x - x).pow(2).mean().sqrt().item()
        print("Per-tensor Asymmetric Quantize L2 Loss rate:  {:.4f}%".format(loss / mean_data * 100))
    print('------------------------------------------------------')


    # tmp_acti = activations['self_attn.q_proj'][0].squeeze().to(torch.float)
    # tmp_array = tmp_acti.numpy()

    # maxChannel = tmp_acti.max(dim = 0).values.max(dim=0)
    # print(maxChannel.indices)

    # showValInChannel(tmp_array, 3197)
    # showValInChannel(tmp_array, maxChannel.indices.item())
    # showValDensityInToken(tmp_array, 7)
    # for i in range(1, 28, 7):
    #     tmp_acti = activations['self_attn.q_proj'][i].squeeze().to(torch.float)
    #     tmp_array = tmp_acti.numpy()
        # maxChannel = tmp_acti.max(dim = 0).values.max(dim=0)
        # print(maxChannel.indices)
        # # showValInChannel(tmp_array, maxChannel.indices.item())
        # showValInChannel(tmp_array, 3197)
    #     showValDensityInToken(tmp_array, 7)
    # plt.show()