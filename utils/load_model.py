import gc
import os
import re
from typing import Union, List

import torch
import torch.nn as nn
from tqdm import tqdm
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils, GenerationConfig, LlamaForCausalLM

from edq.quant_model import quantize_model
from models.qwen2 import Qwen2ForCausalLM
from utils.util import device_warmup
from utils.util import load_json
from utils.color_print import *
from utils.calibration import generate_random_token_sequence

def _setup(device='cuda:0', dtype=torch.float16, seed=None, ):
    if seed is not None:
        from utils.util import set_seed
        set_seed(seed)

    torch.set_default_dtype(dtype)

    modeling_utils._init_weights = False

    def skip(*args, **kwargs):
        pass  # 空函数，用于跳过 PyTorch 的初始化操作

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    device_warmup(device)


def mem_efficient_load_checkpoint(
        model: nn.Module,
        ckpts_folder: Union[str, os.PathLike],
):
    checkpoint_files = [ckpts_folder + "/" + f for f in os.listdir(ckpts_folder) if f.endswith(".pt")]

    # Check if the ckpts match the model
    model_keys = sorted((list(model.state_dict().keys())))
    suffix = r"\.pt$"
    ckpt_keys = sorted(
        [re.sub(suffix, "", f) for f in os.listdir(ckpts_folder) if f.endswith(".pt")]
    )
    assert len(model_keys) == len(ckpt_keys), \
        (f"The number of checkpoint files do not match the model. \n Model has {len(model_keys)} keys, "
         f"while finding {len(ckpt_keys)} checkpoint files in the folder.")
    for key1, key2 in zip(model_keys, ckpt_keys):
        assert (key1 == key2), \
            f"The checkpoint files do not match the model. \nmodel key {key1} != checkpoint key {key2}"

    with tqdm(total=len(checkpoint_files)) as pbar:
        pbar.set_description("Loading checkpoint shards")
        for checkpoint_file in checkpoint_files:
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint, strict=False)
            # Force Python to clean up.
            del checkpoint
            gc.collect()
            pbar.update(1)
    return model


def load_quant_model(model, checkpoint, device, efficient_load=False):
    """
    快速加载量化的 LLaMA/Qwen 类模型
    参数:
        model: 预初始化的模型结构（未加载权重）
        checkpoint: 量化权重文件路径（.pt 或 .safetensors）或内存效率模式下的分片目录
        device: 目标设备（如 'cuda:0'）
        group_size: 量化分组大小（如 128，控制量化精度与计算效率的平衡）
        efficient_load: 是否启用内存效率模式
    """

    # 将找到的线性层替换为量化线性层（WQLinear）
    # 注：仅初始化量化层结构，实际权重将在后续加载 checkpoint 时填充
    quant_config = load_json(checkpoint[0:len(checkpoint)-11]+'/quant_config.json')
    quantize_model(model, init_only=True, _quant_config=quant_config)

    model.tie_weights()

    # 条件分支：内存效率模式加载（适用于低内存环境，分片加载权重）
    if efficient_load:
        assert os.path.isdir(checkpoint), \
            ("You are in mem_efficient_load mode. \n "
             "Please set --load_quant the path to the folder containing all checkpoint files.")
        model = mem_efficient_load_checkpoint(model, checkpoint, )
    else:
        pbar = tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            if checkpoint.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load

                model.load_state_dict(safe_load(checkpoint))
            else:
                model.load_state_dict(torch.load(checkpoint))

    return model.to(device)


def load_from_pretrained(model_path: str,
                         weight_path: str,
                         model_type: str,
                         trust_remote_code=False,
                         local_files_only=True,
                         max_new_tokens=2048,
                         setup=True, dtype=torch.float16,
                         seed=None, device='cuda:0',
                         quant_llm=False):
    assert isinstance(model_type, str) and model_type.lower() in ['llama', 'qwen'], \
        'We only support llama and qwen model now!'
    print("Using Model:", color_text('gre', model_path))
    if setup:
        _setup(device=device, seed=seed, dtype=dtype)

    # config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                        local_files_only=local_files_only)
    # generation_config
    gen_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                                  local_files_only=local_files_only)
    gen_config.max_new_tokens = max_new_tokens

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                              local_files_only=local_files_only)

    # 部分生成类模型在训练时只用 eos_token 做句子结束标记，没有定义 pad_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 定义模型类型字典
    model_type_dict = {
        "llama": LlamaForCausalLM, # TODO: support llama model locally.
        "qwen": Qwen2ForCausalLM,
    }

    # 创建模型实例, 加载量化参数
    model = model_type_dict[model_type.lower()](config)
    if quant_llm:
        print("Loading quantized model:", color_text('gre', weight_path))
        model = load_quant_model(model, weight_path, device)
    else:
        # 官方标准库中定义的Model的实例
        loaded_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=trust_remote_code,
                                                            local_files_only=local_files_only,
                                                            config=config,
                                                            torch_dtype=dtype)
        model.load_state_dict(loaded_model.state_dict())

    model.to(device)

    gen_config.return_legacy_cache = True
    model.generation_config = gen_config
    model.eval()

    # 进行多次前向传播，充分预热模型
    for _ in range(3):
        test_input = generate_random_token_sequence(tokenizer, device=device, seq_length=64)
        with torch.no_grad():
            model.generate(**test_input, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)

    return model, tokenizer

def load_model_tokenizer(model_path, q_model_path, dtype = torch.float16):
    abs_model_path = os.path.abspath(model_path)
    if not os.path.exists(abs_model_path):
        raise FileNotFoundError(f"{abs_model_path} not found!")
    print("Using Model:", color_text('gre', model_path))
    if q_model_path:
        print("Loading quantized model:", color_text('gre', q_model_path))
        config = AutoConfig.from_pretrained(abs_model_path, local_files_only = True)
        gen_config = GenerationConfig.from_pretrained(abs_model_path, local_files_only=True)
        # Load quant config for quant_model.
        quant_config = load_json(q_model_path+'/quant_config.json')
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config=config, torch_dtype = dtype)
        quantize_model(model, init_only=True, _quant_config=quant_config)
        model.tie_weights()
        model.generation_config = gen_config
        load_checkpoint_in_model(model, checkpoint=q_model_path+'/q_model.pt',
                                 device_map={'': 0}, offload_state_dict=True)
        model.eval().cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            abs_model_path,
            local_files_only = True,
            torch_dtype = dtype,
            # attn_implementation='flash_attention_2',
            device_map  = "cpu"
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
    return model, tokenizer