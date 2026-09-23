import gc
import os
import torch
from accelerate import init_empty_weights, load_checkpoint_in_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig

from kairos.quant_model import quantize_model
from utils.util import load_json
from utils.color_print import *

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
            # device_map  = "cpu" # Used in transformers v4
        ).eval()
    tokenizer = AutoTokenizer.from_pretrained(abs_model_path)
    return model, tokenizer