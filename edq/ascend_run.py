import torch
import time
import os
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
    TextStreamer, QuantoQuantizedCache, QuantizedCacheConfig
)
from edq.quant_model import quantize_model
from utils.perf_eval import InferModel

if torch.cuda.is_available():
    import edq_cuda_accel
    import awq_backend
    device = 'cuda'
elif torch.npu.is_available():
    import torch_npu
    device = 'npu:0'
else:
    device = 'cpu'

torch.set_default_dtype(torch.float16)
torch.npu.set_compile_mode(jit_compile=False)

# Avoid ReduceProd operator core dump, see more in: https://github.com/cosdt/llm/issues/4
# option={}
# option["NPU_FUZZY_COMPILE_BLACKLIST"]="ReduceProd"
# torch.npu.set_option(option)

torch.npu.set_device(0)

device = "npu:0"

model_path = "../Qwen/Qwen2___5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map=device).eval()

quantize_model(model)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# prompt = "Which team won the Champions League in 2020?"
prompt = '介绍一下中国移动湾区研究院。'
inputs = tokenizer(prompt, return_tensors="pt").to(device)

infer_model = InferModel(model, tokenizer)
infer_model.infer(prompt, inputs, 100)