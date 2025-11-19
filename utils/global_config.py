import torch
from utils.color_print import *

device_name = torch.cuda.get_device_name(0)
print("In device:", color_text('gre', device_name))
device = 'cuda'
model_root_path  = '/data/share/models'
model_path_qwen2 = model_root_path+'/Qwen/Qwen2___5-7B-Instruct'
model_path_llama = model_root_path+'/LLM-Research/Meta-Llama-3-8B'
dataset_root_path = "/data/share/datasets"

try:
    import kairos_cuda_accel
    enable_accel = True
except ImportError:
    kairos_cuda_accel = None
    enable_accel = False

# 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear'
quant_strategy = input('Please input the quantization strategy.\n')
# quant_strategy = 'DynamicLinear'
print(f'Using quantization strategy: {quant_strategy}')