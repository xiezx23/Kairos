import torch
from utils.color_print import *

device_name = torch.cuda.get_device_name(0)
print("In device:", color_text('gre', device_name))
if device_name == 'NVIDIA A100 80GB PCIe':
    # In SYSU 501 A100 Server
    model_root_path  = '/data/share/models'
    model_path_qwen2 = model_root_path+'/Qwen/Qwen2___5-7B-Instruct'
    model_path_llama = model_root_path+'/LLM-Research/Meta-Llama-3-8B'
    dataset_root_path = "/data/share/datasets"
elif device_name == 'NVIDIA A100-PCIE-40GB':
    # In China Mobile Cloud Server
    model_root_path  = '/test/models'
    model_path_qwen2 = model_root_path+'/Qwen2.5-7B-Instruct'
    # model_path_llama = model_root_path+'/llama3-8B/llama3-8B'
    model_path_llama = model_root_path+'/LLM-Research/Meta-Llama-3-8B'
    dataset_root_path = "/test/datasets"
else: assert 0, print('Unknown Device.')

if hasattr(torch, "cuda") and torch.cuda.is_available():
    device  = 'cuda'
    try:
        import awq_backend
        import edq_cuda_accel
        enable_accel = True
    except ImportError:
        awq_backend = None
        edq_cuda_accel = None
        enable_accel = False
elif hasattr(torch, "npu") and torch.npu.is_available():
    import torch_npu
    device = 'npu'
else:
    device = 'cpu'

# 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear'
quant_strategy = input('Please input the quantization strategy.\n')
# quant_strategy = 'DynamicLinear'
print(f'Using quantization strategy: {quant_strategy}')