# run: CUDA_VISIBLE_DEVICES=2 python -m test.visible_device

import torch

print("可用GPU数量:", torch.cuda.device_count())
print("当前GPU索引:", torch.cuda.current_device())
print("GPU名称:", torch.cuda.get_device_name(0))