import torch
import os
import json
import numpy as np

def device_warmup(device: str):
    warm_up = torch.randn((8192, 8192)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 可选：设置 PyTorch 全局确定性
    # torch.use_deterministic_algorithms(True)

def save_json(save_path, obej):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open((save_path), 'w') as f:
        f.write(json.dumps(obej))
        f.flush()

def load_json(load_path):
    return json.load(open((load_path)))

def save_text(save_path, text):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open((save_path), 'w') as f:
        f.write(text)
        f.close()