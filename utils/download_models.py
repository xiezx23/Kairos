import os
from utils.color_print import *
from utils.global_config import *
from modelscope import snapshot_download

# 下载 Qwen2.5 7B Instruct
if (os.path.exists( model_path_qwen2)):
    print(gre_prefix, f'Already have {model_path_qwen2}', default_color)
else:
    print(f'You do not have {model_path_qwen2}, do you want to download now?')
    print('Input y/n: ')
    ins = input()
    if ins == 'y':
        model_dir = snapshot_download('Qwen/Qwen2___5-7B-Instruct', cache_dir=model_root_path)
        print(f'Finish download: {model_path_qwen2}')

# 下载 Llama3 8B
if (os.path.exists( model_path_llama)):
    print(gre_prefix, f'Already have {model_path_llama}', default_color)
else:
    print(f'You do not have {model_path_llama}, do you want to download now?')
    print('Input y/n: ')
    ins = input()
    if ins == 'y':
        model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B', cache_dir=model_root_path)
        print(f'Finish download: {model_path_llama}')