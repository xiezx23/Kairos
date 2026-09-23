import os
from utils.color_print import *
from utils.global_config import *
from modelscope import snapshot_download

# model_path_qwen3_5_9B = model_root_path + '/Qwen/Qwen3.5-9B'
# model_dir = snapshot_download('Qwen/Qwen3.5-9B', cache_dir=model_root_path)

# model_path_qwen3_5_4B = model_root_path + '/Qwen/Qwen3.5-4B'
# model_dir = snapshot_download('Qwen/Qwen3.5-4B', cache_dir=model_root_path)

# model_path_qwen3_8B = model_root_path + '/Qwen/Qwen3-8B'
# model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir=model_root_path)

# model_path_qwen2_5_coder_1_5B = model_root_path + '/Qwen/Qwen2.5-Coder-1.5B'
# model_dir = snapshot_download('Qwen/Qwen2.5-Coder-1.5B', cache_dir=model_root_path)

# model_dir = snapshot_download('Qwen/Qwen2.5-Coder-0.5B', cache_dir=model_root_path)

model_dir = snapshot_download('Comfy-Org/Qwen3.8-27B', cache_dir=model_root_path)
# # Download Qwen2.5 7B Instruct
# if (os.path.exists( model_path_qwen2)):
#     print(gre_prefix, f'Already have {model_path_qwen2}', default_color)
# else:
#     print(f'You do not have {model_path_qwen2}, do you want to download now?')
#     print('Input y/n: ')
#     ins = input()
#     if ins == 'y':
#         model_dir = snapshot_download('Qwen/Qwen2___5-7B-Instruct', cache_dir=model_root_path)
#         print(f'Finish download: {model_path_qwen2}')

# # Download LLaMa-3 8B
# if (os.path.exists( model_path_llama)):
#     print(gre_prefix, f'Already have {model_path_llama}', default_color)
# else:
#     print(f'You do not have {model_path_llama}, do you want to download now?')
#     print('Input y/n: ')
#     ins = input()
#     if ins == 'y':
#         model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B', cache_dir=model_root_path)
#         print(f'Finish download: {model_path_llama}')