import os
from utils.color_print import *
from utils.global_config import *
from modelscope import dataset_snapshot_download

dataset_name = 'mit-han-lab/pile-val-backup'
# Download calibration data
if (os.path.exists(dataset_root_path + dataset_name)):
    print(gre_prefix, 'Already have the dataset.', default_color)
else:
    print(f'You do not have {dataset_name}, do you want to download now?')
    print('Input y/n: ')
    ins = input()
    if ins != 'y': exit(0)
    data_dir = dataset_snapshot_download(dataset_name, cache_dir=dataset_root_path)
    os.system('zstd -d ' + data_dir + "/val.jsonl.zst")
    os.remove(data_dir + "/val.jsonl.zst")                 
    print(f'Finish download: {data_dir}')

# Download wikitext
dataset_name = 'modelscope/wikitext'
data_dir = dataset_snapshot_download(dataset_name, cache_dir=dataset_root_path)
