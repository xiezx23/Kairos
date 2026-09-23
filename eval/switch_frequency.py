import pandas as pd
import json
import torch
from datasets import load_dataset
from modelscope.msdatasets import MsDataset

# from datasets import load_dataset

# datasets = ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "gov_report", "qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

# for dataset in datasets:
#     data = load_dataset('THUDM/LongBench', dataset, split='test')

# exit(0)

datasets = ['BurstGPT', 'ShareGPT', 'LongBench']
for dataset in datasets:
    path = './tmp/input_len'+dataset+'.pt'
    if 1:
        if dataset == 'BurstGPT':
            data = pd.read_csv('/data/xiezx/BurstGPT/data/BurstGPT_1.csv')
            filtered_data = data[data['Response tokens'] > 0]
            input_len = filtered_data['Request tokens']
            print(max(input_len), min(input_len))
        elif dataset == 'ShareGPT':
            input_len = []
            data = MsDataset.load('swift/sharegpt', 'common-en', trust_remote_code=False, split='train', cache_dir="/data/xiezx/")
            for conversation in data:
                for turn in conversation['conversation']:
                    input_len.append(len(turn['human']))
        elif dataset == 'LongBench':
            input_len = []
            ds =  MsDataset.load('wht1600421526/longbench_v2_without_chinese', trust_remote_code=False, subset_name='default', split='train', cache_dir="/data/xiezx/")
            for q in ds:
                input_len.append(len(q["context"]))
        torch.save(input_len, path)
    else:
        input_len = torch.load(path)
exit(0)
    
from kairos.dynamic_linear_v2 import DynamicLinear
from kairos.perf_profiler_v2 import Profiler
prof_path = './prof_llama/'
shape_list = [(1024, 4096), (4096, 4096), (4096, 14336), (14336, 4096)]
shape_numb = [2,2,2,1]

DynamicLinear.prof = Profiler(shape_list)
DynamicLinear.prof.load(prof_path)

record = dict()

for m in input_len:
    for i in range(len(shape_list)):
        w = shape_list[i]; n = shape_numb[i]
        comp_type = DynamicLinear.prof.get_comp_type(w, m)
        # print(f'm={m}  {comp_type}')
        if not record.get(comp_type): 
            record[comp_type] = n
        else:
            record[comp_type] += n

COMP_TYPE_NAME = {'w4a16_gemm':0, 'w4a16_gemv':1, 'w8a8':2,\
                #    'w8a16_gemm':3, 'w8a16_gemv':4, \
                'fp16':5, 'w4a16_marlin': 6, 'w4a8_qserve':7}
print(record)

# BurstGPT={'FP16 GEMM': 2632344, 'W8A8 GEMM': 2370540, 'W4A16 GEMM MARLIN': 3258056, 'W4A16 GEMV': 1558926, 'W4A16 GEMM': 10192}
# ShareGPT={'FP16 GEMM': 287092, 'W4A16 GEMV': 546066, 'W4A16 GEMM MARLIN': 679321, 'W8A8 GEMM': 135473, 'W4A16 GEMM': 52278}
# LongBench={'FP16 GEMM': 832, 'W8A8 GEMM': 2080}




