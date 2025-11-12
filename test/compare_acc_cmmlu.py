import argparse
from utils.util import load_json

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="qwen",
                    choices=['llama', 'qwen'], help="type of the model")
args = parser.parse_args()

if args.model_type =='qwen':
    model_name = 'Qwen2.5-7B-Instruct'
    result_full  = load_json('./results/qwen/full/cmmlu/0_shot/overall')
    result_quant = load_json('./results/qwen/quant/cmmlu/0_shot/overall')
else: 
    model_name = 'Meta-Llama3-8B'
    result_full  = load_json('./results/llama/full/cmmlu/0_shot/overall')
    result_quant = load_json('./results/llama/quant/cmmlu/0_shot/overall')
reduce = (result_full - result_quant) / result_full
if reduce < 0:
    reduce = 0

print('='*47)
print(f"|    模型能力降低报告:  {model_name:19s}   |")
print('='*47)
print(f"|  全精度模型 CMMLU 综合能力  |    {result_full:5.2f}      |")
print(f"|  量化后模型 CMMLU 综合能力  |    {result_quant:5.2f}      |")
print(f"|  能力降低率                 |     {reduce*100:.2f}  %   |")
print('-'*47)
