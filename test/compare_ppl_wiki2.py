
import argparse
from utils.util import load_json

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="qwen",
                    choices=['llama', 'qwen'], help="type of the model")
args = parser.parse_args()

result_f = load_json('./tmp/ppl_wiki2_full_'+args.model_type)
result_q = load_json('./tmp/ppl_wiki2_quant_'+args.model_type)

reduce = -(result_f-result_q) / result_f

if args.model_type =='qwen':
    model_name = 'Qwen2.5-7B-Instruct'
else: model_name = 'Meta-Llama3-8B'
print('='*47)
print(f"|  模型Wiki2困惑度报告:  {model_name:19s}  |")
print('='*47)
print(f"|  全精度模型困惑度 Perplexity |    {result_f:5.2f}     |")
print(f"|  量化后模型困惑度 Perplexity |    {result_q:5.2f}     |")
print(f"|  困惑度增加率                |     {reduce*100:.2f}%    |")
print('-'*47)
