
import argparse
from utils.util import load_json

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="qwen",
                    choices=['llama', 'qwen'], help="type of the model")
args = parser.parse_args()

# result = {'vram':peak_mem_gb, 'ttft':avg_ttft, 'tpot':avg_tpot}
# if args.full:
#     save_json('./tmp/perf_wiki2_full'+args.model_type, result)
# else:
#     save_json('./tmp/perf_wiki2_quant'+args.model_type, result)

def reduce_rate(subject, result_full, result_quant):
    return (result_full[subject]-result_quant[subject]) / result_full[subject]

result_full = load_json('./tmp/perf_wiki2_full_'+args.model_type)
result_quant = load_json('./tmp/perf_wiki2_quant_'+args.model_type)

vram_reduce_rate = reduce_rate('vram', result_full, result_quant)
if args.model_type =='qwen':
    model_name = 'Qwen2.5-7B-Instruct'
else: model_name = 'Meta-Llama3-8B'
print('='*47)
print(f"|    峰值显存降低报告:  {model_name:19s}   |")
print('='*47)
print(f"|  全精度模型推理峰值显存占用 |    {result_full['vram']:5.2f} GB   |")
print(f"|  量化后模型推理峰值显存占用 |    {result_quant['vram']:5.2f} GB   |")
print(f"|  显存降低率                 |    {vram_reduce_rate*100:.2f}  %   |")
print('-'*47)

# ttft_reduce_rate = reduce_rate('ttft', result_full, result_quant)
# tpot_reduce_rate = reduce_rate('tpot', result_full, result_quant)
print('='*47)
print(f"|     推理速度报告:      {model_name:19s}  |")
print('='*47)
print(f"|   指标 (ms/token)  | 全精度模型 | 量化后模型|")
print(f"|首个token延迟(TTFT) |  {result_full['ttft']:6.2f}    |   {result_quant['ttft']:6.2f}  |")
print(f"|后续token延迟(TPOT) |  {result_full['tpot']:6.2f}    |   {result_quant['tpot']:6.2f}  |")
print('-'*47)