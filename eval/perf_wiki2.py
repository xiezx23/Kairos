import time
import torch
import tqdm
import random
from datasets import load_dataset
from typing import List, Dict, Tuple
import numpy as np
import gc

import argparse
from utils.load_model import load_from_pretrained
from utils.color_print import *
from utils.perf_eval import empty_cache, reset_peak_memory_stats, max_memory_allocated
from utils.calibration import generate_random_token_sequence
from utils.global_config import *
from utils.util import save_json

@torch.no_grad
def eval_perf(model, tokenizer, inputs, ttft_times=4, max_new_tokens=100):
    # 编码输入文本
    # text_input = text_input[0: len(text_input)//2]
    inputs_len = inputs['input_ids'].shape[1]
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 测量TTFT (Time To First Token)
    start_time = time.perf_counter()
    for _ in range (ttft_times):
        model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    first_token_time = time.perf_counter()
    ttft = (first_token_time - start_time) / ttft_times

    # 测量TPOT (Time Per Output Token)
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finish_time = time.perf_counter()
    num_tokens = outputs[0].shape[0] - inputs_len
    total_time = finish_time - start_time
    if num_tokens > 1:
        tpot = (total_time - ttft) / (num_tokens - 1)
    else: tpot = 0
    return {
        'ttft': ttft, 'tpot': tpot,
        'total_time': total_time,
        'num_tokens': num_tokens,
    }

def eval(model, tokenizer, num_samples=-1, max_new_tokens=50):
    for _ in tqdm.tqdm(range(30), desc="Preheating Model"):
        seq_length = random.randint(1, 4096)
        test_input = generate_random_token_sequence(tokenizer, device='cuda', seq_length=seq_length)
        model.generate(**test_input, max_new_tokens=20, pad_token_id = tokenizer.pad_token_id)
        torch.cuda.synchronize()
    """
    在WikiText-2测试集上评估模型性能
    """
    dataset_path = dataset_root_path+"/wikitext"
    dataset_name = 'wikitext-2-v1'
    print('Eval Performance in Dateset:', dataset_name)
    
    dataset_test = load_dataset(dataset_path, dataset_name, 
                                trust_remote_code=False, split="test")
    valid_texts = [text for text in dataset_test["text"] 
                  if text and len(text.strip()) > 20]  # 确保文本有效
    print('Total Text Counts:', len(valid_texts))
    
    if num_samples > 0 and len(valid_texts) > num_samples:
        valid_texts = valid_texts[:num_samples]
    
    ttft_list = []
    tpot_list = []
    total_times = []
    token_counts = []
    
    empty_cache()
    gc.collect()
    reset_peak_memory_stats()

    # eval batchsize = 1
    for text_input in tqdm.tqdm(valid_texts):
        inputs = tokenizer(text_input, return_tensors="pt")
        perf_results = eval_perf(model, tokenizer,
                                  inputs, max_new_tokens)
        ttft_list.append(perf_results['ttft'])
        tpot_list.append(perf_results['tpot'])
        total_times.append(perf_results['total_time'])
        token_counts.append(perf_results['num_tokens'])
    
    # 生成性能报告
    generate_performance_report(ttft_list, tpot_list, 
                                total_times, token_counts, len(valid_texts))

def generate_performance_report(ttft_list: List[float], tpot_list: List[float], 
                              total_times: List[float], token_counts: List[int], 
                              num_samples: int):
    # 计算统计指标
    avg_ttft = np.mean(ttft_list) * 1000  # 转换为毫秒
    avg_tpot = np.mean(tpot_list) * 1000  # 转换为毫秒
    avg_total_time = np.mean(total_times)
    avg_tokens_per_sample = np.mean(token_counts)
    
    # 计算吞吐量
    total_tokens = sum(token_counts)
    total_time = sum(total_times)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0

    peak_mem_bytes = max_memory_allocated()
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    
    # 输出性能报告
    print(gre_prefix+"------PERFORMANCE REPORT------"+default_color)
    print(f"总生成时间:      {total_time:.2f} s")
    print(f"峰值显存占用:    {peak_mem_gb:.2f} GB")
    print(f"平均生成token数: {avg_tokens_per_sample:.1f}")
    print(f"Average TTFT:    {avg_ttft:.2f} ms")
    print(f"Average TPOT:    {avg_tpot:.2f} ms/token")
    print(f"吞吐量:          {tokens_per_second:.2f} tokens/秒")
    print(gre_prefix+"------------------------------"+default_color)
    
    # 保存详细结果到文件
    # save_detailed_results(ttft_list, tpot_list, total_times, token_counts)
    result = {'vram':peak_mem_gb, 'ttft':avg_ttft, 'tpot':avg_tpot}
    if args.full:
        save_json('./tmp/perf_wiki2_full_'+args.model_type, result)
    else:
        save_json('./tmp/perf_wiki2_quant_'+args.model_type, result)


def save_detailed_results(ttft_list: List[float], tpot_list: List[float],
                         total_times: List[float], token_counts: List[int]):
    import csv
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_performance_{timestamp}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample', 'TTFT(ms)', 'TPOT(ms)', 'Total_Time(s)', 'Tokens_Generated'])
        
        for i, (ttft, tpot, total_time, tokens) in enumerate(zip(ttft_list, tpot_list, total_times, token_counts)):
            writer.writerow([i+1, ttft*1000, tpot*1000, total_time, tokens])
    
    print(f"\n详细结果已保存到: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model", )
    parser.add_argument("--weight_path", type=str, default='', help="path to the pre-quanted weights", )
    parser.add_argument("--full", type=bool, default=False, 
                    help="Use full precision model or not.")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=['llama', 'qwen'], help="type of the model")
    args = parser.parse_args()
    #####################################################################
    if args.model_type == 'qwen':
        args.model_path = model_path_qwen2
    elif args.model_type == 'llama':
        args.model_path =  model_path_llama
    #####################################################################

    model, tokenizer = load_from_pretrained(
        model_path=args.model_path,
        weight_path=args.weight_path+'/q_model.pt',
        model_type=args.model_type,
        seed=42,
        setup=True,
        device='cuda',
        trust_remote_code=False,
        local_files_only=True,
        quant_llm= not args.full,
    )
    eval(model, tokenizer, max_new_tokens=10, num_samples=128)
