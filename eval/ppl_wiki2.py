import argparse
import time
import torch
import tqdm
from torch import nn

from utils.load_model import load_from_pretrained
from utils.color_print import colored_print
from datasets import load_dataset
from utils.global_config import *
from utils.util import save_json

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path to the model", )
parser.add_argument("--weight_path", type=str, default='', help="path to the pre-quanted weights", )
parser.add_argument("--full", type=bool, default=False, 
                help="Use full precision model or not.")
parser.add_argument("--model_type", type=str, default="qwen",
                    choices=['llama', 'qwen'], help="type of the model")
args = parser.parse_args()


@torch.no_grad()
def evaluate(model, tokenizer, dataset, seqlen=2048, device='cuda:0'):
    model.to(device)
    model.eval()
    model.seqlen = seqlen
    nlls, total_tokens = [], 0
    tokenized_dataset = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    tokenized_dataset = tokenized_dataset.input_ids
    # 整除计算批数，有余数的部分不计算
    nsamples = tokenized_dataset.numel() // seqlen
    for i in tqdm.tqdm(range(nsamples), desc="Evaluating perplexity"):
        # 逐批量转device, 避免显存占用过大
        input_ids = tokenized_dataset[:, i * seqlen:(i + 1) * seqlen].to(device)
        outputs = model(input_ids=input_ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                           shift_labels.view(-1))
        neg_log_likelihood = loss * seqlen
        nlls.append(neg_log_likelihood)
        total_tokens += input_ids.numel()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    return ppl.item()


if __name__ == '__main__':
    ########################################
    if args.model_type == 'qwen':
        args.model_path = model_path_qwen2
    elif args.model_type == 'llama':
        args.model_path =  model_path_llama
    ########################################
    
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

    """ 在WikiText-2测试集上评估模型PPL """
    dataset_path = dataset_root_path+"/wikitext"
    dataset_name = 'wikitext-2-v1'
    print('Eval PPL in Dateset:', dataset_name)
    
    dataset_test = load_dataset(dataset_path, dataset_name, 
                                trust_remote_code=False, split="test")

    metric = evaluate(model, tokenizer, dataset_test, seqlen=2048)

    colored_print(f"Evaluation on dataset: {dataset_name}", "note")
    colored_print(f"\tPPL: {metric:.5f}", "note")

    # 保存结果到文件
    if args.full:
        save_json('./tmp/ppl_wiki2_full_'+args.model_type, metric)
    else:
        save_json('./tmp/ppl_wiki2_quant_'+args.model_type, metric)