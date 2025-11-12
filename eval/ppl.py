import argparse
import time
import torch
import tqdm
from torch import nn

from utils.load_model import load_from_pretrained
from utils.color_print import colored_print
from datasets import load_dataset


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model", )
    parser.add_argument("--weight_path", type=str, help="path to the pre-quantized weights", )
    parser.add_argument("--quant_llm", action="store_true", help='whether to use quantized llm')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, help="Logging level")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=['llama', 'qwen'], help="type of the model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=4 * 1024,
                        help="maximum number of new tokens to generate")
    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset to be used")
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()

    #####################################################################
    # 以下代码为临时使用
    args.model_path = '/data/share/models/Qwen/Qwen2___5-7B-Instruct'
    args.weight_path = "/home/ljy/workspace/llm_edq_cache/Qwen2___5-7B-Instruct_w8.pt"
    args.dataset_path = "/data/share/datasets/wikitext"
    args.dataset_name = "wikitext-2-raw-v1"
    args.quant_llm = False
    #####################################################################

    model, tokenizer = load_from_pretrained(
        model_path=args.model_path,
        weight_path=args.weight_path,
        model_type=args.model_type,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        setup=True,
        device=args.device,
        trust_remote_code=False,
        local_files_only=True,
        quant_llm=args.quant_llm,
    )

    model.to(args.device)

    dataset_test = load_dataset(args.dataset_path, args.dataset_name,
                                trust_remote_code=False,
                                split=args.split)

    metric = evaluate(model, tokenizer, dataset_test, seqlen=2048, device=args.device)

    colored_print(f"Evaluation on dataset: {args.dataset_name}", "note")
    colored_print(f"\tPPL: {metric:.5f}", "note")
