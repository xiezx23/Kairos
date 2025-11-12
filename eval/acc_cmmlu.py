# https://github.com/haonan-li/CMMLU/blob/master/src/qwen2.py
import os
import torch
import numpy as np
import argparse
from eval.cmmlu.mp_utils import choices, format_example, gen_prompt, run_eval, softmax
from tqdm import tqdm
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from utils.load_model import load_from_pretrained
from utils.global_config import *


def is_eval_success(args) -> bool:
    """judege if eval task is success by checking the result dir"""
    subjects = sorted(
        [f.split(".csv")[0] for f in os.listdir(os.path.join(args.dataset_dir, f"{args.test_set_dir}/"))]
    )
    # abs_save_dir = f"{args.result_dir}/{args.num_few_shot}_shot"
    abs_save_dir = os.path.join(args.result_dir, f"{args.num_few_shot}_shot")
    if not os.path.exists(abs_save_dir):
        return False
    for subject in subjects:
        out_file = os.path.join(abs_save_dir, f"results_{subject}.csv")
        if not os.path.exists(out_file):
            # If any result file NOT exist, the eval isn't finished
            return False
    return True


def init_model(args):
    """Initialize models"""
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

    sampling_params = dict(max_new_tokens=args.max_new_tokens,
                           num_beams=1,  # 单束搜索
                           do_sample=False,  # 评测不依赖随机采样，确保结果可复现
                           temperature=None,  # 0
                           top_k=None,  #
                           top_p=1)

    return model, tokenizer, sampling_params

def eval(model, tokenizer, sampling_params, subject, dev_df, test_df, num_few_shot, max_length, cot):
    """eval Meta-Llama-3-8B, notice that the first token is 128000."""
    choice_ids = [tokenizer(choice)["input_ids"][-1] for choice in choices]
    # test = tokenizer(['A','B', 'C', 'D'])
    # print(test)
    # generated_text = tokenizer.decode(choice_ids, skip_special_tokens=True)
    # print(generated_text)
    cors = []
    all_conf = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        label = test_df.iloc[i, test_df.shape[1] - 1]

        with torch.no_grad():
            input_ids = tokenizer([prompt], padding=False)["input_ids"]
            input_ids = torch.tensor(input_ids, device=model.device)
            attention_mask = torch.ones(1, input_ids.shape[0], device = 'cuda')
            input = {'input_ids':input_ids, 'attention_mask':attention_mask}
            # outputs = model.generate(
            #     **input, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(generated_text)
            logits = model(**input, pad_token_id=tokenizer.eos_token_id)["logits"]
            last_token_logits = logits[:, -1, :]
            if last_token_logits.dtype in {torch.bfloat16, torch.float16}:
                last_token_logits = last_token_logits.to(dtype=torch.float32)
            choice_logits = last_token_logits[:, choice_ids].detach().cpu().numpy()
            conf = softmax(choice_logits[0])[choices.index(label)]
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choice_logits[0])]
            # print(pred)
        all_preds += pred
        all_conf.append(conf)
        cors.append(pred == label)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return acc, all_preds, None

def eval_instruct(
        model, tokenizer, sampling_params, subject, dev_df, test_df, num_few_shot, max_length, cot
):
    """eval Qwen/Qwen2-72B-Instruct
    ref: https://huggingface.co/Qwen/Qwen2-72B-Instruct#quickstart
    """
    cors = []
    all_preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in tqdm(range(test_df.shape[0]), desc=f'Evaluating {subject}'):
        prompt_end = format_example(test_df, i, subject, include_answer=False, cot=cot)
        prompt = gen_prompt(
            dev_df=dev_df,
            subject=subject,
            prompt_end=prompt_end,
            num_few_shot=num_few_shot,
            tokenizer=tokenizer,
            max_length=max_length,
            cot=cot,
        )
        label = test_df.iloc[i, test_df.shape[1] - 1]

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
        generated_ids = model.generate(**model_inputs, **sampling_params, pad_token_id=tokenizer.eos_token_id)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if pred and pred[0] in choices:
            cors.append(pred[0] == label)
        all_preds.append(pred.replace("\n", ""))

    acc = None
    # acc = np.mean(cors)
    # print("Average accuracy {:.3f} - {}".format(acc, subject))
    # print("{} results, {} inappropriate formated answers.".format(
    #         len(cors), len(all_preds) - len(cors)))
    return acc, all_preds, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model", )
    parser.add_argument("--weight_path", type=str, default='', help="path to the pre-quanted weights", )
    parser.add_argument("--quant_llm", type=bool, default=False, help='whether to use quantized llm')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, help="Logging level")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=['llama', 'qwen'], help="type of the model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="maximum number of new tokens to generate")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--dataset_dir", type=str, help="path to the dataset")
    parser.add_argument("--train_set_dir", type=str, default='dev')
    parser.add_argument("--test_set_dir", type=str, default='test')

    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--result_dir", type=str, default="./results/eval/cmmlu/")
    parser.add_argument("--num_few_shot", type=int, default=5)
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    #####################################################################
    # args.model_type = 'llama'
    if args.model_type == 'qwen':
        args.model_path = model_path_qwen2
        eval_func = eval_instruct
    elif args.model_type == 'llama':
        args.model_path =  model_path_llama
        eval_func = eval

    args.dataset_dir = dataset_root_path+'/cmmlu'
    args.test_set_dir = 'test' # In Local A100: 'test_all' / 'test' / 'sampled'
    args.weight_path += '/q_model.pt'

    args.cot = False
    args.num_few_shot = 0
    #####################################################################
    if is_eval_success(args):
        # eval finished, no need load model anymore, just show the result
        model, tokenizer, sampling_params = None, None, None
    else:
        model, tokenizer, sampling_params = init_model(args)

    # 本项目仅讨论instruct模型
    run_eval(model, tokenizer, sampling_params, eval_func, args)
