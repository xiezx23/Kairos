# https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py
import csv
import json
import argparse
import os
import torch
import random
import time
import re
from tqdm import tqdm
import logging
import sys
from datasets import load_dataset
from utils.global_config import *
from utils.load_model import load_from_pretrained

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048


def load_mmlu_pro(trust_remote_code=False):
    dataset = load_dataset(args.dataset_path, trust_remote_code=trust_remote_code)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
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

    sampling_params = dict(max_new_tokens=max_new_tokens,
                           num_beams=1,  # 单束搜索
                           do_sample=False,  # 评测不依赖随机采样，确保结果可复现
                           temperature=None,  # 0
                           top_k=None,  #
                           top_p=1)

    tokenizer.model_max_length = max_model_length
    tokenizer.padding_side = "left"  # 批量推理，padding在左边

    return model, tokenizer, sampling_params


def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model_type
    subjects = args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    prompt = ('The following are multiple choice questions (with answers) about {$}. '
              'Think step by step and then finish your answer with "the answer is (X)" '
              'where X is the correct letter choice.')

    subject = curr["category"]
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]
    prompt = prompt.replace("{$}", subject) + "\n"
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def batch_inference(model, sampling_params, inference_batch, tokenizer, subject, batch_size=4):
    start = time.time()

    response_batch = []
    pred_batch = []

    for i in tqdm(range(0, len(inference_batch), batch_size), desc=f'Evaluating {subject}'):
        batch_prompts = inference_batch[i: min(i + batch_size, len(inference_batch))]

        # 批量编码，设置padding=True自动处理不同长度的输入
        inputs = tokenizer(batch_prompts,
                           return_tensors="pt",
                           padding=True,  # 自动填充到批次中最长序列的长度
                           truncation=True,  # 超过最大长度的序列截断
                           max_length=tokenizer.model_max_length  # 使用模型最大长度
                           ).to(model.device)
        input_length = inputs.input_ids.shape[1]
        outputs = model.generate(**inputs, **sampling_params)

        for j in range(len(batch_prompts)):
            generated_text = tokenizer.decode(outputs[j][input_length:], skip_special_tokens=True)
            response_batch.append(generated_text)
            pred = extract_answer(generated_text)
            pred_batch.append(pred)

    logging.info(str(len(inference_batch)) + "size batch costing time: " + str(time.time() - start))
    return pred_batch, response_batch


def analyse_result(res):
    accu, corr, wrong = 0.0, 0.0, 0.0
    for each in res:
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, sampling_params, val_df, test_df, output_path, batch_size=16,
             save_context=False):
    global choices
    logging.info("evaluating " + subject)
    inference_batches = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1
        inference_batches.append(prompt)

    pred_batch, response_batch = batch_inference(model, sampling_params, inference_batches,
                                                 tokenizer, subject, batch_size)
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)

    accu, corr, wrong = analyse_result(res)
    logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))

    if save_context and output_path is not None:
        with open(output_path, "w") as fo:
            fo.write(json.dumps(res))

    return accu, corr, wrong


def main():
    # 加载本地模型
    model, tokenizer, sampling_params = load_model()
    # 加载本地数据
    full_test_df, full_val_df = load_mmlu_pro()

    # 选择需要评估的subject
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])

    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)
    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    print("selected subjects:\n" + "\n".join(selected_subjects))

    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")

    # 评估每个subject
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}

        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)

        len_subject = len(test_df)
        if len_subject > 24:
            test_df = test_df[0:24]
            val_df = test_df[0:24]

        output_path = os.path.join(save_result_dir, "{}.json".format(subject))

        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, sampling_params, val_df, test_df,
                                                output_path, args.batch_size, args.save_context)

        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))

    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="maximum case number for few shot")
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="../results")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=['llama', 'qwen'], help="type of the model")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--verbose", type=int, default=1, help="Logging level")
    parser.add_argument("--model_path", type=str, help="path to the model", )
    parser.add_argument("--weight_path", type=str, default='', help="path to the pre-quanted weights", )
    parser.add_argument("--quant_llm", default=False, help='whether to use quantized llm')
    parser.add_argument("--dataset_path", type=str, help="path to the dataset")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--save_context", action="store_true", help="whether to save the context of the inference")
    args = parser.parse_args()

    #####################################################################
    # 以下代码为临时使用
    # args.selected_subjects = ['engineering', 'chemistry']
    # args.selected_subjects = 'engineering,chemistry'
    args.selected_subjects = 'all'
    
    # args.model_type = 'llama'
    # args.model_type = 'llama'
    if args.model_type == 'qwen':
        args.model_path = model_path_qwen2
    elif args.model_type == 'llama':
        args.model_path =  model_path_llama
    
    args.dataset_path = dataset_root_path+'/MMLU-Pro'
    #####################################################################
    random.seed(args.seed)

    # 输出根目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 全局记录文件
    global_record_file = os.path.join(args.save_dir, args.global_record_file)
    # 结果输出目录
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )

    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    save_log_dir = os.path.join(args.save_dir, "log")
    summary_path = os.path.join(args.save_dir, "summary", file_name)

    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    os.makedirs(save_log_dir, exist_ok=True)

    # 信息同时输出到文件和控制台
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()
