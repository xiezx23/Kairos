import argparse
import os
import threading
import time

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils, GenerationConfig
from transformers import TextIteratorStreamer

from utils.util import device_warmup
from utils.color_print import colored_print
from edq.quant_model import quantize_model

# # 设置可见的 CUDA 设备为 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEBUG = True
if DEBUG:
    prompts = [
        "Hello, what is your name?",
        "Can you speak in Chinese? Write a simple poetry with it.",
        "Design a simple neural network for image classification.",
        "A cage has 35 heads and 94 legs. How many chickens and rabbits?",
    ]
else:
    prompts = None


def setup(debug=False):
    if debug:
        from utils.util import set_seed
        set_seed(args.seed)
    args.torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # 禁用模型初始化权重: 在加载量化模型时，会提前创建模型实例，此时禁止参数初始化可以加速
    modeling_utils._init_weights = False
    # 设置默认数据类型
    torch.set_default_dtype(args.torch_dtype)

    def skip(*args, **kwargs):
        pass  # 空函数，用于跳过 PyTorch 的初始化操作

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # 首次将数据（warm_up 矩阵）移动到 GPU 时，需要初始化设备上下文、分配显存空间，这会产生一次性开销。
    # 预热阶段通过主动执行内存密集型操作（大矩阵存储与计算），提前完成这些初始化步骤，确保后续模型加载和推理时内存分配高效稳定。
    device_warmup(args.device)


# 统一 prompt 模板
def make_prompt(h):
    # history: [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]

    return tokenizer.apply_chat_template(
        h,
        tokenize=False,
        add_generation_prompt=True,
    )


def update_history(h, role, content):
    h.append({"role": role, "content": content})
    return h


def chat_round(h):
    """
    history: 当前对话历史
    返回：assistant 回复字符串、本轮 TTFT、TBT 列表
    """
    start_time = time.time()  # 记录本轮开始
    prompt = make_prompt(h)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 计时变量
    ttft = None
    tbt_list = []
    first_token_time = None

    def generate():
        nonlocal first_token_time
        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer
            )

    # 启动生成线程
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    # 主线程消费流式文本
    assistant = ""
    for new_text in streamer:
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start_time
        else:
            tbt_list.append(time.time() - first_token_time)
            first_token_time = time.time()
        assistant += new_text
        print(new_text, end="", flush=True)  # 实时打印

    return assistant, ttft, tbt_list


def load_from_pretrained():
    # 加载模型配置\词表\量化模型
    # NOTE: 考虑到边端设备可能没有网络(封闭性测试), 这里要求所有的数据信息均来自本地

    # config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False,
                                        local_files_only=True)
    # generation_config
    gen_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=False,
                                                  local_files_only=True)
    gen_config.max_new_tokens = args.max_seq_len

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False,
                                              local_files_only=True)
    # 部分生成类模型（特别是 decoder-only 的 GPT、Llama 系列）在训练时只用 eos_token 做句子结束标记，没有定义 pad_token。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    # 方式一
    # 量化后的模型架构与Qwen\Llama不一致, 不能直接如此加载
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=False,
    #                                              local_files_only=True, device_map={"": args.device},
    #                                              torch_dtype=args.torch_dtype)

    # 方式二/三
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=args.torch_dtype)
    quantize_model(model, init_only=True)  # 仅替换量化层
    # FIXME: 该方法会导致模型的权重被绑定在一起，是否需要？ 配置中"tie_word_embeddings": false
    model.tie_weights()
    # load_checkpoint_and_dispatch 内部会调用load_checkpoint_in_model
    model = load_checkpoint_and_dispatch(model,
                                         checkpoint=args.weight_path,
                                         device_map={"": args.device},
                                         dtype=args.torch_dtype,
                                         offload_state_dict=True)
    # load_checkpoint_in_model(model,
    #                          checkpoint=args.weight_path,
    #                          device_map={"": args.device},
    #                          dtype=args.torch_dtype,
    #                          offload_state_dict=True)
    # model = model.to(args.device)  # load_checkpoint_in_model不会自动做这步

    model.generation_config = gen_config
    model.eval()

    # 进行一次前向传播，预热模型
    model(input_ids=torch.randint(0, 1000, (1, args.max_seq_len // 2), device=args.device),
          attention_mask=torch.ones(1, args.max_seq_len // 2, device=args.device))

    return model, tokenizer


def stream_chat(debug=False, single_round=False):
    history = []  # 每轮 append user / assistant
    round_id = 0

    print("=== 多轮对话开始，直接回车退出 ===")
    k = 0
    while True:
        if debug:
            if k == len(prompts):
                break
            user_input = prompts[k]
            k += 1
        else:
            try:
                user_input = input("\nUSER: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nEXIT...")
                break
            if not user_input:
                print("EXIT...")
                break

        history = update_history(history, "user", user_input)

        assistant, ttft, tbt_list = chat_round(history)
        total_tokens = len(tbt_list) + 1  # 首 token + 后续 token
        avg_tbt = sum(tbt_list) / len(tbt_list) if tbt_list else 0.0

        history = update_history(history, "assistant", assistant)

        if single_round:
            history = []  # 单轮对话，清空历史

        # 打印统计
        colored_print(f"\n[Round {round_id}] TTFT={ttft * 1000:.1f}ms, "
                      f"TBT={avg_tbt * 1000:.1f}ms, #Tokens={total_tokens}", color='BLUE')
        round_id += 1


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_llm", action="store_true", help="whether to use full precision model")
    parser.add_argument("--model_path", type=str, help="path to the quantized model with config files", )
    parser.add_argument("--weight_path", type=str, help="path to the checkpoint file", )
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])  # FIXME
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="maximum sequence length")
    parser.add_argument("--single_round", action="store_true", help="whether to memorize previous conversations")
    args = parser.parse_args()

    #####################################################################
    args.model_path = '/data/share/models/Qwen/Qwen2___5-7B-Instruct'
    args.weight_path = "/home/ljy/workspace/llm_edq_cache/Qwen2___5-7B-Instruct_w8.pt"
    #####################################################################

    setup(debug=DEBUG)
    model, tokenizer = load_from_pretrained()
    stream_chat(debug=DEBUG, single_round=args.single_round)
