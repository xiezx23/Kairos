import argparse
import time
import torch

from utils.load_model import load_from_pretrained
from utils.color_print import colored_print
from edge_chatbot.context_manager import ContextManager
from transformers import TextStreamer

from utils.perf_eval import ProfilingTextStreamer

# # 设置可见的 CUDA 设备为 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


def single_round_with_profile(inputs, streamer: ProfilingTextStreamer, past_key_values=None):
    streamer.reset()

    # 仅在profile_mem模式下清空缓存，尽量避免影响推理时间开销
    if args.profile_mem:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    begin_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=past_key_values,
            return_dict_in_generate=True,
            return_legacy_cache=False,  # 避免将返回的past_key_values转为tuple
            streamer=streamer,
        )
    end_time = streamer.finish_time

    metric = dict()
    if args.profile_mem:
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        metric['peak_mem_usage'] = peak_mem_gb

    total_time = end_time - begin_time
    actual_tokens = streamer.token_count - 1
    if streamer.first_token_time is not None:
        ttft = (streamer.first_token_time - begin_time) * 1000
    else:
        ttft = total_time * 1000

    if actual_tokens > 1:
        tpot = ((total_time - (streamer.first_token_time - begin_time)) / (actual_tokens - 1)) * 1000  # 毫秒/token
    elif actual_tokens == 1:
        tpot = 0
    else:
        tpot = float('nan')

    metric['prompt_length'] = streamer.prompt_len
    metric['generate_time'] = total_time
    metric['generate_token'] = actual_tokens
    metric['TTFT'] = ttft
    metric['TPOT'] = tpot

    outputs['metric'] = metric
    return outputs


def stream_chat_with_profile():
    # 初始化上下文管理器
    context_manager = ContextManager(
        tokenizer=tokenizer,
        model_max_length=getattr(model.config, "max_position_embeddings", 8 * 1024),
        max_ctx_to_keep=args.max_ctx_to_keep,
        user_role=USER_ROLE,
        assistant_role=ASSISTANT_ROLE,
        verbose=args.verbose,
        quant_kv=args.quant_kv,
    )

    streamer = ProfilingTextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    k = 0
    while True:
        if k == len(prompts):
            break
        input_prompt = prompts[k]
        k += 1

        # 添加用户消息到上下文
        context_manager.add_user_message(input_prompt)

        # 检查并执行必要的截断
        context_manager.truncate_if_needed()

        # 生成当前轮的完整Prompt
        prompt = context_manager.get_prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]

        # 获取当前的past_key_values
        past_key_values = context_manager.get_past_key_values()

        outputs = single_round_with_profile(inputs, streamer, past_key_values)
        metric = outputs['metric']
        if args.verbose > 0:
            colored_print('=====PERFORMANCE REPORT=====', color='BLUE')
            colored_print(f"prompt长度:    {metric['prompt_length']} token", color='BLUE')
            if args.profile_mem:
                colored_print(f"峰值显存占用:   {metric['peak_mem_usage']:.2f} GB", color='BLUE')
            colored_print(f"总生成时间:     {metric['generate_time']:.2f} s", color='BLUE')
            colored_print(f"生成token数:   {metric['generate_token']}", color='BLUE')
            colored_print(f"TTFT:         {metric['TTFT']:.2f} ms", color='BLUE')
            colored_print(f"TPOT:         {metric['TPOT']:.2f} ms/token", color='BLUE')
            colored_print('============================', color='BLUE')

        if args.single_round:
            context_manager.reset()
        else:
            # 更新KV Cache
            context_manager.set_past_key_values(outputs.past_key_values)

            # 添加助手回复到上下文
            generated_tokens = outputs.sequences[0][input_length:]
            model_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            context_manager.add_assistant_message(model_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path to the model", )
    parser.add_argument("--weight_path", type=str, help="path to the pre-quanted weights", )
    parser.add_argument("--quant_llm", action="store_true", help='whether to use quantized llm')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1, help="Logging level")
    parser.add_argument("--quant_kv", action="store_true", help="whether to quantize kv cache.")
    parser.add_argument("--profile_mem", action="store_true",
                        help="whether to profile the memory consumption.")
    parser.add_argument("--model_type", type=str, default="qwen",
                        choices=['llama', 'qwen'], help="type of the model")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=4 * 1024,
                        help="maximum number of new tokens to generate")
    parser.add_argument("--max_ctx_to_keep", type=float, default=0.75,
                        help="maximum context length to keep: positive integer for absolute value, 0<float<1 for ratio")
    parser.add_argument("--single_round", action="store_true",
                        help="whether to memorize previous conversations")
    parser.add_argument("--truncate_kv", action="store_true", help="whether to truncate kv cache.")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
                        help="data type for the model")
    args = parser.parse_args()

    #####################################################################
    # 以下代码为临时使用
    args.model_path = '/data/share/models/Qwen/Qwen2___5-7B-Instruct'
    args.weight_path = "/home/ljy/workspace/llm_edq_cache/Qwen2___5-7B-Instruct_w8_acc.pt"
    args.profile = True
    args.profile_mem = True
    # args.single_round = True
    args.max_ctx_to_keep = 64
    args.quant_kv = True
    args.truncate_kv = args.truncate_kv and not args.quant_kv  # 目前不支持量化kv-cache的截断
    args.quant_llm = True
    #####################################################################
    # 测试模式的参数配置
    # TODO: 选择合适的 prompts
    prompts = [
        "Hello, what is your name?",
        "Can you speak in Chinese? Write a simple poetry with it.",
        "Design a simple neural network for image classification.",
        "A cage has 35 heads and 94 legs. How many chickens and rabbits?",
    ]
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
    stream_chat_with_profile()
