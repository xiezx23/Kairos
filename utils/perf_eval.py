from transformers import TextStreamer
import time
import tqdm
import gc
import torch
from contextlib import contextmanager
from utils.calibration import generate_random_token_sequence
from utils.color_print import *
if torch.cuda.is_available():
    device = 'cuda'
    synchronize = torch.cuda.synchronize
    max_memory_allocated = torch.cuda.max_memory_allocated
    empty_cache = torch.cuda.empty_cache
    reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
else:
    device = 'cpu'

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

@contextmanager
def timer(desc="Execution time", n = 1, recordList = None, print_flag = True):
    start.record()
    try:
        yield
    finally:
        end.record()
        synchronize()
        # torch.cuda.current_stream().synchronize()
        exe_time = (start.elapsed_time(end) / n)
        if recordList is not None:
            recordList.append(exe_time)
        if print_flag:
            print(f"{desc}: {exe_time:5.4f} ms")

@contextmanager
def mem_monitor(desc="Execution"):
    empty_cache()
    gc.collect()
    reset_peak_memory_stats()
    try:
        yield
    finally:
        peak_mem_bytes = max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        print(f"{desc} peak VRAM usage: {peak_mem_gb:.2f} GB")

class ProfilingTextStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=False, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.first_token_time = None
        self.token_count = 0
        self.prefill_flag = True
    
    def put(self, token_ids):
        if self.token_count == 0:
            self.prompt_len = token_ids.shape[-1]
        else:
            if self.prefill_flag:
                self.first_token_time = time.time()
                self.prefill_flag = False
        self.token_count += len(token_ids)
        assert(len(token_ids) == 1)
        super().put(token_ids)
    
    def end(self):
        self.finish_time = time.time()
        super().end()

    def reset(self):
        self.first_token_time = None
        self.token_count = 0
        self.prefill_flag = True

class InferModel():
    @torch.no_grad()
    def __init__(self, model, tokenizer):
        self.streamer = ProfilingTextStreamer(
            tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True
        )
        self.model = model
        self.pad_token_id = tokenizer.eos_token_id
        # return
        m_list = [1, 2, 4, 32, 64, 128, 256, 512, 1024, 1024*2, 1024*4, 1024*8, 1024*16]
        for i in tqdm.tqdm(range(len(m_list)), desc="Preheating Model"):
            test_input = generate_random_token_sequence(tokenizer, device='cuda', seq_length=m_list[i])
            model.generate(**test_input, max_new_tokens=10, pad_token_id = self.pad_token_id)

    def infer(self, prompt, inputs, max_new_tokens = 300):
        print(gre_prefix+"---------INPUT PROMPT---------"+default_color)
        print(prompt)
        print(gre_prefix+"---------MODEL OUTPUT---------"+default_color)
        self.streamer.reset()
        empty_cache()
        gc.collect()
        reset_peak_memory_stats()
        begin_time = time.time()
        with torch.no_grad():
            self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=self.streamer,
                pad_token_id = self.pad_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        end_time = self.streamer.finish_time
        peak_mem_bytes = max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        total_time = end_time - begin_time
        actual_tokens = self.streamer.token_count - 1
        if self.streamer.first_token_time is not None:
            ttft = (self.streamer.first_token_time - begin_time) * 1000
        else:
            ttft = total_time * 1000
        if actual_tokens > 1:
            tpot = ((total_time - (self.streamer.first_token_time - begin_time)) /
                     (actual_tokens - 1)) * 1000  # ms/token
        elif actual_tokens == 1:
            tpot = 0 
        else:
            tpot = float('nan')
        # Output Performance Report
        print(gre_prefix+"------PERFORMANCE REPORT------"+default_color)
        print(f"Prompt Length:   {self.streamer.prompt_len}")
        print(f"Peak Mem. Usage: {peak_mem_gb:.2f} GB")
        print(f"Total Exe. Time: {total_time:.2f} s")
        print(f"Output Token Num:{actual_tokens}")
        print(f"TTFT:            {ttft:.2f} ms")
        print(f"TPOT:            {tpot:.2f} ms/token")
        print(gre_prefix+"------------------------------"+default_color)


@torch.no_grad
def eval_perf_v2(model, tokenizer, inputs, ttft_times=10, max_new_tokens=100):
    
    warmup_steps = 5
    for _ in range(warmup_steps):
        model.generate(
            **inputs,
            max_new_tokens=20,
            max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    start_prefill = time.perf_counter()
    for _ in range(ttft_times):
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'], use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
    ttft = (time.perf_counter() - start_prefill) / ttft_times
    
    current_input = next_token
    
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                max_length=None,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            # print(past_key_values.layers[0].keys.shape)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        current_input = next_token
    t1 = time.perf_counter()
    decode_times = t1 - t0
    tpot = decode_times / (max_new_tokens)
    # print(f"TTFT (Prefill)   : {ttft*1000:.2f} ms")
    # print(f"TPOT (avg decode): {tpot*1000:.2f} ms")
    return {
        'ttft': ttft, 'tpot': tpot,
        'total_time': (t1-start_prefill),
        'num_tokens': max_new_tokens,
    }


@torch.no_grad
def eval_perf(model, tokenizer, inputs, ttft_times=10, max_new_tokens=100):
    # text_input = text_input[0: len(text_input)//2]
    inputs_len = inputs['input_ids'].shape[1]
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}

    warmup_steps = 5
    for _ in range(warmup_steps):
        _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    start_time = time.perf_counter()
    for _ in range (ttft_times):
        model.generate(
            **inputs,
            max_new_tokens=1,
            max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    first_token_time = time.perf_counter()
    ttft = (first_token_time - start_time) / ttft_times

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length = None,
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

@torch.no_grad
def eval_batch_perf(model, tokenizer, input_seq_len, batch_size, ttft_times=10, max_new_tokens=100):
    input_ids = torch.randint(1,12, (batch_size, input_seq_len)).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    warmup_steps = 5
    for _ in range(warmup_steps):
        _ = model.generate(input_ids, attention_mask=attention_mask,
                           max_new_tokens=max_new_tokens, max_length = None,
                           pad_token_id=tokenizer.eos_token_id)

    start_time = time.perf_counter()
    for _ in range (ttft_times):
        model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=1, max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    first_token_time = time.perf_counter()
    ttft = (first_token_time - start_time) / ttft_times

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    finish_time = time.perf_counter()
    num_tokens = (outputs[0].shape[0] - input_seq_len)
    total_time = finish_time - start_time
    if num_tokens > 1:
        tpot = (total_time - ttft) / (num_tokens - 1)
    else: tpot = 0
    return {
        'ttft': ttft, 'tpot': tpot,
        'total_time': total_time,
        'num_tokens': num_tokens,
    }


@torch.no_grad
def eval_perf_v3(model, tokenizer, inputs, ttft_times=10, max_new_tokens=100, batch_size=1):
    warmup_steps = 5
    for _ in range(warmup_steps):
        model.generate(
            **inputs,
            max_new_tokens=20,
            max_length = None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    attention_mask = inputs['attention_mask']
    start_prefill = time.perf_counter()
    for _ in range(ttft_times):
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'],attention_mask=attention_mask, use_cache=True)
            past_key_values = outputs.past_key_values
            last_token_indices = (attention_mask.sum(dim=1) - 1).long()
        next_token_logits = outputs.logits[torch.arange(batch_size), last_token_indices]
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
    ttft = (time.perf_counter() - start_prefill) / ttft_times
    
    current_input = next_tokens
    new_attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=model.device)], dim=1)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input,
                attention_mask=new_attention_mask,
                past_key_values=past_key_values,
                max_length=None,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            # print(past_key_values.layers[0].keys.shape)
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        current_input = next_tokens
        new_attention_mask = torch.cat(
            [new_attention_mask, torch.ones(batch_size, 1, device=model.device)], dim=1
        )
    t1 = time.perf_counter()
    decode_times = t1 - t0
    tpot = decode_times / (max_new_tokens)
    # print(f"TTFT (Prefill)   : {ttft*1000:.2f} ms")
    # print(f"TPOT (avg decode): {tpot*1000:.2f} ms")
    return {
        'ttft': ttft, 'tpot': tpot,
        'total_time': (t1-start_prefill),
        'num_tokens': max_new_tokens,
    }
