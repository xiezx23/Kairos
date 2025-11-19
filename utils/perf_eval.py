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