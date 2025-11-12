import tqdm
from eval.perf_wiki2 import eval_perf
from utils.load_model import load_from_pretrained
from utils.calibration import generate_random_token_sequence


quant_llm = True
model_type= 'qwen'

if __name__ == "__main__":
    # model_type = 'llama'
    if model_type == 'qwen':
        model_path = '/data/share/models/Qwen/Qwen2___5-7B-Instruct'
        # weight_path = "/data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct_v3/q_model.pt"
        weight_path = "/data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct_v2/q_model.pt"
        # weight_path = "/data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct/q_model.pt"
        # weight_path = "/data/share/models/quant_model/edq_model/q_model.pt"
    elif model_type == 'llama':
        model_path =  '/data/share/models/LLM-Research/Meta-Llama-3-8B'
        weight_path = "/data/share/models/quant_model/edq_model_Llama-3-8B_v2/q_model.pt"
        # weight_path = "/data/share/models/quant_model/edq_model_Llama-3-8B/q_model.pt"

    # quant_llm = False

    model, tokenizer = load_from_pretrained(
        model_path=model_path,
        weight_path=weight_path,
        model_type=model_type,
        seed=42,
        setup=True,
        device='cuda',
        trust_remote_code=False,
        local_files_only=True,
        quant_llm=quant_llm,
    )

    for _ in tqdm.tqdm(range(20), desc="Preheating Model"):
        test_input = generate_random_token_sequence(tokenizer, device='cuda', seq_length=256)
        model.generate(**test_input, max_new_tokens=10, pad_token_id = tokenizer.pad_token_id)    
    for _ in tqdm.tqdm(range(20), desc="Preheating Model"):
        test_input = generate_random_token_sequence(tokenizer, device='cuda', seq_length=32)
        model.generate(**test_input, max_new_tokens=10, pad_token_id = tokenizer.pad_token_id)

    ttft = []
    tpot = []
    inputs_len = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 1024*6, 1024*8]
    for i in inputs_len:
        inputs = generate_random_token_sequence(tokenizer, 'cuda', i)
        result = eval_perf(model, tokenizer, inputs, max_new_tokens=20)
        ttft.append(result['ttft'] * 1000)
        tpot.append(result['tpot'] * 1000)
        print(f'input len: {i}')
        print(f'ttft     : {result["ttft"] * 1000:.2f} ms')
        print(f'tpot     : {result["tpot"] * 1000:.2f} ms')
        print('-'*20)
    