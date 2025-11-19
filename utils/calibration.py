import torch
import os
from datasets import load_dataset as data_loader
from utils.global_config import dataset_root_path

def get_calibration_set(tokenizer, sample_num = 512, per_seq_len = 512, model_type='qwen'):
    try:
        raise FileNotFoundError
        data_path = '../mit-han-lab/pile-val-backup'
        data = data_loader(data_path).shuffle(seed=11)
        samples = []
        for d in data['validation']:
            input_seq = d['text'].strip()
            input_token = tokenizer.encode(input_seq)
            if (len(input_token) <= per_seq_len * 2 and len(input_token) > 0): 
                samples.append(torch.tensor([input_token]))
                if (len(samples) == sample_num):
                    break
        samples = torch.cat(samples, dim=1)             # torch.Size = ([1, totalTokenLen])
        seq_num = samples.shape[-1] // per_seq_len
        split_samples = [samples[:, i*per_seq_len : (i+1)*per_seq_len] for i in range(seq_num)]
        res =  torch.cat(split_samples, dim=0).cuda()   # torch.Size = ([seq_num, per_seq_len])
        attention_mask = torch.ones(1, per_seq_len, device='cuda')
        return {'input_ids':res, 'attention_mask':attention_mask}
    except FileNotFoundError:
        print('use local calibration')
        cali_path = 'calibration_' + model_type + '.pt'
        if os.path.exists(cali_path):
            calibration_inputs = torch.load(cali_path)
            return calibration_inputs
        else:
            assert 0, print("Couldn't find dataset or calibration.pt")

def generate_random_token_sequence(tokenizer, device, seq_length=100):
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_length)).to(device)
    # input_ids = torch.tensor([128 for _ in range(seq_length)]).to(device).view(1,seq_length)
    # Attention Mask(attention_mask)
    attention_mask = torch.ones(1, seq_length).to(device)
    return {'input_ids':input_ids, 'attention_mask':attention_mask}