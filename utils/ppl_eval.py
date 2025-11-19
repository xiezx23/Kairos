import torch
import tqdm
from datasets import load_dataset
from utils.global_config import dataset_root_path

def eval(model, tokenizer):
    dataset_path = dataset_root_path + "/wikitext/"
    dataset_name = 'wikitext-2-v1'  # "wikitext-2-raw-v1"
    print('Eval PPL: ', dataset_name)
    dataset_test = load_dataset(dataset_path, dataset_name, 
                                trust_remote_code = False, split="test")
    dataset_test = tokenizer("\n\n".join(dataset_test["text"]), return_tensors="pt")
    model.seqlen = 1024 * 2
    dataset_test = dataset_test.input_ids.to(model.device)
    nsamples = dataset_test.numel() // model.seqlen
    model = model.eval()
    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = dataset_test[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = dataset_test[
            :, (i * model.seqlen) : ((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    print(f"Perplexity: {ppl.item()}")
    return ppl.item()
