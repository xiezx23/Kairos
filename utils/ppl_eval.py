import torch
import tqdm
from datasets import load_dataset
from utils.global_config import dataset_root_path
# from modelscope.msdatasets import MsDataset

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

# # NOTE: no-used now
# class Evaluator:
    # def __init__(self, dataset, tokenizer, device, n_samples=40):
    #     self.dataset = dataset
    #     self.tokenizer = tokenizer
    #     self.device = device
    #     self.dataset = tokenizer(
    #         "\n\n".join(dataset["text"]), return_tensors="pt"
#         ).input_ids.to(device)

#         self.n_samples = n_samples

#     @torch.no_grad()
#     def evaluate(self, model):
#         model.eval()
#         nlls = []
#         n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
#         for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
#             batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
#             with torch.no_grad():
#                 lm_logits = model(batch).logits
#             shift_logits = lm_logits[:, :-1, :].contiguous().float()
#             shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
#             loss_fct = torch.nn.CrossEntropyLoss()
#             loss = loss_fct(
#                 shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
#             )
#             neg_log_likelihood = loss.float() * 2048
#             nlls.append(neg_log_likelihood)

#         return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))

# # NOTE: no-used now
# def evaluate_ppl(model, tokenizer) -> None:
    # dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', trust_remote_code=False, split='test')
    # evaluator = Evaluator(dataset, tokenizer, "cuda")
    # ppl = evaluator.evaluate(model)
    # print(f"Perplexity: {ppl}")