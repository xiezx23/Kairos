
from utils.util import load_json

model_type = ['llama', 'qwen']
quant_strategy = ['None', 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear']

print('-'*25)
for model in model_type:
    print(f'   {model:5s}       PPL   Loss')
    for strategy in quant_strategy:
        path = f"tmp/{model}_{strategy}_ppl"
        ppl = load_json(path+".json"); ppl2 = ppl
        if strategy != 'None':
            ppl2 = load_json(path+"_scale.json")
        else:
            ref_ppl = ppl
        print(f'{strategy:13s}: {ppl2:4.2f}  {(ppl2-ref_ppl)*100/ref_ppl:4.2f}')
    print('-'*25)