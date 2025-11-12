# Evaluation Result

## Device

1 * NVIDIA A100 80GB PCIe



## Models

1. Qwen2.5-7B-Instruct
2. Meta-Llama8B



## Quantization and Dump

Qwen2.5-7B-Instruct：

```
python -m edq.main --dump_quant_model /data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct --model_type qwen
```

Meta-Llama8B：

```
python -m edq.main --dump_quant_model /data/share/models/quant_model/edq_model_Llama3-8B --model_type llama
```



量化效果：体积压缩率 = (全精度模型体积 - 量化后模型体积) / 全精度模型体积

|            | Qwen2.5-7B-Instruct | Meta-Llama3-8B |
| ---------- | ------------------- | -------------- |
| 全精度模型 | 14.23 GB            | 14.96 GB       |
| 量化后模型 | 5.70 GB             | 6.00 GB        |
| 体积压缩率 | 59.94%              | 59.89%         |



## Load Quantized Model

Qwen2.5-7B-Instruct：

```
python -m edq.main --load_quant_model /data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct --model_type qwen
```

Meta-Llama8B：

```
python -m edq.main --load_quant_model /data/share/models/quant_model/edq_model_Llama3-8B --model_type llama
```

Ps: You can modify the prompt in edq.main to test other chat with models, or we suggest you to use edq_chatbot.



## Wiki2

### Qwen2.5-7B-Instruct Test Result

全精度：

```
python -m edq.main --full 1 --model_type qwen --eval_ppl True
```

```
Perplexity: 7.56548547744751
```

量化后：（Perplexity <= 7.943759751319885）

```
python -m edq.main --load_quant_model /data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct --model_type qwen --eval_ppl True
```

```
Perplexity: 7.646196365356445
```



### Llama3-8B Test Result

全精度：

```
python -m edq.main --full 1 --model_type llama --eval_ppl True
```

```
Perplexity: 6.382576942443848
```

量化后：（Perplexity <= 6.701705789566041）

```
python -m edq.main --load_quant_model /data/share/models/quant_model/edq_model_Llama3-8B --model_type llama --eval_ppl True
```

```
Perplexity: 6.614130973815918
```



## CMMLU

### Qwen2.5-7B-Instruct Test Result

全精度：

```
python -m eval.acc_cmmlu --model_type qwen --result_dir ./results/cmmlu/qwen/full/
```

```
STEM                                     70.87
Humanities                               79.16
Social Science                           77.20
Other                                    80.12
China specific                           74.78
Overall                                  76.63
```

量化后：>= 72.7985

```
python -m eval.acc_cmmlu --model_type qwen --weight_path /data/share/models/quant_model/edq_model_Qwen2.5-7B-Instruct --quant_llm True --result_dir ./results/cmmlu/qwen/quant/
```

```
STEM                                     69.66
Humanities                               79.72
Social Science                           76.43
Other                                    79.22
China specific                           74.00
Overall                                  75.98
```



### Llama3-8B Test Result

全精度：

```
python -m eval.acc_cmmlu --model_type llama --result_dir ./results/cmmlu/llama/full/
```

```
STEM                                     42.04
Humanities                               51.39
Social Science                           49.78
Other                                    50.15
China specific                           44.17
Overall                        			 48.21
```

量化后：>= 45.80

```
python -m eval.acc_cmmlu --model_type llama --weight_path /data/share/models/quant_model/edq_model_Llama3-8B --quant_llm True --result_dir ./results/cmmlu/llama/quant/
```

```
STEM                                     40.48
Humanities                               48.93
Social Science                           47.55
Other                                    49.90
China specific                           41.75
Overall                        		 	 46.55
```



## MMLU-Pro

安装 opencompass：

```
pip install -U opencompass
```

### Qwen2.5-7B-Instruct Test Result

全精度：

```
opencompass eval/open_compass/eval_scripts/template/eval_full_instruct.py
```

```

```

量化后：(Overall >= )

```
opencompass eval/open_compass/eval_scripts/template/eval_quant_instruct.py
```

```
mmlu_pro_math               70.47
mmlu_pro_physics            55.97
mmlu_pro_chemistry          53.71
mmlu_pro_law                28.25
mmlu_pro_engineering        42.72
mmlu_pro_other              51.19
mmlu_pro_economics          65.40
mmlu_pro_health             56.60
mmlu_pro_psychology         62.41
mmlu_pro_business           63.12
mmlu_pro_biology            73.50
mmlu_pro_philosophy         42.28
mmlu_pro_computer_science   58.05
mmlu_pro_history            44.09

Overall						54.84
```



### Llama3-8B Test Result

全精度：

```
opencompass eval/open_compass/eval_scripts/template/eval_full_base.py
```

参考结果：

```
mmlu_pro_math               21.24
mmlu_pro_physics            25.17
mmlu_pro_chemistry          19.43
mmlu_pro_law                13.62
mmlu_pro_engineering        17.13
mmlu_pro_other              21.65
mmlu_pro_economics          24.64
mmlu_pro_health             19.93
mmlu_pro_psychology         41.85
mmlu_pro_business           14.96
mmlu_pro_biology            36.96
mmlu_pro_philosophy         33.47
mmlu_pro_computer_science   23.90
mmlu_pro_history            34.91

Overall						24.92
```

量化后：

```
opencompass eval/open_compass/eval_scripts/template/eval_quant_base.py
```

参考结果：(Overall >= 23.67)

```
mmlu_pro_math               23.17
mmlu_pro_physics            25.33
mmlu_pro_chemistry          22.08
mmlu_pro_law                9.99
mmlu_pro_engineering        17.75
mmlu_pro_other              22.73
mmlu_pro_economics          23.58
mmlu_pro_health             30.93
mmlu_pro_psychology         38.85
mmlu_pro_business           15.21
mmlu_pro_biology            32.36
mmlu_pro_philosophy         30.86
mmlu_pro_computer_science   26.34
mmlu_pro_history            29.13

Overall						24.88
```