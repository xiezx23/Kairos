#!/bin/bash

source ./script/global_config.config

echo "--------------------

----------------------------------"
echo "开始测试模型能力 Wiki2 困惑度测试"
echo "测试模型: 1.Qwen2.5-7B-Instruct   2.Meta-Llama3-8B"
echo "------------------------------------------------------"
echo "开始测试 1.1.全精度 Qwen2.5-7B-Instruct"
python -m eval.ppl_wiki2 --full 1 --model_type qwen
echo "开始测试 1.2.量化后 Qwen2.5-7B-Instruct"
python -m eval.ppl_wiki2 --model_type qwen --weight_path $model_path_quant_qwen2
echo "Qwen2.5-7B-Instruct 测试完成"
echo "------------------------------------------------------"
echo "开始测试 2.1.全精度 Meta-Llama3-8B"
python -m eval.ppl_wiki2 --full 1 --model_type llama
echo "开始测试 2.2.量化后 Meta-Llama3-8B"
python -m eval.ppl_wiki2 --model_type llama  --weight_path $model_path_quant_llama
echo " Meta-Llama3-8B测试完成"
echo "------------------------------------------------------"

echo "汇总结果，生成报告: "
python test/compare_ppl_wiki2.py --model_type qwen
python test/compare_ppl_wiki2.py --model_type llama