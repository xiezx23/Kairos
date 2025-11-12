#!/bin/bash

source ./script/global_config.config

echo "------------------------------------------------------"
echo "开始测试模型能力"
echo "测试数据集:      CMMLU"
echo "测试模型: 1.Qwen2.5-7B-Instruct   2.Meta-Llama3-8B"
echo "------------------------------------------------------"
echo "开始测试 1.1.全精度 Qwen2.5-7B-Instruct"
python -m eval.acc_cmmlu --model_type qwen --result_dir ./results/qwen/full/cmmlu

echo "开始测试 1.2.量化后 Qwen2.5-7B-Instruct"
python -m eval.acc_cmmlu --model_type qwen --weight_path $model_path_quant_qwen2 --quant_llm True --result_dir ./results/qwen/quant/cmmlu

echo "Qwen2.5-7B-Instruct 测试完成"
echo "------------------------------------------------------"
echo "开始测试 2.1.全精度 Meta-Llama3-8B"
python -m eval.acc_cmmlu --model_type llama  --result_dir ./results/llama/full/cmmlu

echo "开始测试 2.2.量化后 Meta-Llama3-8B"
python -m eval.acc_cmmlu --model_type llama --weight_path $model_path_quant_llama --quant_llm True --result_dir ./results/llama/quant/cmmlu

echo " Meta-Llama3-8B测试完成"
echo "------------------------------------------------------"

echo "汇总结果，生成报告: "
python test/compare_acc_cmmlu.py --model_type qwen
python test/compare_acc_cmmlu.py --model_type llama