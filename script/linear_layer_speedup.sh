echo Test LLaMa-3-8B
echo DynamicLinear | python -m eval.eval_llm_linear --model_type llama > log/linearLayer/DynamicLinear_llama.log

echo Test Qwen2.5-Instruct-7B
echo DynamicLinear | python -m eval.eval_llm_linear --model_type qwen > log/linearLayer/DynamicLinear_qwen.log
