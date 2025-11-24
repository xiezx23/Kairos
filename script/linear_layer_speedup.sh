mkdir log/
mkdir log/linear/

echo Test LLaMa-3-8B
echo DynamicLinear | python -m eval.eval_llm_linear --model_type llama > log/linear/llama.log

echo Test Qwen2.5-Instruct-7B
echo DynamicLinear | python -m eval.eval_llm_linear --model_type qwen > log/linear/qwen.log
