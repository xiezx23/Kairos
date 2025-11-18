echo Test LLaMa-3-8B
echo None | python -m edq.main --eval_ppl 1 --model_type llama --full 1
echo W8A8Linear | python -m edq.main --eval_ppl 1 --model_type llama
echo W4A16Linear | python -m edq.main --eval_ppl 1 --model_type llama
echo W4A8Linear | python -m edq.main --eval_ppl 1 --model_type llama
echo DynamicLinear | python -m edq.main --eval_ppl 1 --model_type llama

echo W8A8Linear | python -m edq.main --eval_ppl 1 --model_type llama --enable_scale 1
echo W4A16Linear | python -m edq.main --eval_ppl 1 --model_type llama --enable_scale 1
echo W4A8Linear | python -m edq.main --eval_ppl 1 --model_type llama --enable_scale 1
echo DynamicLinear | python -m edq.main --eval_ppl 1 --model_type llama --enable_scale 1

echo Test Qwen2.5-Instruct-7B
echo None | python -m edq.main --eval_ppl 1 --model_type qwen --full 1
echo W8A8Linear | python -m edq.main --eval_ppl 1 --model_type qwen
echo W4A16Linear | python -m edq.main --eval_ppl 1 --model_type qwen
echo W4A8Linear | python -m edq.main --eval_ppl 1 --model_type qwen
echo DynamicLinear | python -m edq.main --eval_ppl 1 --model_type qwen

echo W8A8Linear | python -m edq.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo W4A16Linear | python -m edq.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo W4A8Linear | python -m edq.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo DynamicLinear | python -m edq.main --eval_ppl 1 --model_type qwen --enable_scale 1