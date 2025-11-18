echo Test LLaMa-3-8B
echo None | python -m edq.main --model_type llama --full 1 > Full_llama.log
echo W8A8Linear | python -m edq.main --model_type llama > W8A8Linear_llama.log
echo W4A16Linear | python -m edq.main --model_type llama > W4A16Linear_llama.log
echo W4A8Linear | python -m edq.main --model_type llama > W4A8Linear_llama.log
echo DynamicLinear | python -m edq.main --model_type llama > DynamicLinear_llama.log


echo Test Qwen2.5-Instruct-7B
echo None | python -m edq.main --model_type qwen --full 1 > Full_qwen.log
echo W8A8Linear | python -m edq.main --model_type qwen > W8A8Linear_qwen.log
echo W4A16Linear | python -m edq.main --model_type qwen > W4A16Linear_qwen.log
echo W4A8Linear | python -m edq.main --model_type qwen > W4A8Linear_qwen.log
echo DynamicLinear | python -m edq.main --model_type qwen > DynamicLinear_qwen.log
