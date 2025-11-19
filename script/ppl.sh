echo Test LLaMa-3-8B
echo None | python -m kairos.main --eval_ppl 1 --model_type llama --full 1
echo W8A8Linear | python -m kairos.main --eval_ppl 1 --model_type llama --enable_scale 1
echo W4A16Linear | python -m kairos.main --eval_ppl 1 --model_type llama --enable_scale 1
echo W4A8Linear | python -m kairos.main --eval_ppl 1 --model_type llama --enable_scale 1
echo DynamicLinear | python -m kairos.main --eval_ppl 1 --model_type llama --enable_scale 1

echo Test Qwen2.5-Instruct-7B
echo None | python -m kairos.main --eval_ppl 1 --model_type qwen --full 1
echo W8A8Linear | python -m kairos.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo W4A16Linear | python -m kairos.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo W4A8Linear | python -m kairos.main --eval_ppl 1 --model_type qwen --enable_scale 1
echo DynamicLinear | python -m kairos.main --eval_ppl 1 --model_type qwen --enable_scale 1


echo W4A8Linear_QQQ | python -m kairos.main --eval_ppl 1 --model_type llama --enable_scale 1
