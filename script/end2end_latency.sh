echo Test LLaMa-3-8B
echo None | python -m kairos.main --model_type llama --full 1 > log/end2end/None_llama.log
echo W8A8Linear | python -m kairos.main --model_type llama > log/end2end/W8A8Linear_llama.log
echo W4A16Linear | python -m kairos.main --model_type llama > log/end2end/W4A16Linear_llama.log
echo W4A16Linear_Marlin | python -m kairos.main --model_type llama > log/end2end/W4A16Linear_Marlin_llama.log
echo W4A8Linear | python -m kairos.main --model_type llama > log/end2end/W4A8Linear_llama.log
echo W4A8Linear_QQQ | python -m kairos.main --model_type llama > log/end2end/W4A8Linear_QQQ_llama.log
echo DynamicLinear | python -m kairos.main --model_type llama > log/end2end/DynamicLinear_llama.log


echo Test Qwen2.5-Instruct-7B
echo None | python -m kairos.main --model_type qwen --full 1 > log/end2end/None_qwen.log
echo W8A8Linear | python -m kairos.main --model_type qwen > log/end2end/W8A8Linear_qwen.log
echo W4A16Linear | python -m kairos.main --model_type qwen > log/end2end/W4A16Linear_qwen.log
echo W4A16Linear_Marlin | python -m kairos.main --model_type qwen > log/end2end/W4A16Linear_Marlin_qwen.log
echo W4A8Linear | python -m kairos.main --model_type qwen > log/end2end/W4A8Linear_qwen.log
echo W4A8Linear_QQQ | python -m kairos.main --model_type qwen > log/end2end/W4A8Linear_QQQ_qwen.log
echo DynamicLinear | python -m kairos.main --model_type qwen > log/end2end/DynamicLinear_qwen.log
