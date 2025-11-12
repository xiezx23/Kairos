mkdir log
mkdir log/batch_decode
mkdir log/prefill

test_case=batch_decode
# 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear'
quant_type=W8A8Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=W4A16Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=W4A8Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=DynamicLinear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
python -m edq.main --test_case $test_case --full True > log/$test_case/FullLinear

test_case=prefill
# 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear'
quant_type=W8A8Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=W4A16Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=W4A8Linear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
quant_type=DynamicLinear
python -m edq.main --test_case $test_case --quant_type $quant_type > log/$test_case/$quant_type
python -m edq.main --test_case $test_case --full True > log/$test_case/FullLinear