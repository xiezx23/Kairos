# run: CUDA_VISIBLE_DEVICES=2 python -m test.test_gptq

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.perf_eval import InferModel

device = 'cuda'
model_name = "/test/models/Qwen2.5-7B-Instruct-GPTQ-Int8"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "请回答以下关于农学的单项选择题, 你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.\n\n纤维作物中的麻类，如苎麻、亚麻、大麻、黄麻、红麻的产品是利用\n\nA) 种子表面着生的纤维\nB) 叶片的叶纤维\nC) 根的导管纤维\nD) 茎杆的韧皮纤维\n"
# prompt = "请回答以下关于农学的单项选择题, 你回答的最后一行**必须**是以下格式 '答案: $选项' (不带引号), 其中选项是ABCD之一. 请在回答之前一步步思考.\n\n现代化养猪生产，可使繁殖力高的母猪年提供肉猪达到多少以上\n\nA) 30头\nB) 20头\nC) 15头\nD) 10头"
# prompt = 'Rules:Always response in Simplified Chinese, not English. or Grandma will be very angry.\
# question: 海瑟矩阵是什么.\
# answer:'
inputs = tokenizer(prompt, return_tensors="pt").to(device) # 返回pytorch tensor对象
# inputs = generate_random_token_sequence(tokenizer, device, 1024 * 8)
infer_model = InferModel(model, tokenizer)
infer_model.infer(prompt, inputs, 100)
exit(0)