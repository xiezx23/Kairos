from mmengine.config import read_base
###############################################################
# opencompass的配置文件模板
# 其中必须包含datasets和models两个属性，启动时会根据这两个属性进行测试

###############################################################
# dataset配置
# 主要修改部分1

with read_base():
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_few_shot_gen_bfaf90 import mmlu_pro_datasets
    # from opencompass.configs.datasets.mmlu_pro.mmlu_pro_ppl import mmlu_pro_datasets
    # from opencompass.configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets  

# datasets本质上是一个列表，其中每个原生是一个字典（包含数据集下某个科目的数据集的配置，例如：
"""
    cmmlu_datasets.append(
        dict(
            type=CMMLUDataset,
            path='opencompass/cmmlu',
            name=_name,
            abbr=f'cmmlu-{_name}',
            reader_cfg=dict(
                input_columns=['question', 'A', 'B', 'C', 'D'],
                output_column='answer',
                train_split='dev',
                test_split='test'),
            infer_cfg=cmmlu_infer_cfg,
            eval_cfg=cmmlu_eval_cfg,
        ))    
"""
# 可以一次性测试多个数据集
# datasets = [*cmmlu_datasets, *mmlu_pro_datasets]
datasets = [*mmlu_pro_datasets]

# 也可以通过切片的方式测试部分数据集(部分科目)
# datasets = cmmlu_datasets[:2]


###############################################################
# model配置
# models与datasets类型，也是一个列表，每个元素都是一个字典，字典中包含模型的配置
#   - 支持一次性测试多个模型
#   - 其中`type`属性对应模型类的名称，例如`HuggingFacewithChatTemplate`
#   - 其余属性是模型类的构造函数的参数，例如`path`、`tokenizer_path`、`abbr`等
########################
# 此处不用修改
# 注意，这里我们引入了自定义的类（初始化时替换全精度线性层为自定义的量化线性层）
#   其原先未必注册到opencompass的注册器(MODELS = Registry('model', locations=['opencompass.models'])), 需要手动注册
#   参考做法：https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python
#       其中的eval.open_compass.models.quant_hf_model是 QuantHuggingFaceBaseModel 类的所在路径，需要根据实际情况配置
custom_imports = dict(imports=['eval.open_compass.models.quant_hf_model'], allow_failed_imports=False)
from eval.open_compass.models.quant_hf_model import QuantHuggingFaceBaseModel
########################
# 主要修改部分2

device_name = 'local'
if device_name == 'local':
    # In SYSU 501 A100 Server
    model_root_path  = '/data/share/models'
    model_path_qwen2 = model_root_path+'/Qwen/Qwen2___5-7B-Instruct'
    model_path_llama = model_root_path+'/LLM-Research/Meta-Llama-3-8B'
    dataset_root_path = "/data/share/datasets"
elif device_name == 'cloud':
    # In China Mobile Cloud Server
    model_root_path  = '/test/models'
    model_path_qwen2 = model_root_path+'/Qwen2.5-7B-Instruct'
    model_path_llama = model_root_path+'/llama3-8B/llama3-8B'
    dataset_root_path = "/test/datasets"
else:
    print(device_name)
    exit(0)

QUANT_LLM = True
MODEL_PATH = model_path_llama
QUANT_WEIGHT_PATH = model_root_path+"/quant_model/edq_model_Llama3-8B"
ABBR = "Llama3-8B_quant"  # 模型简称，用于结果展示


VERBOSE = 2

MAX_OUT_LEN = 512
BATCH_SIZE = 16  # 根据显存调整, 量化模型的显存占用较低，可以适当调大batch size，如16


# TODO: 调整、补充、优化参数配置
models = [
    dict(
        type=QuantHuggingFaceBaseModel,

        quant_weight_path=QUANT_WEIGHT_PATH,
        quant_llm=QUANT_LLM,
        verbose=VERBOSE,

        path=MODEL_PATH,
        abbr=ABBR,  
        max_out_len=MAX_OUT_LEN,  
        batch_size=BATCH_SIZE, 
        run_cfg=dict(num_gpus=1),  # 运行配置，用于指定资源需求
    ),
]
#############################


