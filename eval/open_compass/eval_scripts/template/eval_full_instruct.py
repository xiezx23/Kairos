from mmengine.config import read_base
###############################################################
# opencompass的配置文件模板
# 其中必须包含datasets和models两个属性，启动时会根据这两个属性进行测试

###############################################################
# dataset配置
# 主要修改部分1
with read_base():
    # 阅读文档理解设置方法：https://opencompass.readthedocs.io/zh-cn/latest/user_guides/datasets.html
    #   数据集配置文件名由以下命名方式构成 {数据集名称}_{评测方式}_{prompt版本号}.py
    #   不带版本号的文件，例如： CLUE_afqmc_gen.py 则指向该评测方式最新的prompt配置文件，通常来说会是精度最高的prompt。
    #   因此此处默认使用 xxx_gen 或者 xxx_ppl 版本的数据
    
    
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets  

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

# 也可以通过切片的方式测试部分数据集
# datasets = cmmlu_datasets[:1]

datasets = [*mmlu_pro_datasets]

###############################################################
# model配置
# models与datasets类型，也是一个列表，每个元素都是一个字典，字典中包含模型的配置
#   - 支持一次性测试多个模型
#   - 其中`type`属性对应模型类的名称，例如`HuggingFacewithChatTemplate`
#   - 其余属性是模型类的构造函数的参数，例如`path`、`tokenizer_path`、`abbr`等
########################
from opencompass.models import HuggingFacewithChatTemplate  
from opencompass.models import TurboMindModelwithChatTemplate 
# inference backend of LMDeploy. It can be either 'turbomind' or 'pytorch'. # If the model is not supported by 'turbomind', it will fallback to # 'pytorch'
BACKEND = "turbomind"  # "pytorch, turbomind"  

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

MODEL_PATH = model_path_qwen2
ABBR = "qwen2_5_7B"  # 模型简称，用于结果展示

USE_LMDEPLOY = False  # 是否使用LMDeploy进行推理加速
MAX_OUT_LEN = 4096
BATCH_SIZE = 8  # 16 比 8 快，但是内存占用会显著增加
# STOP_WORDS=['<|end_of_text|>', '<|eot_id|>']
STOP_WORDS = []

# TODO: 调整、补充、优化相关参数配置

if USE_LMDEPLOY:
    # 使用方法
    # https://opencompass.readthedocs.io/zh-cn/latest/advanced_guides/evaluation_lmdeploy.html
    
    # 参数配置
    # For the detailed engine config and generation config, please refer to
        # https://github.com/InternLM/lmdeploy/blob/main/lmdeploy/messages.py
        
    # engine_config={'tp': 1},
    # gen_config={'do_sample': False},  # FIXME: 这样写会报错
    
    models = [
        dict(
            type=TurboMindModelwithChatTemplate,
            abbr=ABBR,  
            path=MODEL_PATH,
            
            # engine_config=engine_config,  # FIXME: 这样写会报错
            # gen_config=gen_config,
            engine_config=dict(tp=1),
            gen_config=dict(do_sample=False),
            backend=BACKEND,

            max_out_len=MAX_OUT_LEN, 
            batch_size=BATCH_SIZE,  
            run_cfg=dict(num_gpus=1),  # 运行配置，用于指定资源需求
            stop_words = STOP_WORDS,
        ),
    ]
else:
    # NOTE: 测试发现该类型的模型也可以通过LMDeploy进行推理，启动时其会被转换到turbomind format
    models = [
        dict(
            type= HuggingFacewithChatTemplate,

            path=MODEL_PATH,
            tokenizer_path=MODEL_PATH,
            
            abbr=ABBR,  
            max_out_len=MAX_OUT_LEN,  
            batch_size=BATCH_SIZE, 
            run_cfg=dict(num_gpus=1),  # 运行配置，用于指定资源需求
            stop_words = STOP_WORDS,
        ),
    ]



