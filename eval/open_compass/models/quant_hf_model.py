import torch
import tqdm
import os

from opencompass.models import HuggingFaceCausalLM
from opencompass.models import HuggingFaceBaseModel
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.registry import MODELS
from utils.color_print import colored_print
from edq.quant_model import quantize_model
from utils.util import load_json

try:
    import edq_cuda_accel
    import awq_backend
    enable_accel = True
except ImportError:
    enable_accel = False


device_name = torch.cuda.get_device_name(0)

@MODELS.register_module()
class QuantHuggingFaceCausalLM(HuggingFaceCausalLM):
    def __init__(self,
                 quant_llm: bool = True,
                 quant_weight_path: str = "",
                 verbose: int = 1,
                 *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.quant_llm = quant_llm
        self.quant_weight_path = quant_weight_path
        self.verbose = verbose

        assert self.model is not None
        if self.quant_llm:
            if self.verbose > 0:
                colored_print(f"Loading quantized model from {self.quant_weight_path}", "note")
                colored_print(f"Enable acceleration with edq_cuda_accel: {enable_accel}", "note")

            self._load_quantized_weight()
                

    def _load_quantized_weight(self):
        ""
        # 替换模型中的线性层为量化后的线性层
        quantize_model(self.model, init_only=True)
        self.model.tie_weights()

        # 加载量化后的权重
        self.model.load_state_dict(torch.load(self.quant_weight_path))

        pbar = tqdm.tqdm(range(1))
        pbar.set_description("Loading checkpoint")
        for i in pbar:
            if self.quant_weight_path.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load

                self.model.load_state_dict(safe_load(self.quant_weight_path))
            else:
                self.model.load_state_dict(torch.load(self.quant_weight_path))


@MODELS.register_module()
class QuantHuggingFacewithChatTemplate(HuggingFacewithChatTemplate):
    def __init__(self,
                 quant_llm: bool = True,
                 quant_weight_path: str = "",
                 verbose: int = 1,
                 *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.quant_llm = quant_llm
        self.quant_weight_path = quant_weight_path
        self.verbose = verbose

        assert self.model is not None
        if self.quant_llm:
            if self.verbose > 0:
                colored_print(f"Loading quantized model from {self.quant_weight_path}", "note")
                colored_print(f"Enable acceleration with edq_cuda_accel: {enable_accel}", "note")

            self._load_quantized_weight()

    def _load_quantized_weight(self):
        ""
        _QUANT_CONFIG = 'quant_config.json'
        _MODEL_WEIGHT = 'q_model.pt'
        
        config_path = os.path.join(self.quant_weight_path, _QUANT_CONFIG)
        model_weight_path = os.path.join(self.quant_weight_path, _MODEL_WEIGHT)
        
        # 替换模型中的线性层为量化后的线性层
        quant_config = load_json(config_path)
        quantize_model(self.model, init_only=True, _quant_config=quant_config)
        self.model.tie_weights()
        # 加载量化后的权重
        self.model.load_state_dict(torch.load(model_weight_path))


@MODELS.register_module()
class QuantHuggingFaceBaseModel(HuggingFaceBaseModel):
    def __init__(self,
                 quant_llm: bool = True,
                 quant_weight_path: str = "",
                 verbose: int = 1,
                 *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.quant_llm = quant_llm
        self.quant_weight_path = quant_weight_path
        self.verbose = verbose
        self.device_name = device_name

        assert self.model is not None
        if self.quant_llm:
            if self.verbose > 0:
                colored_print(f"Loading quantized model from {self.quant_weight_path}", "note")
                colored_print(f"Enable acceleration with edq_cuda_accel: {enable_accel}", "note")

            self._load_quantized_weight()

    def _load_quantized_weight(self):
        ""
        _QUANT_CONFIG = 'quant_config.json'
        _MODEL_WEIGHT = 'q_model.pt'
        
        config_path = os.path.join(self.quant_weight_path, _QUANT_CONFIG)
        model_weight_path = os.path.join(self.quant_weight_path, _MODEL_WEIGHT)
        
        # 替换模型中的线性层为量化后的线性层
        quant_config = load_json(config_path)
        quantize_model(self.model, init_only=True, _quant_config=quant_config)
        self.model.tie_weights()
        # 加载量化后的权重
        self.model.load_state_dict(torch.load(model_weight_path))
