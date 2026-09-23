import torch
import tqdm
import gc
from kairos.quantization import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
if hasattr(torch, "cuda") and torch.cuda.is_available():
    device  = 'cuda'
    try:
        import awq_backend
    except ImportError:
        awq_backend = None
else:
    device = 'cpu'
    
class W4A16Linear_AWQ(torch.nn.Module):
    """
        activation  FP16 \\
        weight      FP16 -> quantize -> INT4\\
        O = A @ deq(W) (FP16 GEMM) -> FP16\\
    """
    def __init__(self, in_features, out_features, bias, device, group_size = -1):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        if group_size == -1:
            group_size = out_features
        self.group_size   = group_size
        buffer_kwargs = {"device" : device, "dtype": torch.float16}
        if self.group_size < 0:
            self.group_size = in_features
        h = calculate_zeros_width(in_features, self.group_size) * 8
        self.register_buffer(
            "qweight",  torch.empty(
                (out_features//4, in_features),
                dtype=torch.int16,
                device=device))
        self.register_buffer(
            "scales",    torch.empty(
                (h, out_features),**buffer_kwargs))
        self.register_buffer(
            "scaled_zeros",  torch.empty(
                (h, out_features),**buffer_kwargs))
        if bias:
            self.register_buffer(
                "bias", torch.empty((out_features), **buffer_kwargs))
        else:
            self.bias = None
    
    @classmethod
    def from_module(cls, linear, device, return_dtype=torch.float16, init_only=False, name=''):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device, group_size=128)
        # assert not(init_only and not q_linear.enable_accel)
        q_linear.return_dtype = return_dtype
        if init_only: 
            return q_linear
        q_linear.qweight, q_linear.scales, q_linear.scaled_zeros = \
            quant_weight_awq(linear.weight.data.clone(), q_linear.group_size)
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half().contiguous()
        return q_linear

    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        # input = input.to(torch.float16)
        intput_shape = input.shape
        input = input.reshape(-1, intput_shape[-1])
        batch_token_len = input.shape[0]
        # batch_token_len = input.numel() // input.shape[-1]
        if batch_token_len < 8:
            out = awq_backend.gemv_forward_cuda_new(
                input, self.qweight, self.scales, self.scaled_zeros,
                batch_token_len, self.out_features, self.in_features, self.group_size)
        else:
            out = awq_backend.gemm_forward_cuda_new(
                input, self.qweight, self.scales, self.scaled_zeros)
        if self.bias is not None:
            out.add_(self.bias)
        # if len(intput_shape) == 2: return out #DEBUG
        return out.reshape(intput_shape[0], intput_shape[1], -1)
        # return out.to(self.return_dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

# @torch.no_grad()
# def quantize_model(model):
#     decoderLayers = model.model.layers  # Qwen2___5-7B 是28层堆叠的Qwen2DecoderLayer
#     for layer_idx in tqdm.tqdm(range(len(decoderLayers)), desc="Quantize Model"):
#         layer = decoderLayers[layer_idx]
#         # 获取模型的线性层，建立名称->线性层的字典
#         def get_name_linears(module):  
#             return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
#         name2linears = get_name_linears(layer)
#         for name, module in name2linears.items():
#             # if 'attn' in name:
#             #     continue
#             return_dtype = torch.float16
#             q_linear = W4A16Linear.from_module(module, model.device, return_dtype)
#             module.cpu()
#             def set_op_by_name(layer, name, new_module):
#                 levels = name.split(".")
#                 if len(levels) > 1:
#                     mod_ = layer
#                     for l_idx in range(len(levels) - 1):
#                         if levels[l_idx].isdigit():
#                             mod_ = mod_[int(levels[l_idx])]
#                         else:
#                             mod_ = getattr(mod_, levels[l_idx])
#                     setattr(mod_, levels[-1], new_module)
#                 else:
#                     setattr(layer, name, new_module)
#             set_op_by_name(layer, name, q_linear)
#         # break
#     torch.cuda.empty_cache()
#     gc.collect()
