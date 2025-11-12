import torch
import edq_cuda_accel
import awq_backend
import torch.cuda.nvtx as nvtx
from utils.color_print import *
from utils.perf_eval import timer
from edq.quantization import (quantize_tensor_int8, pack_int4_data, dequantize_tensor_int4)
from edq.mix_prec_tensor_pool import TensorPool, MixPrecTensorPool

sub_stream = torch.cuda.Stream(device='cuda:0', priority=0)
# sub_stream_ptr = sub_stream.cuda_stream
cur_stream = torch.cuda.current_stream()
a = torch.tensor(1, device='cuda')

@torch.compile()
@torch.no_grad()
def edq_quantize_tensor_int4(x, group_size = -1):
    x_shape = x.shape    
    if group_size < 0:
        x = x.reshape(-1, x_shape[-1])
    else:
        assert x_shape[-1] % group_size == 0
        x = x.reshape(-1, group_size)
    x = x.to(torch.float32)
    max_int = 15    # 2**4 - 1``
    max_val = x.amax(dim=1, keepdim=True)
    min_val = x.amin(dim=1, keepdim=True)
    scale_f   = ((max_val-min_val).clamp(min=1e-7) / max_int)
    zero_pt_f = (-(min_val / scale_f))
    # NOTE: Use floor instead of Round to prevent overflow during quantization.
    scale_i   = torch.floor(scale_f).to(torch.int8)
    zero_pt_i = torch.round(zero_pt_f).to(torch.int8)
    x.div_(scale_f).add_(zero_pt_f).round_()
    assert torch.isnan(x).sum() == 0, print(scale_f.amin())
    x.clamp_(0, max_int)
    return x.reshape(x_shape).to(torch.int8), {
        'scale_f' : scale_f.to(torch.float16), 'scale_i' : scale_i,
        'zero_pt_f' : zero_pt_f.to(torch.float16), 'zero_pt_i' : zero_pt_i, 
        'group_size':group_size}

class DynamicLinearManager():
    @torch.no_grad
    def __init__(self):
        self.linear_pool = []   # 存储每个线性层的索引
        self.event_pool = []    # 事件池用于等待权重转换
        self.size = 0
        # self.stream = torch.cuda.Stream(device='cuda:0', priority=0)
        self.stream = torch.cuda.Stream()
        self.stream_ptr = self.stream.cuda_stream

    @torch.no_grad
    def append_linear(self, d_linear:"DynamicLinear") -> int:
        self.linear_pool.append(d_linear)
        self.event_pool.append(d_linear.event)
        self.size += 1
        return self.size - 1
    
    @torch.no_grad
    def get_linear(self, linear_idx):
        return self.linear_pool[linear_idx]
        
    @torch.no_grad
    def evict_weight(self, linear_idx):
        self.linear_pool[linear_idx]._evict_weight()
        
    @torch.no_grad
    def trans_weight(self, linear_idx):
        self.linear_pool[linear_idx]._dynamic_trans_weight()
        
    @torch.no_grad
    def get_event(self, linear_idx):
        return self.event_pool[linear_idx]
        
    @torch.no_grad
    def record_event(self, linear_idx):
        self.event_pool[linear_idx].record(self.stream)

global_DLM = DynamicLinearManager()

shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)]
dtype_list  = [torch.int8, torch.float16]
global_TP  = MixPrecTensorPool(shape_list, dtype_list)
        
class DynamicLinear(torch.nn.Module):
    """
        Weight   FP16 -> INT8 -> INT4
        Dynamically select Wq and Aq at runtime.
    """
    @torch.no_grad
    def __init__(self, in_features, out_features, bias, device, group_size):
        super().__init__()
        self.enable_accel = False
        self.return_dtype = torch.float16
        self.in_features  = in_features
        self.out_features = out_features
        self.group_size   = group_size
        i8_buffer_kwargs  = {"device" : device, "dtype": torch.int8}
        fp_buffer_kwargs  = {"device" : device, "dtype": torch.float16}
        self.register_buffer(
            "qweight",   torch.empty((out_features, in_features // 2), **i8_buffer_kwargs))
        self.register_buffer(   # Used for int8 -> fp16.
            "scale_1",    torch.empty((out_features), **fp_buffer_kwargs))
        self.register_buffer(
            "scale_2_f",        # Used for int4 -> int8 in dynamic transformation.
            torch.empty((out_features * (in_features//group_size)), **fp_buffer_kwargs))
        self.register_buffer(
            "zero_2_f",         # Used for int4 -> int8 in dynamic transformation.
            torch.empty((out_features * (in_features//group_size)), **fp_buffer_kwargs))
        self.register_buffer(
            "scale_2_i",        # Used for int4 -> int8 in W4A8 kernel.
            torch.empty((out_features * (in_features//group_size)), **i8_buffer_kwargs))
        self.register_buffer(
            "zero_2_i",         # Used for int4 -> int8 in W4A8 kernel.
            torch.empty((out_features * (in_features//group_size)), **i8_buffer_kwargs))
        if bias:
            self.register_buffer(
                "bias", torch.empty((out_features), **fp_buffer_kwargs))
        else:
            self.bias = None
        self.weight = None
        self.shape  = (out_features, in_features)
        self.stream = global_DLM.stream
        self.stream_ptr = global_DLM.stream_ptr
        self.event     = torch.cuda.Event(enable_timing=True)
        self.grid_size = max(256, self.out_features // 8)
        # self.grid_size = 16
        # if (self.shape == (512, 3584)):
        #     self.grid_size = 512
        # elif (self.shape == (3584, 3584)):
        #     self.grid_size = 3584
        # elif (self.shape == (18944, 3584)):
        #     self.grid_size = 18944
        # elif (self.shape == (3584, 18944)):
        #     self.grid_size = 3584
        # else: assert 0
        # self.grid_size = self.out_features
        self.linear_idx   = global_DLM.append_linear(self)
    
    @classmethod
    def from_module(cls, linear, device, return_dtype=torch.float16, name = '',init_only=False):
        q_linear = cls(
            linear.in_features, linear.out_features, 
            linear.bias is not None, device, group_size = 128)
        q_linear.return_dtype = return_dtype
        q_linear.name = name
        # q_linear.ori_weight = linear.weight.data.clone().T
        if init_only: 
            return q_linear
        
        # Step1: Quantize weight from fp16 to int8.
        if q_linear.enable_accel:
            weight_i8 = torch.empty((q_linear.out_features, q_linear.in_features),
                                    device = device, dtype = torch.int8)
            edq_cuda_accel.quant_fp16_to_int8(
                weight_i8, linear.weight.data.to(torch.float16), q_linear.scale_1)
        else:
            weight_i8, q_config1 = quantize_tensor_int8(linear.weight.data, max_int=120)
            q_linear.scale_1 = q_config1['scale'].view(-1).contiguous()
        # Step2: Quantize weight from int8 to uint4.
        weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=q_linear.group_size)
        q_linear.scale_2_f  = q_config2['scale_f'].view(-1).contiguous()
        q_linear.scale_2_i  = q_config2['scale_i'].view(-1).contiguous()
        q_linear.zero_2_f   = q_config2['zero_pt_f'].view(-1).contiguous()
        q_linear.zero_2_i   = q_config2['zero_pt_i'].view(-1).contiguous()

        q_linear.weight_i8 = dequantize_tensor_int4( weight_i4, q_linear.scale_2_i, 
                zero_pt=q_linear.zero_2_i, group_size=q_linear.group_size)
        assert (q_linear.weight_i8.amax() <= 127), print((q_linear.weight_i8.amax()))
        assert (q_linear.weight_i8.amin() >= -128), print((q_linear.weight_i8.amin()))
        q_linear.weight_i8 = q_linear.weight_i8.to(torch.int8)

        q_linear.scale_f = q_linear.scale_2_f.view(q_linear.scale_1.shape[0], -1)*q_linear.scale_1.view(-1,1)

        # Step3: Pack 2 uint4 data into an int8 space.
        q_linear.qweight = pack_int4_data(weight_i4)
        
        if linear.bias is not None:
            q_linear.bias = linear.bias.clone().half()
        return q_linear
    
    @torch.no_grad
    def _dynamic_trans_weight(self):
        # nvtx.range_push(f"DynamicLinear_{self.linear_idx}:_dynamic_trans_weight")
        
        # with torch.cuda.stream(self.stream):
            # weight_i8 = dequantize_tensor_int4(
            #     unpack_int4_data(self.qweight), self.scale_2_f, 
            #     zero_pt=self.zero_2_f, group_size=self.group_size)
            # self.weight = dequantize_tensor_int8(weight_i8, self.scale_1)
            # self.weight = self.ori_weight

        ###########################################################################
        #           DEQUANT WEIGHT FROM PACKED UINT4 TO FLOAT16 FORMAT            #
        # ###########################################################################
        # if self.weight is None:
        #     self.weight = global_TP.get_tensor(self.shape, torch.float16)
        # self.scale_f = self.scale_2_f.view(self.scale_1.shape[0], -1)*self.scale_1.view(-1,1)
        # edq_cuda_accel.dequant_int4_to_fp16(self.weight, self.qweight, 
        #                 self.scale_f, self.zero_2_f, self.group_size)
        # edq_cuda_accel.dequant_int4_to_fp16_stream(self.weight, self.qweight, 
        #                 self.scale_f, self.zero_2_f, self.group_size, self.stream_ptr)
        # edq_cuda_accel.dequant_int4_to_fp16_new(self.weight, self.qweight, 
        #                 self.scale_f, self.zero_2_f, self.group_size, self.stream_ptr,
        #                 self.event_ptr, 28)

        ###########################################################################
        #           DEQUANT WEIGHT FROM PACKED UINT4 TO INT8 FORMAT               #
        ###########################################################################
        if self.weight is None:
            self.weight = global_TP.get_tensor(self.shape, torch.int8)
        # with torch.cuda.stream(self.stream):
            # edq_cuda_accel.dequant_int4_to_int8(self.weight, self.qweight, 
            #                     self.scale_2_i, self.zero_2_i, self.group_size)
        # edq_cuda_accel.dequant_int4_to_int8_new_2(self.weight, self.qweight, 
        #                 self.scale_2_i, self.zero_2_i, self.group_size, self.grid_size)
        # torch.cuda.set_stream(self.stream)
        # FIXME: get cudaStream in cuda kernel?
        with torch.cuda.stream(sub_stream):
            edq_cuda_accel.dequant_int4_to_int8_new(self.weight, self.qweight, 
                            self.scale_2_i, self.zero_2_i, self.group_size, self.grid_size)
            # edq_cuda_accel.dequant_int4_to_int8_new_2(self.weight, self.qweight, 
            #                 self.scale_2_i, self.zero_2_i, self.group_size, self.grid_size)
        # torch.cuda.set_stream(cur_stream)
        # print(self.event.cuda_event)
        # with torch.cuda.stream(self.stream):
            global_DLM.event_pool[self.linear_idx].record()
        # torch.cuda.default_stream().wait_stream(global_DLM.stream)
        # global_DLM.stream.synchronize()
        # cur_stream.wait_event(self.event)
        # self.stream.synchronize()
            
        # print(self.event.cuda_event)
        # self.stream.synchronize()
        # global_DLM.record_event(self.linear_idx)
        # nvtx.range_pop()

    @torch.no_grad
    def _evict_weight(self):
        if self.weight is not None:
            global_TP.free_tensor(self.shape, self.weight.dtype)
            self.weight = None
    
    @torch.no_grad
    def forward(self, input:torch.Tensor) -> torch.Tensor:
        # nvtx.range_push(f"{self.name}:_forward")
        if self.weight is None:
            self._dynamic_trans_weight()
        global_DLM.evict_weight((self.linear_idx - 1 + global_DLM.size) % global_DLM.size)
        # global_DLM.stream.synchronize()
        out = self._forward(input)
        global_DLM.trans_weight((self.linear_idx + 3) % global_DLM.size)
        # nvtx.range_pop()
        return out

    @torch.no_grad
    # @torch.compile
    def _forward(self, input:torch.Tensor) -> torch.Tensor:
        # global_DLM.stream.synchronize()
        intput_shape = input.shape
        input = input.view(-1, intput_shape[-1])

        int_x = torch.empty_like(input, device='cuda', dtype=torch.int8)
        out = torch.empty((input.shape[0], self.shape[0]), 
                            device='cuda', dtype=torch.float16)
        scale_x = torch.empty(input.shape[0], device='cuda', dtype=torch.float16)
        # global_DLM.stream.synchronize()
        # self.event.synchronize()
        edq_cuda_accel.quant_fp16_to_int8(int_x, input, scale_x)

        # Wait for the previous prepare for weight to be done

        # TODO: have bug here, only stream synchronize can correctly work.
        # cur_stream.wait_event(self.event)
        # torch.cuda.current_stream().wait_event(self.event)
        # self.event.synchronize()
        # self.stream.synchronize()
        # torch.cuda.synchronize()
        # self.stream.synchronize()
        # torch.cuda.default_stream().wait_stream(global_DLM.stream)
        # global_DLM.event_pool[self.linear_idx].synchronize()
        # out = input @ self.weight.T
        sub_stream.synchronize()
        # b  = torch.empty_like(input, device='cuda', dtype=torch.int8)
        # global_DLM.event_pool[self.linear_idx].synchronize()
        # diff = (self.weight_i8[-1,-1] - self.weight[-1,-1])
        # assert diff == 0
        awq_backend.w8a8_gemm_forward_cuda(int_x, self.weight, self.scale_1, scale_x, out)
        if self.bias is not None:
            out += self.bias
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    
    @torch.no_grad
    def forward_ori(self, input:torch.Tensor) -> torch.Tensor:
        intput_shape = input.shape
        input = input.reshape(-1, intput_shape[-1]).to(torch.float16)
        # Wait for the previous prepare for weight to be done
        out = input @ self.weight
        if self.bias is not None:
            out += self.bias
        return out.reshape(intput_shape[0], intput_shape[1], -1)