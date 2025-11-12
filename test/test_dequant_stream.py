import torch
import edq_cuda_accel
from edq.dynamic_linear import edq_quantize_tensor_int4
from edq.quantization import (quantize_tensor_int8, pack_int4_data, dequantize_tensor_int4)
from edq.mix_prec_tensor_pool import TensorPool, MixPrecTensorPool
from utils.perf_eval import timer

shape_list = [(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)]
dtype_list  = [torch.int8, torch.float16]
global_TP  = MixPrecTensorPool(shape_list, dtype_list)

enable_accel = True
in_features = 3584
out_features = 3584
device = 'cuda'
group_size = 128

weight = torch.randn((out_features, in_features), dtype=torch.float16, device=device)

a = torch.randn((1, 3584), dtype=torch.float16, device=device)
b = torch.randn((3584, 18944), dtype=torch.float16, device=device)


# Step1: Quantize weight from fp16 to int8.
if enable_accel:
    scale_1 = torch.empty(out_features, device = device, dtype = torch.float16)
    weight_i8 = torch.empty((out_features, in_features),
                            device = device, dtype = torch.int8)
    edq_cuda_accel.quant_fp16_to_int8(
        weight_i8, weight, scale_1)
else:
    weight_i8, q_config1 = quantize_tensor_int8(weight, max_int=120)
    scale_1 = q_config1['scale'].view(-1).contiguous()
# Step2: Quantize weight from int8 to uint4.
weight_i4, q_config2 = edq_quantize_tensor_int4(weight_i8, group_size=group_size)
scale_2_f  = q_config2['scale_f'].view(-1).contiguous()
scale_2_i  = q_config2['scale_i'].view(-1).contiguous()
zero_2_f   = q_config2['zero_pt_f'].view(-1).contiguous()
zero_2_i   = q_config2['zero_pt_i'].view(-1).contiguous()

scale_f = scale_2_f.view(scale_1.shape[0], -1)*scale_1.view(-1,1)

# Step3: Pack 2 uint4 data into an int8 space.
qweight = pack_int4_data(weight_i4)

sub_stream = torch.cuda.Stream(device='cuda:0', priority=0)
sub_stream_ptr = sub_stream.cuda_stream
event = torch.cuda.Event(enable_timing=True)

edq_cuda_accel.dequant_int4_to_fp16(weight, qweight, scale_f, zero_2_f, group_size)

weight_list = []
for _ in range(16):
    weight_list.append(None)


event_list = []
for _ in range(16):
    event = torch.cuda.Event(enable_timing=True)
    event_list.append(event)


weight_i8 = torch.empty((out_features, in_features), dtype=torch.int8, device=device)
edq_cuda_accel.dequant_int4_to_int8_new(weight_i8, qweight, 
                        scale_2_i, zero_2_i, group_size, 16)

@torch.no_grad
def trans_weight(idx):
    grid_size = max(256, out_features // 16)
    grid_size = 1
    weight_list[idx] = global_TP.get_tensor((out_features, in_features), torch.int8)
    with torch.cuda.stream(sub_stream):
        edq_cuda_accel.dequant_int4_to_int8_new(weight_list[idx], qweight, 
                        scale_2_i, zero_2_i, group_size, grid_size)
        # edq_cuda_accel.dequant_int4_to_fp16_new(weight_list[idx], qweight, scale_f, zero_2_f, group_size, 16)
        event_list[idx].record()
    # c = a @ b

@torch.no_grad
def forward(idx):
    if weight_list[idx] is None:
        trans_weight(idx)
    global_TP.free_tensor((out_features, in_features), torch.int8)
    diff = _forward(idx)
    trans_weight(idx+4)
    return diff.sum()

@torch.no_grad
def _forward(idx):
    event_list[idx].synchronize()
    # sub_stream.synchronize()
    diff = weight_i8-weight_list[idx]
    return diff


with torch.no_grad():
    res = 0
    with timer():
        res += forward(0)
        res += forward(1)
        res += forward(2)

        res += forward(3)
        res += forward(4)
        res += forward(5)
        res += forward(6)
    print(res)