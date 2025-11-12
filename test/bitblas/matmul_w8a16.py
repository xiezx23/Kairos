import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
import edq_cuda_accel
import awq_backend

torch.manual_seed(seed=11)

# bitblas.set_log_level("Debug")
M = 1
K = 18944
N = 3584
# K = 3584
# N = 512
# group_size = min(16, K)
group_size = -1

matmul_config = bitblas.MatmulConfig(
    M=(1),  # M dimension
    N=N,    # N dimension
    K=K,    # K dimension
    A_dtype="float16",      # activation A dtype
    W_dtype="int8",        # weight W dtype
    accum_dtype="float32",  # accumulation dtype
    out_dtype="float16",    # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode='original',  # setting for how to calculating zeros
)

# Create input matrices
n = 1
input_tensor  = torch.randn((M, K), dtype=torch.float16).cuda()
# input_tensor   = torch.eye((K), dtype=torch.float16).cuda()
# input_tensor   = torch.randint(-64, 64 + 1, (M, K), dtype=torch.int8).cuda()
weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

weight_int8_edq, config_int8_edq = quantize_tensor_int8(weight_tensor, group_size=group_size)
weight_tensor = dequantize_tensor_int8(weight_int8_edq, config_int8_edq['scale'], group_size=group_size)

scale_w = config_int8_edq['scale'].squeeze(-1)

input_tensor = input_tensor.to(torch.float16)
weight_tensor = weight_tensor.to(torch.float16)
for _ in range(n):
    ref_result = input_tensor @ weight_tensor.T
with timer(desc='RefFp16'):
    ref_result = input_tensor @ weight_tensor.T
print("RefFp16 Output:", ref_result)

# for _ in range(5):
#     input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
#     scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
#     edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
#     output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
#     awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8_edq, scale_w, scale_x, output)
#     # edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_int8_edq, scale_x, scale_w, output)
# with timer('EDQ W8A8 '):
#     input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
#     scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
#     edq_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
#     output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
#     awq_backend.w8a8_gemm_forward_cuda(input_int8, weight_int8_edq, scale_w, scale_x, output)
    # edq_cuda_accel.w8a8_wsas_gemm_cuda(input_int8, weight_tensor_int8, scale_x, scale_w, output)

# loss = (ref_result-output).abs().amax()
# print('loss:', loss.item())

matmul = bitblas.Matmul(config=matmul_config, target='nvidia/nvidia-a100')
weight_tensor_int8 = matmul.transform_weight(weight_int8_edq)
for _ in range(10):
    output_tensor = matmul(input_tensor, weight_tensor_int8, scale=scale_w)
with timer('BitBLAS'):
    output_tensor = matmul(input_tensor, weight_tensor_int8, scale=scale_w)
torch.cuda.synchronize()
print("BitBLAS Output:", output_tensor)

loss = (ref_result-output_tensor).abs().amax()
print("BitBLAS Output:", output_tensor)
print("RefFp16 Output:", ref_result)
print("Diff Output:", ref_result-output_tensor)
print('loss:', loss.item())
# print(matmul.get_source())
exit(0)

import bitblas
import torch

# enabling debug output
# bitblas.set_log_level("Debug")

model = bitblas.Linear(
    in_features=1024,
    out_features=1024,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="int8",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=False,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode=None,  # setting for how to calculating zeros
    # Target optimization var for dynamic symbolic.
    # For detailed information please checkout docs/PythonAPI.md
    # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
    opt_M=[1, 16, 32, 64, 128],
)

# Create an integer weight tensor
intweight = torch.randint(-7, 7, (1024, 1024), dtype=torch.int8).cuda()

# Load and transform weights into the BitBLAS linear module
model.load_and_transform_weight(intweight)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
dummpy_input = torch.randn(1, 1024, dtype=torch.float16).cuda()

# Perform inference
output = model(dummpy_input)
print("BitBLAS output:", output)

refout = dummpy_input @ intweight.to(torch.float16).T
print("Ref output:", refout)
# Please checkout the correctness evaluation code in `testing/python/module/test_bitblas_linear.py`

