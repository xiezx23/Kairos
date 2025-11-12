import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer

torch.manual_seed(seed=10)

# bitblas.set_log_level("Debug")
M = 16
K = 16
N = 8
# group_size = min(16, K)
group_size = -1

matmul_config = bitblas.MatmulConfig(
    M=(1, 4, M),  # M dimension
    N=N,    # N dimension
    K=K,    # K dimension
    A_dtype="int8",      # activation A dtype
    W_dtype="uint4",        # weight W dtype
    accum_dtype="int32",  # accumulation dtype
    out_dtype="float16",    # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode='original',  # setting for how to calculating zeros
)

# # Create input matrices
# input_tensor  = torch.randint(-127, 127 + 1, (M, K), dtype=torch.int8).cuda()
# weight_tensor = torch.randint(0, 16, (N, K), dtype=torch.int8).cuda()
# # # Quantize weight tensor from edq
# with timer('RefFp16'):
#     ref_result = input_tensor.to(torch.float16) @ weight_tensor.T.to(torch.float16)
# print("RefFp16 Output:", ref_result)

# matmul = bitblas.Matmul(config=matmul_config, target='nvidia/nvidia-a100')
# weight_tensor_int4 = matmul.transform_weight(weight_tensor)

# with timer('BitBLAS'):
#     # Perform mixed-precision matrix multiplication
#     output_tensor=matmul(input_tensor, weight_tensor_int4)
# print("BitBLAS Output:", output_tensor)
    
# # # Assert that the results are close within a specified tolerance, note that the int4 randint value is a little bigger than the float16 value, so we set the atol to 1.0
# loss = (ref_result-output_tensor).abs().amax()
# print('loss:', loss.item())

# Create input matrices
input_tensor   = torch.eye((K), dtype=torch.int8).cuda()
# input_tensor   = torch.randint(-64, 64 + 1, (M, K), dtype=torch.int8).cuda()
# weight_tensor  = torch.zeros((N, K), dtype=torch.int8).cuda()
# weight_tensor[0][0] = 32
# weight_tensor[0][1] = 48
# weight_tensor[0][2] = 32
# weight_tensor[0][3] = 48
weight_tensor  = torch.randint(-64, 64 + 1, (N, K), dtype=torch.int8).cuda()
# weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

weight_tensor = weight_tensor.to(torch.float16)
# # Quantize weight tensor from edq
# with timer('OriFp16'):
#     ref_result = input_tensor.to(torch.float32) @ weight_tensor.T.to(torch.float32)
# print("OriFp16 Output:", ref_result)

weight_int4_edq, config_int4_edq = quantize_tensor_int4(weight_tensor, group_size=group_size)

scales = config_int4_edq['scale'].to(torch.int8)
zeros  = config_int4_edq['zero_pt'].to(torch.int8)
weight_tensor = dequantize_tensor_int4(weight_int4_edq, scales,
                                       zero_pt=zeros, group_size=group_size)

with timer('RefFp16'):
    ref_result = input_tensor.to(torch.float32) @ weight_tensor.T.to(torch.float32)
print("RefFp16 Output:", ref_result)

matmul = bitblas.Matmul(config=matmul_config, target='nvidia/nvidia-a100')
weight_tensor_int4 = matmul.transform_weight(weight_int4_edq)
print(weight_int4_edq)
print(weight_tensor_int4)


with timer('BitBLAS'):
    output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scales,
                        zeros=zeros)
torch.cuda.synchronize()
print("BitBLAS Output:", output_tensor)

with timer('BitBLAS'):
    output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scales,
                        zeros=zeros)
torch.cuda.synchronize()
print("BitBLAS Output:", output_tensor)

with timer('BitBLAS'):
    output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scales,
                        zeros=zeros)
torch.cuda.synchronize()
print("BitBLAS Output:", output_tensor)

with timer('BitBLAS'):
    output_tensor = matmul(input_tensor, weight_tensor_int4, scale=scales,
                        zeros=zeros)
torch.cuda.synchronize()
print("BitBLAS Output:", output_tensor)

loss = (ref_result-output_tensor).abs().amax()
print('loss:', loss.item())
