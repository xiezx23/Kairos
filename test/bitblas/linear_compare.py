# RUN: python3 -m test.bitblas.linear_compare

import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *

# enabling debug output
# bitblas.set_log_level("Debug")
torch.manual_seed(seed=10)
M = 16
K = 3584
N = 3584

model = bitblas.Linear(
    in_features=K,
    out_features=N,
    bias=False,
    A_dtype="float16",  # activation A dtype
    W_dtype="int8",  # weight W dtype
    accum_dtype="float16",  # accumulation dtype
    out_dtype="float16",  # output dtype
    # configs for weight only quantization
    group_size=None,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=False,  # setting for zeros
    zeros_mode='original',  # setting for how to calculating zeros
    # Target optimization var for dynamic symbolic.
    # For detailed information please checkout docs/PythonAPI.md
    # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
    opt_M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
)

# Create input matrices
input_tensor  = torch.randn((M, K), dtype=torch.float16).cuda()
# input_tensor   = torch.eye((K), dtype=torch.float16).cuda()
# input_tensor   = torch.randint(-64, 64 + 1, (M, K), dtype=torch.int8).cuda()
weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()

weight_int8_edq, config_int8_edq = quantize_tensor_int8(weight_tensor)
weight_tensor = dequantize_tensor_int8(weight_int8_edq, config_int8_edq['scale'])

# Load and transform weights into the BitBLAS linear module
model.load_and_transform_weight(weight_int8_edq, scales=config_int8_edq['scale'])
# Set the model to evaluation mode
model.eval()
model.cuda()

# Preheating Model
output = model(input_tensor)
refout = input_tensor @ weight_tensor.T

n = 4
input_len = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)

recordList = [[[] for _ in range(2)] for _ in range(len(input_len))]
test_time = 20
for _ in range(test_time):
    for midx in range(len(input_len)):
        m = input_len[midx]
        # print('input M: ', m)
        input_tensor  = torch.randn((m, K), dtype=torch.float16).cuda()

        with timer('BitBLAS', recordList=recordList[midx][0], print_flag=False):
            output = model(input_tensor)
        
        with timer('RefFp16', recordList=recordList[midx][1], print_flag=False):
            refout = input_tensor @ weight_tensor.T

        loss = (refout-output).abs().mean()
        # print('loss:', loss.item())
        # print('---------------------')

def aver(lst):
    return sum(lst) / len(lst)

print("---------AVERAGE REPORT---------")
for midx in range(len(input_len)):
    m = input_len[midx]
    print('input M: ', m)
    bitblas_aver = aver(recordList[midx][0])
    torch_f_aver = aver(recordList[midx][1])
    print(gre_prefix, end='')
    print('BitBLAS time:', bitblas_aver)
    print(default_color, end='')
    print('RefFp16 time:', torch_f_aver)
    print('---------------------')
