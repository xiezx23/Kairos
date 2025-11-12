import bitblas
import torch
import edq_cuda_accel
import awq_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width
from bitblas.cache import global_operator_cache, get_database_path

BITBLAS_DATABASE_PATH = get_database_path()
BITBLAS_TARGET = 'nvidia/nvidia-a100'
global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)

torch.manual_seed(seed=10)


def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

MSELoss = torch.nn.MSELoss()

in_features = 3584
out_features = 3584
K = in_features
N = out_features
group_size = 128
enable_tuning = True

matmul_config = bitblas.MatmulConfig(
    M=[1, 16, 32, 64],  # M dimension
    N=N,    # N dimension
    K=K,    # K dimension
    A_dtype="float16",      # activation A dtype
    W_dtype="uint4",        # weight W dtype
    accum_dtype="float32",  # accumulation dtype
    out_dtype="float16",    # output dtype
    layout="nt",  # matrix layout, "nt" indicates the layout of A is non-transpose and the layout of W is transpose
    with_bias=False,  # bias
    # configs for weight only quantization
    group_size=group_size,  # setting for grouped quantization
    with_scaling=True,  # setting for scaling factor
    with_zeros=True,  # setting for zeros
    zeros_mode='original',  # setting for how to calculating zeros
)

weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()
# # Quantize weight tensor from edq
weight_int4_edq, config_int4_edq = quantize_tensor_int4(weight_tensor, group_size=group_size)
weight_tensor_dq = dequantize_tensor_int4(weight_int4_edq, **config_int4_edq)
mat_args = {'scale':config_int4_edq['scale'], 'zeros':config_int4_edq['zero_pt']}

matmul = global_operator_cache.get(matmul_config)
if matmul is None:
    # should disable tuning for the first time because we may require loading bitblas operator from database.
    matmul = bitblas.Matmul(matmul_config, target=BITBLAS_TARGET, enable_tuning=False)
    if enable_tuning:
        print('BitBLAS Tuning...')
        matmul.hardware_aware_finetune(topk=20)
        global_operator_cache.add(matmul_config, matmul)
        global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
        print("BitBLAS Tuning done, appended operator to global_operator_cache.")
    else:
        print("BitBLAS Operator created.")
else:
    print("BitBLAS Operator found in global_operator_cache.")

weight_tensor_int4 = matmul.transform_weight(weight_int4_edq)
    

input_len = [i for i in range(1, 64, 8)]


for m in input_len:
    # Create input matrices
    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
    for _ in range(10):
        ref_output = input_tensor @ weight_tensor.T
        torch.cuda.synchronize()
    for _ in range(10):
        model_output = matmul(input_tensor, weight_tensor_int4, **mat_args)
        torch.cuda.synchronize()

for M in input_len:
    print('input M: ', M)
    # Create input matrices
    input_tensor  = torch.randn((M, K), dtype=torch.float16).cuda()
    ref_output = input_tensor @ weight_tensor.T

    print(gre_prefix, end='')
    with timer('BitBLAS '):
        model_output = matmul(input_tensor, weight_tensor_int4, **mat_args)
    print(default_color, end='')

    with timer('Pytorch '):
        qua_output = input_tensor @ weight_tensor_dq.T

    # print("Ref output:", ref_output)
    # print("BitBLAS output:", model_output)


    mse_loss = relative_error(ref_output, model_output)
    acu_loss = relative_error(qua_output, model_output)
    ide_loss = relative_error(ref_output, qua_output)
    print(f'RE  to Full Precision   : {mse_loss.item():.4f}')
    print(f'RE  to Fake Quantization: {acu_loss.item():.4f}')
    print(f'RE  of Fake Quantization: {ide_loss.item():.4f}')
    mse_loss = MSELoss(ref_output, model_output)
    acu_loss = MSELoss(qua_output, model_output)
    ide_loss = MSELoss(ref_output, qua_output)
    print(f'MSE to Full Precision   : {mse_loss.item():.4f}')
    print(f'MSE to Fake Quantization: {acu_loss.item():.4f}')
    print(f'MSE of Fake Quantization: {ide_loss.item():.4f}')

exit(0)