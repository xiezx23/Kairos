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
# bitblas.set_log_level("Debug")

torch.manual_seed(seed=10)

enable_tuning=True

def awq_forward(input, qweight, scales, scaled_zeros, N, K, group_size):
    batch_token_len = input.numel() // input.shape[-1]
    if batch_token_len < 8:
        out = awq_backend.gemv_forward_cuda_new(
            input, qweight, scales, scaled_zeros,
            batch_token_len, N, K, group_size)
    else:
        out = awq_backend.gemm_forward_cuda_new(
            input, qweight, scales, scaled_zeros)
    return out

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

MSELoss = torch.nn.MSELoss()

def test(K, N):
    group_size = min(128, K)

    matmul_config = bitblas.MatmulConfig(
        M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],  # M dimension
        # M=[4],  # M dimension
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

    # Create input matrices
    weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()
    # # Quantize weight tensor from edq
    weight_int4_edq, config_int4_edq = quantize_tensor_int4(weight_tensor, group_size=group_size)
    weight_tensor_q = dequantize_tensor_int4(weight_int4_edq, **config_int4_edq)
    mat_args = {'scale':config_int4_edq['scale'], 'zeros':config_int4_edq['zero_pt']}
    qweight, scales, scaled_zeros = quant_weight_awq(weight_tensor.clone(), group_size)

    bitblas_matmul = global_operator_cache.get(matmul_config)
    if bitblas_matmul is None:
        # should disable tuning for the first time because we may require loading bitblas operator from database.
        bitblas_matmul = bitblas.Matmul(matmul_config, target=BITBLAS_TARGET, enable_tuning=False)
        if enable_tuning:
            print('BitBLAS Tuning...')
            bitblas_matmul.hardware_aware_finetune(topk=20)
            global_operator_cache.add(matmul_config, bitblas_matmul)
            global_operator_cache.save_into_database(BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
            print("BitBLAS Tuning done, appended operator to global_operator_cache.")
        else:
            print("BitBLAS Operator created.")
    else:
        print("BitBLAS Operator found in global_operator_cache.")

    # bitblas_matmul = bitblas.Matmul(config=matmul_config, target='nvidia/nvidia-a100', enable_tuning=enable_tuning)
    weight_tensor_int4 = bitblas_matmul.transform_weight(weight_int4_edq)

    input_len = [i for i in range(1, 16, 1)]
    input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for _ in range(10):
            ref_output = input_tensor @ weight_tensor.T
            torch.cuda.synchronize()
        for _ in range(10):
            model_output = bitblas_matmul(input_tensor, weight_tensor_int4, **mat_args)
            torch.cuda.synchronize()
        for _ in range(10):
            awq_output = awq_forward(input_tensor, qweight, scales, scaled_zeros, N, K, group_size)
            torch.cuda.synchronize()

    n = 100
    for m in input_len:
        print('input M: ', m)
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        qua_output = input_tensor @ weight_tensor_q.T

        with timer('Pytorch ', n = n):
            for _ in range(n):
                ref_output = input_tensor @ weight_tensor.T

        print(gre_prefix, end='')
        with timer('BitBLAS ', n = n):
            for _ in range(n):
                model_output = bitblas_matmul(input_tensor, weight_tensor_int4, **mat_args)
        print(default_color, end='')

        with timer('AWQ     ', n = n):
            for _ in range(n):
                awq_output = awq_forward(input_tensor, qweight, scales, scaled_zeros, N, K, group_size)
        
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
        print('-' * 20)

    # print(bitblas_matmul.get_source())
    
if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(3584, 18944), (18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])