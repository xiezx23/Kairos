# RUN: python3 -m test.bitblas.linear_w8a16
import bitblas
import torch
import edq_cuda_accel
import awq_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from thirdparty.AWQ.awq_method import quant_weight_awq, calculate_zeros_width

torch.manual_seed(seed=10)

# bitblas.set_log_level("Debug")

class BitblasLinearW4A16(torch.nn.Module):
    def __init__(self, K, N, group_size, weight_tensor):
        super().__init__()
        self.model = bitblas.Linear(
            in_features=K,
            out_features=N,
            bias=False,
            A_dtype="float16",  # activation A dtype
            W_dtype="uint4",  # weight W dtype
            accum_dtype="float32",  # accumulation dtype
            out_dtype="float16",  # output dtype
            # configs for weight only quantization
            group_size=group_size,  # setting for grouped quantization
            with_scaling=True,  # setting for scaling factor
            with_zeros=True,  # setting for zeros
            zeros_mode='original',  # setting for how to calculating zeros
            # Target optimization var for dynamic symbolic.
            # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
            # opt_M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
            # opt_M=[1, 16, 32, 64],  # M dimension
            opt_M=[1, 4, 8, 16, 32, 64],  # M dimension
        )
        weight_int4, config_int4 = quantize_tensor_int4(weight_tensor, group_size=group_size)
        # Load and transform weights into the BitBLAS linear module
        self.model.load_and_transform_weight(weight_int4, config_int4['scale'], config_int4['zero_pt'])

        self.model.eval().cuda()

    def forward(self, input):
        return self.model(input)

    def get_source(self):
        return self.model.bitblas_matmul.get_source()

def awq_forward(input, qweight, scales, scaled_zeros):
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

if __name__ == '__main__':
    M = 8192
    K = 3584
    N = 3584
    group_size = min(128, K)
    assert K % group_size == 0


    weight_tensor = torch.randn((N, K), dtype=torch.float16, device = 'cuda')
    weight_tensor_q = simu_quantize_tensor(weight_tensor, bit=4, q_type='A', group_size=group_size)

    qweight, scales, scaled_zeros = quant_weight_awq(weight_tensor.clone(), group_size)

    model_w4a16 = BitblasLinearW4A16(K, N, group_size, weight_tensor)

    bitblas_matmul = model_w4a16.model.bitblas_matmul

    weight_int4_edq, config_int4_edq = quantize_tensor_int4(weight_tensor, group_size=group_size)
    weight_tensor_int4 = bitblas_matmul.transform_weight(weight_int4_edq, scale=config_int4_edq['scale'],
                        zeros=config_int4_edq['zero_pt'])
    weight_tensor_dq = dequantize_tensor_int4(weight_int4_edq, **config_int4_edq)

    mat_args = {'scale':config_int4_edq['scale'].contiguous(), 'zeros':config_int4_edq['zero_pt'].contiguous()}

    input_len = [i for i in range(8, 2048, 64)]
    input_len = [1, 64]

    # prehit
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        qua_output = input_tensor @ weight_tensor_q.T

        for _ in range(5):
            ref_output = input_tensor @ weight_tensor.T
            torch.cuda.synchronize()
        for _ in range(5):
            awq_output = awq_forward(input_tensor, qweight, scales, scaled_zeros)
            torch.cuda.synchronize()
        
        for _ in range(5):
            model_output = bitblas_matmul(input_tensor, weight_tensor_int4, **mat_args)
            torch.cuda.synchronize()
       
        for _ in range(5):
            model_output = model_w4a16(input_tensor)
            torch.cuda.synchronize()
    n = 200
    for m in input_len:
        print('input M: ', m)
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        qua_output = input_tensor @ weight_tensor_q.T

        # for _ in range(5):
        #     ref_output = input_tensor @ weight_tensor.T
        #     torch.cuda.synchronize()
        with timer('Pytorch ', n = n):
            for _ in range(n):
                ref_output = input_tensor @ weight_tensor.T
        # print('Ref Orin:', ref_output)
        # print('Ref Quan:', qua_output)
        print(gre_prefix, end='')
        # for _ in range(5):
        #     model_output = model_w4a16(input_tensor)
        #     torch.cuda.synchronize()
        with timer('BitBLAS ', n = n):
            for _ in range(n):
                model_output = model_w4a16(input_tensor)
        # print('BitBLAS', model_output)
        print(default_color, end='')

        # for _ in range(5):
        #     awq_output = awq_forward(input_tensor, qweight, scales, scaled_zeros)
        #     torch.cuda.synchronize()
        with timer('AWQ     ', n = n):
            for _ in range(n):
                awq_output = awq_forward(input_tensor, qweight, scales, scaled_zeros)
        # print(awq_output)

        print(red_prefix, end='')
        # for _ in range(5):
        #     model_output = bitblas_matmul(input_tensor, weight_tensor_int4)
        #     torch.cuda.synchronize()
        input_tensor = bitblas_matmul.transform_input(input_tensor)
        with timer('BitMat  ', n = n):
            for _ in range(n):
                model_output = bitblas_matmul(input_tensor, weight_tensor_int4, **mat_args)
        # print('BitMat', model_output)
        print(default_color, end='')

        
        # print(ref_output)
        # print(model_output)
        # print(awq_output)

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

    # print(model_w4a16.get_source())    
    exit(0)