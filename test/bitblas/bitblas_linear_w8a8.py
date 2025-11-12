# RUN: python3 -m test.bitblas.linear_w8a16
import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *

torch.manual_seed(seed=10)

# bitblas.set_log_level("Debug")
def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

MSELoss = torch.nn.MSELoss()

class BitblasLinearW8A8(torch.nn.Module):
    def __init__(self, K, N, weight_tensor):
        super().__init__()
        self.model = bitblas.Linear(
            in_features=K,
            out_features=N,
            bias=False,
            A_dtype="int8",  # activation A dtype
            W_dtype="int8",  # weight W dtype
            accum_dtype="int32",  # accumulation dtype
            out_dtype="float32",  # output dtype
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=False,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode=None,  # setting for how to calculating zeros
            # Target optimization var for dynamic symbolic.
            # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
            opt_M=[1],
        )
        weight_int8, config_int8 = quantize_tensor_int8(weight_tensor)
        # Load and transform weights into the BitBLAS linear module
        self.w_scale=config_int8['scale']
        print('weight:', weight_tensor_int8.T)
        self.model.load_and_transform_weight(weight_int8)
        self.model.eval().cuda()

    def forward(self, input, x_scale):
        # print(self.w_scale.shape)
        # print(x_scale.shape)
        # print(input.shape)
        # exit(0)
        print(input)
        out = self.model(input)
        print(out.to(torch.int32))
        exit(0)
        # return self.model(input).mul_(self.w_scale.T).mul_(x_scale)


if __name__ == '__main__':
    M = 2
    K = 32
    N = 64

    # Create input matrices
    input_tensor = torch.randn((M, K), dtype=torch.float16, device = 'cuda')
    input_tensor_int8, config_input = quantize_tensor_int8(input_tensor)
    input_tensor_q = dequantize_tensor_int8(input_tensor_int8, **config_input)
    input_tensor_int8 = torch.zeros_like(input_tensor, dtype=torch.int8)
    input_tensor_int8[0][0] = 1
    input_tensor_int8[0][1] = 1

    weight_tensor = torch.randn((N, K), dtype=torch.float16, device = 'cuda')
    weight_tensor_int8, config_weight = quantize_tensor_int8(weight_tensor.clone())
    weight_tensor_q = dequantize_tensor_int8(weight_tensor_int8, **config_weight)

    print('input:', input_tensor_int8)
    print('weight:', weight_tensor_int8.T)

    # ref_quant_out = torch._int_mm(input_tensor_int8, weight_tensor_int8.T)
    ref_quant_out = input_tensor_int8.to(torch.float32) @ weight_tensor_int8.to(torch.float32).T
    print(ref_quant_out.to(torch.int32))

    model_w8a8 = BitblasLinearW8A8(K, N, weight_tensor)

    qua_output = input_tensor_q @ weight_tensor_q.T

    ref_output = input_tensor @ weight_tensor.T

    model_output = model_w8a8(input_tensor_int8, config_input['scale'])

    ref_output = ref_output.to(torch.float32)
    qua_output = qua_output.to(torch.float32)
    model_output = model_output.to(torch.float32)

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