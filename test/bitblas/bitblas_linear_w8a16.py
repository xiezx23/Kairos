# RUN: python3 -m test.bitblas.linear_w8a16
import bitblas
import torch
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *

torch.manual_seed(seed=10)

# bitblas.set_log_level("Debug")

class BitblasLinearW8A16(torch.nn.Module):
    def __init__(self, K, N, weight_tensor):
        super().__init__()
        self.model = bitblas.Linear(
            in_features=K,
            out_features=N,
            bias=False,
            A_dtype="float16",  # activation A dtype
            W_dtype="int8",  # weight W dtype
            accum_dtype="float32",  # accumulation dtype
            out_dtype="float16",  # output dtype
            # configs for weight only quantization
            group_size=None,  # setting for grouped quantization
            with_scaling=True,  # setting for scaling factor
            with_zeros=False,  # setting for zeros
            zeros_mode='original',  # setting for how to calculating zeros
            # Target optimization var for dynamic symbolic.
            # By default, the optimization var is [1, 16, 32, 64, 128, 256, 512]
            opt_M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048],
        )
        weight_int8, config_int8 = quantize_tensor_int8(weight_tensor)
        # Load and transform weights into the BitBLAS linear module
        self.model.load_and_transform_weight(weight_int8, scales=config_int8['scale'])
        self.model.eval().cuda()

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    M = 16
    K = 3584
    N = 3584

    # Create input matrices
    input_tensor = torch.randn((M, K), dtype=torch.float16, device = 'cuda')
    weight_tensor = torch.randn((N, K), dtype=torch.float16, device = 'cuda')
    weight_tensor_q = simu_quantize_tensor(weight_tensor, bit=8)

    model_w8a16 = BitblasLinearW8A16(K, N, weight_tensor)

    qua_output = input_tensor @ weight_tensor_q.T

    ref_output = input_tensor @ weight_tensor.T

    model_output = model_w8a16(input_tensor)

    loss = (model_output - ref_output).abs().mean() / ref_output.abs().mean()
    ideal_loss = (qua_output - ref_output).abs().mean() / ref_output.abs().mean()
    accu = (model_output - qua_output).abs().mean() / qua_output.abs().mean()
    print(f'Loss      : {loss.item():.4f} %')
    print(f'Accuracy  : {accu.item():.4f} %')
    print(f'Ideal Loss: {ideal_loss.item():.4f} %')

    exit(0)