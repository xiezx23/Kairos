import torch
import time
from utils.color_print import color_text
from kairos.quantization import *
import kairos_cuda_accel
import awq_backend
from utils.perf_eval import timer


def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()


if __name__ == '__main__':
    torch.manual_seed(11)
    group_size = 4
    x = torch.rand((4,28), dtype=torch.float16, device = 'cuda')
    w = torch.rand((10, 28), dtype=torch.float16, device = 'cuda')

    print(x)

    x_int8, config_x_int8 = quantize_tensor_int8(x, q_type='S')
    print('x_int8', x_int8)
    x_int4, config_x_int4 = quantize_tensor_int4(x_int8, q_type='A', group_size=group_size)
    print('x_int4', x_int4)
    x_deq = dequantize_tensor_int4(x_int4, **config_x_int4)
    x_deq = dequantize_tensor_int4(x_deq, **config_x_int8)

    # print(x_deq)
    loss = relative_error(x, x_deq)
    print('Loss:', loss.item())

    x_int4, config_x_int4 = quantize_tensor_int4(x, q_type='A', group_size=group_size)
    print('x_int4', x_int4)
    x_deq = dequantize_tensor_int4(x_int4, **config_x_int4)
    # print(x_deq)
    loss = relative_error(x, x_deq)
    print('Loss:', loss.item())

    exit(0)