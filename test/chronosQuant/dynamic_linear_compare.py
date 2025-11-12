import torch
import numpy
import matplotlib.pyplot as plt
import edq_cuda_accel
import awq_backend
import omniserve_backend
from edq.quantization import *
from utils.perf_eval import timer
from utils.color_print import *
from edq.dynamic_linear import DynamicLinear

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

sub_stream = torch.cuda.Stream()
enable_tuning=True
MSELoss = torch.nn.MSELoss()

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))

def test(K, N, bias):
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')

    preheat_time = 20
    input_len = [1024*8, 1024*16, 1024*32]
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for _ in range(preheat_time):
            out = dlinear(input_tensor)
        torch.cuda.empty_cache()
        with timer(kernel_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):

        
if __name__ == '__main__':
    w_shape_proj = ['k_proj, v_proj', 'q_proj',
                'o_proj', 'gate_proj, up_proj', 'down_proj']
    w_shape = [(3584, 512, True), (3584, 3584, True),
               (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    # w_shape = [(3584, 512, False), (3584, 3584, False),
    #            (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    # w_shape = [(4096, 512, False), (4096, 4096, False),
    #            (4096, 4096, False), (4096, 18944, False), (18944, 4096, False)]
    for i in range(len(w_shape)):
        print('-'*20)
        print(w_shape_proj[i])
        s = w_shape[i]
        test(s[0], s[1], s[2])
    plt.show()

