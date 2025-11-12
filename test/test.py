import torch
from utils.perf_eval import timer
import matplotlib.pyplot as plt

torch.manual_seed(seed=10)


def test(K, N):

    # Create input matrices
    weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()


    input_len = [i for i in range(1, 16, 1)]

    # Preheat Kernels
    preheat_time = 20
    for m in input_len:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for _ in range(preheat_time):
            ref_output = input_tensor @ weight_tensor.T
    n = 100
    print_flag = True
    kernel_name_list = ['Pytorch  FP16']
    recordList = [[] for _ in range(len(kernel_name_list))]
    for midx in range(len(input_len)):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N}')
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for _ in range(preheat_time):
            ref_output = input_tensor @ weight_tensor.T
        with timer('Pytorch  FP16', n = n, recordList=recordList[0], print_flag=print_flag):
            for _ in range(n):
                ref_output = input_tensor @ weight_tensor.T
        print('-' * 30)
        
    plt.figure()
    plt.plot(input_len, recordList[0], color = 'r')
    plt.xlabel('m')
    plt.ylabel('latency')
    plt.grid(True)

    
if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(3584, 18944), (18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])
    plt.show()

# int_x = torch.zeros_like(x, dtype=torch.int8, device='cuda')
# out1 = torch.zeros((x.shape[0], w.shape[0]), device='cuda', dtype=torch.float16)
# scale_x = torch.zeros(x.shape[0], device='cuda', dtype=torch.float16)
# int_w = torch.zeros_like(w, dtype=torch.int8, device='cuda')
# scale_w = torch.zeros(w.shape[0], device='cuda', dtype=torch.float16)
# awq_backend.w8a8_gemm_forward_cuda(int_x, int_w, scale_w, scale_x, out1)
# torch.cuda.synchronize()
# print('#####')
# torch.cuda.synchronize()
# with timer('W8A8 GEMM'):
#     edq_cuda_accel.quant_fp16_to_int8(int_x, x, scale_x)
#     # torch.cuda.synchronize()
#     tmp = int_w.clone()
#     awq_backend.w8a8_gemm_forward_cuda(int_x, tmp, scale_w, scale_x, out1)




# loss = (ori_out - out1).pow(2).to(torch.float16).mean().sqrt().item()
# print(loss)

# loss = (ori_out - out2).pow(2).to(torch.float16).mean().sqrt().item()
# print(loss)

# loss = (out1 - out2).pow(2).to(torch.float16).mean().sqrt().item()
# print(loss)