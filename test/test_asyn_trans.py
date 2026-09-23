import torch
import matplotlib.pyplot as plt
import kairos_cuda_accel
import awq_backend
import omniserve_backend
import torch.cuda.nvtx as nvtx
from utils.perf_eval import timer
from thirdparty.Qserve.w4a8_linear import W4A8Linear


def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

def check_output(target, current):
    re = relative_error(target, current).item()*100
    if (re >= 20 or re != re):
        print(color_text('red', f'RE  to Full Precision   : {re:3.2f} %'))


stream = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

class test:
    def _test(K, N):
        group_size = min(128, K)

        # Create input matrices
        weight_tensor = torch.randn((N, K), dtype=torch.float16).cuda()
        # Prepare Qserve W4A8
        qserve_w4a8 = W4A8Linear.from_weight(
            weight_tensor, None, group_size)
        qserve_args = (qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales,)
        qweight_cpu = qserve_w4a8.qweight.cpu()

        input_len = [i for i in range(1, 16, 2)]
        # input_len = [1, 16, 32, 64, 128, 256, 512, 1024]
        # input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024]

        # Preheat Kernels
        preheat_time = 20
        for m in input_len:
            # Create input matrices
            input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
            for _ in range(preheat_time):
                ref_output = input_tensor @ weight_tensor.T
            for _ in range(preheat_time):
                input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                    input_int8, *qserve_args, scale_x, qse_output,
                )
            check_output(ref_output, qse_output)

        n = 10
        for m in input_len:
            with timer('Qserve   W4A8 v1', n = n, print_flag=True):
                for _ in range(n):
                    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )

            with timer('Qserve   W4A8 v1', n = n, print_flag=True):
                for _ in range(n):
                    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )

            qserve_w4a8.qweight.cpu()
            torch.cuda.synchronize()
            with timer('Qserve   W4A8 v2', n = n, print_flag=True):
                for _ in range(n):
                    with torch.cuda.stream(stream):
                        qserve_w4a8.qweight.cuda()
                    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    torch.cuda.synchronize()
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )
            qserve_w4a8.qweight.cpu()
            torch.cuda.synchronize()
            with timer('Qserve   W4A8 v2', n = n, print_flag=True):
                for _ in range(n):
                    qserve_w4a8.qweight.to('cuda', non_blocking=True)
                    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qserve_w4a8.qweight, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )
            torch.cuda.synchronize()
            with timer('Qserve   W4A8 v2', n = n, print_flag=True):
                for _ in range(n):
                    qweight_gpu = qweight_cpu.cuda(non_blocking = True)
                    input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
                    input_int8 = torch.empty_like(input_tensor, dtype=torch.int8, device='cuda')
                    scale_x = torch.empty(input_tensor.shape[0], device='cuda', dtype=torch.float16)
                    kairos_cuda_accel.quant_fp16_to_int8(input_int8, input_tensor, scale_x)
                    qse_output = torch.empty(input_tensor.shape[0], weight_tensor.shape[0], device='cuda', dtype=torch.float16)
                    omniserve_backend.qgemm_w4a8_per_group.gemm_forward_cuda(
                        input_int8, qweight_gpu, qserve_w4a8.s2_zeros, qserve_w4a8.s2_scales, qserve_w4a8.s1_scales, scale_x, qse_output,
                    )
            with timer('trans time      ', n = n, print_flag=True):
                for _ in range(n):
                    qweight_gpu = qweight_cpu.cuda(non_blocking = True)
            print("-----------")

# test._test(512, 512)

class Layer:
    M = 1024; K = 4096; N = 4096*2
    w = torch.randn((N, K), dtype=torch.float16, device='cuda')
    w2 = torch.randn((N, K), dtype=torch.float16, device='cuda')
    a = torch.randn((M, K), dtype=torch.float16, device='cuda')
    w_cpu = w.cpu()
    w_gpu = torch.empty((N, K), dtype=torch.float16, device='cuda')
    o = torch.zeros((M, N), dtype=torch.float16, device='cuda')

    def asyn_prefetch(self):
        with torch.cuda.stream(stream):
            self.w_gpu = self.w_cpu.to('cuda', non_blocking = True)
        return
    
    def compute(self):
        self.o = self.a @ self.w.T

layer = Layer()

for _ in range(20):
    layer.asyn_prefetch()
    layer.compute()
torch.cuda.synchronize()

nvtx.range_push(f"Test")

with timer('', n = 1, print_flag=True):
    for _ in range(1):
        layer.asyn_prefetch()

with timer('', n = 1, print_flag=True):
    for _ in range(1):
        layer.compute()

with timer('', n = 1, print_flag=True):
    for _ in range(1):
        layer.asyn_prefetch()
        layer.compute()

with timer('', n = 1, print_flag=True):
    for _ in range(1):
        layer.compute()
        layer.asyn_prefetch()

nvtx.range_pop()
    
# nsys profile -w true -t cuda,nvtx,osrt -o profile -f true python -m test.test_asyn_trans
# nsys-ui profile.nsys-rep 