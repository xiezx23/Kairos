import gc
import torch
from edq.quantization import *
from thirdparty.AWQ.awq_method import quant_weight_awq
from utils.perf_eval import timer
from utils.color_print import *
import matplotlib.pyplot as plt

from thirdparty.AWQ.w4a16_linear import W4A16Linear

# import kernels.edq_cuda_accel as edq_cuda_accel

import edq_cuda_accel
import awq_backend


torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# torch.cuda.set_device(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

warmup = 20
repeat = 100

def relative_error(target, current):
    return (target - current).abs().to(torch.float32).sum() / (target).abs().to(torch.float32).sum()

# 为了避免其他操作的干扰这里测试Kernel时直接调用
# 最多加上条件判断区分gemm和gemv

@torch.no_grad()
def fp16_test(M, N, K, recordList):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)

    for _ in range(warmup):
        C = A @ B.T
    
    torch.cuda.synchronize()
    with timer("FP16 GEMM", n=repeat, recordList=recordList):
        for _ in range(repeat):
            C = A @ B.T
    torch.cuda.synchronize()


@torch.no_grad()
def awq_w8a8_test(M, N, K, recordList):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C = torch.zeros(size=(M, N), dtype=torch.float16, device=device)
    C_t = A @ B.T

    # A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
    # A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
    B_quant = torch.empty_like(B, dtype=torch.int8, device=device)
    B_scale = torch.empty(B.shape[0], dtype=torch.float16, device=device)
    edq_cuda_accel.quant_fp16_to_int8(B_quant, B, B_scale)

    for _ in range(warmup):
        A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
        A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
        edq_cuda_accel.quant_fp16_to_int8(A_quant, A, A_scale)
        if M < 8:
            awq_backend.w8a8_gemm_forward_cuda(
                A_quant, B_quant, B_scale, A_scale, C
            )
        else:
            awq_backend.w8a8_gemm_forward_cuda(
                A_quant, B_quant, B_scale, A_scale, C
            )
    
    torch.cuda.synchronize()
    with timer("AWQ w8a8", n=repeat, recordList=recordList):
        for _ in range(repeat):
            A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
            A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
            edq_cuda_accel.quant_fp16_to_int8(A_quant, A, A_scale)
            if M < 8:
                awq_backend.w8a8_gemm_forward_cuda(
                    A_quant, B_quant, B_scale, A_scale, C
                )
            else:
                awq_backend.w8a8_gemm_forward_cuda(
                    A_quant, B_quant, B_scale, A_scale, C
                )
    torch.cuda.synchronize()
    loss = relative_error(C_t, C)
    print(f"awq w8a8 loss: {loss}")


@torch.no_grad()
def edq_w8a8_test(M, N, K, recordList, kernel_type='V'):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C = torch.empty(size=(M, N), dtype=torch.float16, device=device)
    C_t = A @ B.T

    # TODO: 把A_quant个A_sacale放进去测试
    A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
    A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
    B_quant = torch.empty_like(B, dtype=torch.int8, device=device)
    B_scale = torch.empty(B.shape[0], dtype=torch.float16, device=device)
    edq_cuda_accel.quant_fp16_to_int8(B_quant, B, B_scale)

    for _ in range(warmup):
        # A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
        # A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
        edq_cuda_accel.quant_fp16_to_int8(A_quant, A, A_scale)
        if M < 8 and kernel_type == 'V':
            edq_cuda_accel.w8a8_wsas_gemv_cuda(
                A_quant, B_quant, A_scale, B_scale, C
            )
        else:
            edq_cuda_accel.w8a8_wsas_gemm_cuda(
                A_quant, B_quant, A_scale, B_scale, C
            )
    torch.cuda.synchronize()
    with timer("EDQ w8a8", n=repeat, recordList=recordList):
        for _ in range(repeat):
            # A_quant = torch.empty_like(A, dtype=torch.int8, device=device)
            # A_scale = torch.empty(A.shape[0], dtype=torch.float16, device=device)
            edq_cuda_accel.quant_fp16_to_int8(A_quant, A, A_scale)
            if M < 8 and kernel_type == 'V':
                edq_cuda_accel.w8a8_wsas_gemv_cuda(
                    A_quant, B_quant, A_scale, B_scale, C
                )
            else:
                edq_cuda_accel.w8a8_wsas_gemm_cuda(
                    A_quant, B_quant, A_scale, B_scale, C
                )
    torch.cuda.synchronize()
    loss = relative_error(C_t, C)
    assert(loss < 0.05)
    print(f"edq w8a8 loss: {loss}")

@torch.no_grad()
def edq_w8a16_sym_test(M, N, K, recordList, kernel_type='V'):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C = torch.empty(size=(M, N), dtype=torch.float16, device=device)
    C_t = A @ B.T

    B_quant = torch.empty_like(B, dtype=torch.int8, device=device)
    B_scale = torch.empty(B.shape[0], dtype=torch.float16, device=device)
    edq_cuda_accel.quant_fp16_to_int8(B_quant, B, B_scale)

    for _ in range(warmup):
        if M < 8 and kernel_type == 'V':
            edq_cuda_accel.w8a16_ws_gemv_cuda(
                A, B_quant, B_scale, C
            )
        else:
            edq_cuda_accel.w8a16_ws_gemm_cuda(
                A, B_quant, B_scale, C
            )
    torch.cuda.synchronize()
    with timer("EDQ w8a16 sym", n=repeat, recordList=recordList):
        for _ in range(repeat):
            if M < 8 and kernel_type == 'V':
                edq_cuda_accel.w8a16_ws_gemv_cuda(
                    A, B_quant, B_scale, C
                )
            else:
                edq_cuda_accel.w8a16_ws_gemm_cuda(
                    A, B_quant, B_scale, C
                )
    torch.cuda.synchronize()
    loss = relative_error(C_t, C)
    assert(loss < 0.05)
    print(f"edq w8a16 sym loss: {loss}")


@torch.no_grad()
def awq_w4a16_test(M, N, K, recordList):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C = torch.empty(size=(M, N), dtype=torch.float16, device=device)
    C_t = A @ B.T

    i4w, i4s, i4z = quant_weight_awq(B, 128)
    for _ in range(warmup):
        if M < 8:
            C = awq_backend.gemv_forward_cuda_new(
                A, i4w, i4s, i4z,
                M, N, K, 128
            )
        else:
            C = awq_backend.gemm_forward_cuda_new(
                A, i4w, i4s, i4z
            )
    torch.cuda.synchronize()
    with timer("AWQ", n=repeat, recordList=recordList):
        for _ in range(repeat):
            if M < 8:
                C = awq_backend.gemv_forward_cuda_new(
                    A, i4w, i4s, i4z,
                    M, N, K, 128
                )
            else:
                C = awq_backend.gemm_forward_cuda_new(
                    A, i4w, i4s, i4z
                )
    torch.cuda.synchronize()
    loss = relative_error(C_t, C)
    print(f"awq w4a16 loss: {loss}")


@torch.no_grad()
def test(M, N, K, recordList, kernel_type = 'V'):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    A = torch.randn(M, K, device=device, dtype=dtype)    # Matrix A with shape (M, K)
    B = torch.randn(N, K, device=device, dtype=dtype)    # Matrix B with shape (N, K)
    C_t = A @ B.T

    linear = torch.nn.Linear(K, N)
    linear.weight.data = B.clone()
    linear.bias = None

    linear = W4A16Linear.from_module(linear, device, dtype)
    C = torch.empty(
        A.shape[:-1] + (linear.scales.shape[1],), dtype=linear.return_dtype, device=A.device
    )
    for _ in range(warmup):
        if M < 8 and kernel_type == 'V':
            edq_cuda_accel.w4a16_wa_gemv_cuda(
                A.view((-1, A.shape[-1])),
                linear.qweight,
                linear.scales,
                linear.scaled_zeros,
                C.view((-1, C.shape[-1])),
            )
        else:
            edq_cuda_accel.w4a16_wa_gemm_cuda(
                A.view((-1, A.shape[-1])),
                linear.qweight,
                C.view((-1, C.shape[-1])),
                linear.scales,
                linear.scaled_zeros,
                linear.workspace,
                -1, -1, -1, 16
            )
    torch.cuda.synchronize()
    with timer("Marlin", n=repeat, recordList=recordList):
        for _ in range(repeat):
            if M < 8 and kernel_type == 'V':
                edq_cuda_accel.w4a16_wa_gemv_cuda(
                    A.view((-1, A.shape[-1])),
                    linear.qweight,
                    linear.scales,
                    linear.scaled_zeros,
                    C.view((-1, C.shape[-1])),
                )
            else:
                edq_cuda_accel.w4a16_wa_gemm_cuda(
                    A.view((-1, A.shape[-1])),
                    linear.qweight,
                    C.view((-1, C.shape[-1])),
                    linear.scales,
                    linear.scaled_zeros,
                    linear.workspace,
                    -1, -1, -1, 16
                )
    torch.cuda.synchronize()
    loss = relative_error(C_t, C)
    assert(loss < 0.15)
    print(f"marlin w4a16 loss: {loss}")

@torch.no_grad()
def w8a8_gemv_test(N, K):
    input_range = [1, 2, 3, 4, 5, 6, 7]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        edq_w8a8_test(m, N, K, recordListAll[0], "V")
        edq_w8a8_test(m, N, K, recordListAll[1], "M")
        awq_w8a8_test(m, N, K, recordListAll[2])
        fp16_test(m, N, K, recordListAll[3])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'b')
    plt.plot(recordListAll[3], color = 'k')
    plt.xticks(range(len(input_range)), input_range)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Edq w8a8 GEMV', 'Edq w8a8 GEMM', 'AWQ w8a8', "FP16 GEMM"], loc='best')
    plt.title(f'W8A8 GEMV Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w8a8_gemv_wquant_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def w8a8_gemm_test(N, K):
    input_range = [i * 16 for i in range(1, 65, 1)]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        edq_w8a8_test(m, N, K, recordListAll[0])
        awq_w8a8_test(m, N, K, recordListAll[1])
        fp16_test(m, N, K, recordListAll[2])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'b')
    step = max(1, len(input_range) // 7)
    indices = range(0, len(input_range), step)
    labels = [input_range[i] for i in indices]
    plt.xticks(indices, labels)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Edq w8a8 GEMM', 'AWQ w8a8', "FP16 GEMM"], loc='best')
    plt.title(f'W8A8 GEMM Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w8a8_gemm_wquant_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def w8a16_gemv_test(N, K):
    input_range = [1, 2, 3, 4, 5, 6, 7]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        edq_w8a16_sym_test(m, N, K, recordListAll[0], "V")
        edq_w8a16_sym_test(m, N, K, recordListAll[1], "M")
        fp16_test(m, N, K, recordListAll[2])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'b')
    plt.xticks(range(len(input_range)), input_range)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Edq w8a16 GEMV', 'Edq w8a16 GEMM', "FP16 GEMM"], loc='best')
    plt.title(f'W8A16 GEMV Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w8a16_gemv_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def w8a16_gemm_test(N, K):
    input_range = [i * 16 for i in range(1, 65, 1)]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        edq_w8a16_sym_test(m, N, K, recordListAll[0])
        fp16_test(m, N, K, recordListAll[1])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    step = max(1, len(input_range) // 7)
    indices = range(0, len(input_range), step)
    labels = [input_range[i] for i in indices]
    plt.xticks(indices, labels)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Edq w8a16 GEMM', "FP16 GEMM"], loc='best')
    plt.title(f'W8A16 GEMM Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w8a16_gemm_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')



@torch.no_grad()
def w4a16_wa_gemv_test(N, K):
    input_range = [1, 2, 3, 4, 5, 6, 7]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        test(m, N, K, recordListAll[0], "V")
        test(m, N, K, recordListAll[1], "M")
        awq_w4a16_test(m, N, K, recordListAll[2])
        fp16_test(m, N, K, recordListAll[3])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'b')
    plt.plot(recordListAll[3], color = 'k')
    plt.xticks(range(len(input_range)), input_range)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Marlin w4a16 GEMV', 'Marlin w4a16 GEMM', 'AWQ w4a16', "FP16 GEMM"], loc='best')
    plt.title(f'W4A16 GEMV Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w4a16_wa_gemv_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def w4a16_wa_gemm_test(N, K):
    input_range = [i * 16 for i in range(1, 65, 1)]

    recordListAll = [[] for _ in range(4)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        test(m, N, K, recordListAll[0])
        awq_w4a16_test(m, N, K, recordListAll[1])
        fp16_test(m, N, K, recordListAll[2])

    plt.figure()
    plt.plot(recordListAll[0], color = 'r')
    plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'b')
    step = max(1, len(input_range) // 7)
    indices = range(0, len(input_range), step)
    labels = [input_range[i] for i in indices]
    plt.xticks(indices, labels)
    plt.ylabel('Time(ms)')
    plt.xlabel('Token Length')
    plt.grid(True)
    plt.legend(['Marlin w4a16', 'AWQ w4a16', "FP16 GEMM"], loc='best')
    plt.title(f'W4A16 GEMM Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w4a16_wa_gemm_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')


@torch.no_grad()
def speed_test(N, K):
    input_range = [1, 2, 3, 4, 5, 6, 7] # gemv
    # 65对应1024
    # input_range = [i * 16 for i in range(1, 65 * 4, 1)] # gemm
    # print(input_range)

    recordListAll = [[] for _ in range(10)]
    for m in input_range:
        print(f"Shape M, N, K = {m},{N},{K}")
    
        # test(m, N, K, recordListAll[0])
        # awq_w4a16_test(m, N, K, recordListAll[1])
        edq_w8a8_test(m, N, K, recordListAll[2], "V")
        edq_w8a8_test(m, N, K, recordListAll[3], "M")
        awq_w8a8_test(m, N, K, recordListAll[4])
        fp16_test(m, N, K, recordListAll[5])

    # print(f"Marlin speed")
    # print(recordListAll[0])
    # print(f"AWQ speed")
    # print(recordListAll[0])
    # print(recordListAll[2])
    # print(recordListAll[3])

    plt.figure()
    # plt.plot(recordListAll[0], color = 'r')
    # plt.plot(recordListAll[1], color = 'g')
    plt.plot(recordListAll[2], color = 'r')
    plt.plot(recordListAll[3], color = 'g')
    plt.plot(recordListAll[4], color = 'b')
    plt.plot(recordListAll[4], color = 'k')
    plt.grid(True)
    # plt.legend(['Marlin w4a16', 'AWQ w4a16', 'Edq w8a8', 'AWQ w8a8'], loc='best')
    plt.legend(['Edq w8a8 GEMV', 'Edq w8a8 GEMM', 'AWQ w8a8', "FP16 GEMM"], loc='best')
    plt.title(f'W8A8 GEMV Speed Test (N={N}, K={K})')
    plt.savefig(f'./test/tune/w8a8_gemv_wquant_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')
    # plt.title(f'W8A8 GEMM Speed Test (N={N}, K={K})')
    # plt.savefig(f'./test/tune/w8a8_gemm_wquant_speed_test_N{N}_K{K}.png', dpi=300, bbox_inches='tight')
    # plt.close()

if __name__ == "__main__":
    # N, K
    # w_shape = [(18944, 3584), (3584, 18944), (3584, 3584), (512, 3584)]
    # w_shape = [(18944, 3584), (3584, 18944)]
    w_shape = [(3584, 3584), (512, 3584)]
    # w_shape = [(18944, 3584)]
    # w_shape = [(3584, 18944)]
    # w_shape = [(3584, 3584)]
    # w_shape = [(512, 3584)]
    # for s in w_shape:
    #     test(s[0], s[1])
    # plt.show()
    for s in w_shape[:]:
        # speed_test(s[0], s[1])
        w8a8_gemv_test(s[0], s[1])
        # w8a8_gemm_test(s[0], s[1])
        # w4a16_wa_gemv_test(s[0], s[1])
        # w4a16_wa_gemm_test(s[0], s[1])
        # w8a16_gemv_test(s[0], s[1])
        # w8a16_gemm_test(s[0], s[1])