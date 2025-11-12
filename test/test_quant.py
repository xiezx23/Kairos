import sys
# sys.path.append('/home/xiezx/workspace/llm_EDQ/llm-Q4E-demo/llm_edq')

import torch
import time
from utils.color_print import color_text
from edq.quantization import *
import edq_cuda_accel
import awq_backend
from utils.perf_eval import timer


@torch.compile
def Q_GEMM_DQ(f_x, i_w, scale_w):# -> Any:
    i_x, config_x = quantize_tensor_int8(f_x)
    if f_x.shape[0] <= 16:
        z = torch.zeros(32-f_x.shape[0], f_x.shape[1], dtype=torch.int8, device=f_x.device)
        i_x = torch.cat([i_x, z], dim=0)
        o = torch._int_mm(i_x, i_w)
        o = o[0:f_x.shape[0], :]
    else:
        o = torch._int_mm(i_x, i_w)
    o = post_gemm_dequant(o, i_x, i_w, config_x, scale_w)
    return o

@torch.compile
def DQ_GEMM(f_x, i_w, scale_w):
    f_w = dequantize_tensor(i_w, scale_w)
    o = torch.mm(f_x, f_w)
    return o

def testGEMM(n, c, m):
    dev = 'cuda'
    # dev = 'cpu'
    f_x = torch.rand((n, c), dtype=torch.bfloat16, device=dev)
    f_x = f_x * 10
    f_w = torch.rand((m, c), dtype=torch.bfloat16, device=dev)
    f_x = f_x * 10
    i_w, congfig_w = quantize_tensor(f_w)
    f_w = f_w.T
    i_w = i_w.T
    print('------------------------------------------------------')
    print('        X(', f_x.shape[0], f_x.shape[1],') W(',f_w.shape[0], f_w.shape[1] ,')')
    ans_dict = {}
    for i in range(4):
        ans_dict[i] = 0
    for _ in range(3):
        torch.mm(f_x, f_w)
        a = time.time()
        for _ in range(32):
            o = torch.mm(f_x, f_w)
        torch.cuda.synchronize()
        b = time.time()
        ans_dict[0] += (b-a)
        # print(o.max())
        i_x, config_x = quantize_tensor(f_x)
        if f_x.shape[0] <= 16:
            z = torch.zeros(32-f_x.shape[0], f_x.shape[1], dtype=torch.int8, device=dev)
            i_x = torch.cat([i_x, z], dim=0)
        o = torch._int_mm(i_x, i_w)
        a = time.time()
        for _ in range(32):
            o = torch._int_mm(i_x, i_w)
            # o = o[0:f_x.shape[0], :]
        torch.cuda.synchronize()
        b = time.time()
        # print(o.max())
        ans_dict[1] += (b-a)

        DQ_GEMM(f_x, i_w, congfig_w['scale'])
        a = time.time()
        for _ in range(32):
            o = DQ_GEMM(f_x, i_w, congfig_w['scale'])
        torch.cuda.synchronize()
        b = time.time()
        # print(o.max())
        ans_dict[2] += (b-a)

        Q_GEMM_DQ(f_x, i_w, congfig_w['scale'])
        a = time.time()
        for _ in range(32):
            o = Q_GEMM_DQ(f_x, i_w, congfig_w['scale'])
        torch.cuda.synchronize()
        b = time.time()
        # print(o.max())
        ans_dict[3] += (b-a)
    print("exeTime: FLOAT16GEMM:              {}{:6.2f}{} ms".format(color_text('gre', ans_dict[0]/3*1000)))
    print("exeTime: INT8GEMM:                 {}{:6.2f}{} ms".format(color_text('gre', ans_dict[1]/3*1000)))
    print("exeTime: DQ(W) + FLOAT16GEMM:      {}{:6.2f}{} ms".format(color_text('gre', ans_dict[2]/3*1000)))
    print("exeTime: Q(A) + INT8GEMM + DQ(Y):  {}{:6.2f}{} ms".format(color_text('gre', ans_dict[3]/3*1000)))
    print('------------------------------------------------------')

def testPerformanceModel(n, k, m):
    P = 156e12
    B = 1555e9

    f_x = torch.rand((n, k), dtype=torch.float16, device='cuda') + 0.3
    f_w = torch.rand((m, k), dtype=torch.float16, device='cuda') + 0.2
    f_w = f_w.T
    torch.cuda.empty_cache()
    for _ in range(10):
        y = torch.mm(f_x, f_w)
    # # 获取cuBLAS选择的算法
    # with torch.backends.cudnn.flags(enabled=False):  # 禁用cuDNN干扰
    #     flag = torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
    #     if flag:
    #         P *= 2
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y = torch.mm(f_x, f_w)
    end.record() 
    torch.cuda.synchronize()
    x = y.amax()
    print("------------------------")
    print('X(', f_x.shape[0], f_x.shape[1],') W(',f_w.shape[0], f_w.shape[1] ,')')
    print("real exe time: {:5.3f} ms".format(start.elapsed_time(end)))

    AI = m*k*n/(m*k+n*k+n*m)
    esti_time = 2*m*n*k / min(AI * B, P)
    print("esti exe time: {:5.3f} ms".format(esti_time*1000 ))
    print("esti cau time: {:5.3f} ms".format(2*m*k*n / P*1000))
    print("esti mem time: {:5.3f} ms".format(2*m*k*n / (AI * B)*1000))
    print("------------------------")
 
def test_module(N, K, M):
    print("------------------------")
    print('X(',N,K,') W(',K,M,')')
    x = torch.randn(1, N, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

    ori_out = x @ w.T
    with timer('ORI  FP16'):
        ori_out = x @ w.T

    int_w = torch.empty_like(w, dtype=torch.int8, device='cuda')
    scale_w = torch.empty(w.shape[0], device='cuda', dtype=torch.float16)
    edq_cuda_accel.quant_fp16_to_int8(int_w, w.to(torch.float16), scale_w)
    
    def W8A8_forward(x):
        mx = x.reshape(-1,x.shape[2]).to(torch.float16)
        int_x = torch.empty_like(mx, dtype=torch.int8, device='cuda')
        out1 = torch.empty((mx.shape[0], w.shape[0]), device='cuda', dtype=torch.float16)
        scale_x = torch.empty(mx.shape[0], device='cuda', dtype=torch.float16)
        edq_cuda_accel.quant_fp16_to_int8(int_x, mx, scale_x)
        # torch.cuda.synchronize()
        awq_backend.w8a8_gemm_forward_cuda(int_x, int_w, scale_w, scale_x, out1)
        out1 = out1.to(torch.bfloat16)
    W8A8_forward(x)
    with timer('W8A8 GEMM'):
        W8A8_forward(x)

    q_w, q_scales, q_scaled_zeros = quant_weight_awq(w.to(torch.float16), 128)

    def W4A16_forward(x):
        out2 = awq_backend.gemm_forward_cuda_new(
            x.to(torch.float16), q_w, q_scales, q_scaled_zeros)
        out2 = out2.to(torch.bfloat16)
    W4A16_forward(x)
    with timer('AWQ W4A16'):
        W4A16_forward(x)

def test(n):
    w_dim = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    for w_dim_p in w_dim:
        test_module(n, w_dim_p[0], w_dim_p[1])
        # testGEMM(n, w_dim_p[0], w_dim_p[1])
        # testPerformanceModel(n, w_dim_p[0], w_dim_p[1])

if __name__ == '__main__':
    torch.manual_seed(10)
    for n in [1, 32, 60, 64, 128, 256]:
        test(n)



    # f_x = torch.rand((60, 3584), dtype=torch.float16, device='cuda')
    # f_w = torch.rand((512, 3584), dtype=torch.float16, device='cuda')
    
    # f_wt = f_w.T
    # ori_out = f_x@f_wt
    # with timer('ori matmul'):
    #     ori_out = f_x@f_wt

    # int_w, config_w = quantize_tensor_int8(f_w)
    # int_x, config_x = quantize_tensor_int8(f_x)
    # out = torch._int_mm(int_x, int_w.T)
    # out = post_gemm_dequant(out, int_x, int_w.T, config_x['scale'], config_w['scale'])
    # loss_x = (out - ori_out).pow(2).mean().sqrt().item()
    # print(loss_x)

    # i_w = torch.empty_like(f_w, dtype=torch.int8, device='cuda')
    # scale_w = torch.empty(f_w.shape[0], device='cuda', dtype=torch.float16)
    # edq_cuda_accel.quant_fp16_to_int8(i_w, f_w, scale_w)

    # out1 = torch.empty((f_x.shape[0], f_w.shape[0]), device='cuda', dtype=torch.float16)
    # w_x1 = torch.empty_like(f_x, dtype=torch.int8, device='cuda')
    # scale1 = torch.empty(f_x.shape[0], device='cuda', dtype=torch.float16)
    # edq_cuda_accel.quant_fp16_to_int8(w_x1, f_x, scale1)
    # # torch.cuda.synchronize()
    # awq_backend.w8a8_gemm_forward_cuda(int_x, int_w, config_w['scale'], config_x['scale'], out1)
    # with timer('W8A8 GEMM'):
    #     edq_cuda_accel.quant_fp16_to_int8(w_x1, f_x, scale1)
    #     # torch.cuda.synchronize()
    #     awq_backend.w8a8_gemm_forward_cuda(int_x, int_w, config_w['scale'], config_x['scale'], out1)
    # loss_x = (ori_out - out1).pow(2).mean().sqrt().item()
    # print(loss_x)
    
    # edq_cuda_accel.w8a16_gemm_forward_cuda(f_x, int_w, config_w['scale'].to(torch.float16), out1)
    # with timer('W8A16 GEMM'):
    #     edq_cuda_accel.w8a16_gemm_forward_cuda(f_x, int_w, config_w['scale'].to(torch.float16), out1)
    # loss_x = (ori_out - out1).pow(2).mean().sqrt().item()
    # print(loss_x)
    
    # f_w_new = int_w.to(torch.float16) * config_w['scale']
    # out1 = f_x@f_w_new.T
    # with timer('DQ+ GEMM'):
    #     f_w_new = int_w.to(torch.float16) * config_w['scale']
    #     out1 = f_x@f_w_new.T
    # loss_x = (ori_out - out1).pow(2).mean().sqrt().item()
    # print(loss_x)
    
    # o = torch._int_mm(w_x1, i_w.T)
    # with timer('pytorch quant'):
    #     o = torch._int_mm(w_x1, i_w.T)
    
    # loss_x = (out - out1).pow(2).to(torch.float16).mean().sqrt().item()
    # print(loss_x)
    # loss_x = (out - out2).pow(2).to(torch.float16).mean().sqrt().item()
    # print(loss_x)

    # print(f_x)
    # print(w_x1)
    # print(w_x2)

    # loss_x = (w_x1 - w_x2).pow(2).to(torch.float16).mean().sqrt().item()
    # loss_s = (scale1 - scale2).pow(2).mean().sqrt().item()
    # print(loss_x)
    # print(loss_s)