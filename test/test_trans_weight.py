from edq.quantization import *
from edq.w4a16_linear import W4A16Linear, _perm
from thirdparty.AWQ.w4a16_linear import W4A16Linear_AWQ
from thirdparty.AWQ.awq_method import *
from edq.trans_weight import permute_weight
from utils.perf_eval import timer
from edq.dynamic_linear import edq_quantize_tensor_int4
import edq_cuda_accel

torch.manual_seed(seed=11)

def print_num(num):
    l =[]
    for _ in range(8):
        l.append(num&15)
        num = num >> 4
    for i in range(8):
        print(l[7-i], end=',')
    print('')

# @torch.compile()
def permute_by_lut(w, lut):
    w_shape = w.shape
    w_flat = w.view(-1)
    return w_flat[lut]

def test(K, N):
    print(f'K: {K} N: {N}')
    el_list = torch.tensor([i for i in range(K*N//2)], device='cuda')
    perm_list = permute_weight(el_list.reshape(N, K//2)).reshape(-1)
    # output = torch.ones((N,K), device='cuda', dtype=torch.float16)
    
    torchLinear= torch.nn.Linear(in_features=K, out_features=N, 
                                 bias=False, dtype=torch.float16).cuda()
    
    # ori_weight = torchLinear.weight.data.clone()
    ori_weight = torch.rand((N,K), dtype = torch.float16, device='cuda')
    # test_row = 1
    # print(ori_weight[test_row][128:160])

    # qw, qs, qz = real_quantize_tensor(ori_weight.clone())
    # print(qw, qs,'\n', qz, '\n', '-'*30)

    qw, q_config = edq_quantize_tensor_int4(ori_weight.clone(), group_size=128)
    qs = q_config['scale_f'].reshape(N, -1)
    qz = q_config['zero_pt_f'].reshape(N, -1) * 10
    # print(qw, qs,'\n', qz, '\n', '-'*30)
    dqw = dequantize_tensor_awq(qw, qs, qz)
    # print(dqw[test_row][128:160])

    # qlinear_w4a16_edq = W4A16Linear.from_module(torchLinear, 'cuda')
    # ori_weight = qlinear_w4a16_edq.qweight.clone()
    # qw, qs, qzp = real_quantize_tensor_edq(torchLinear.weight.data.clone().to(torch.float16), 4, 128)
    # qw_edq_t = pack_int4_data(qw).T.contiguous()
    # new_qw  = permute_weight(qw_edq_t).view(torch.int32)
    # print_num(ori_weight[0][0].item())
    # print_num(new_qw[0][0].item())

    qlinear_w4a16_awq = W4A16Linear_AWQ.from_module(torchLinear, 'cuda')
    # q_w = qlinear_w4a16_awq.qweight
    # q_s = qlinear_w4a16_awq.scales[0:K//128, :].T.contiguous()
    # q_z = qlinear_w4a16_awq.scaled_zeros[0:K//128, :].T.contiguous()
    # q_s = qlinear_w4a16_awq.scales[0:K//128, :]
    # q_z = qlinear_w4a16_awq.scaled_zeros[0:K//128, :]
    q_w = pack_int_awq(qw)
    q_s = qs.T.contiguous()
    q_z = -(qs*qz).T.contiguous()
    # q_s = qs
    # q_z = -(qs*qz)
    output = torch.zeros((N,K), device='cuda', dtype=torch.float16)
    edq_cuda_accel.dequant_interleaved_int4_to_fp16(output, q_w, q_s, q_z, 128)
    # output = output.reshape(N, K//32, 4,4,2).permute(0, 1, 3, 2, 4).reshape(N,K)
    # print(output[test_row][128:160])
    # print('-' * 30)
    diff = (dqw-output).abs()

    
    # print(diff[test_row][128:160])
    assert (diff > 0.01).sum() == 0, print(ori_weight,'\n', dqw,'\n', output,'\n', diff.max())
    print(diff.max())

    # qw, qs, qzp = real_quantize_tensor_edq(torchLinear.weight.data.clone().to(torch.float16), 4, 128)
    # qw_edq_t = pack_int4_data(qw).T.contiguous()
    # new_qw  = permute_weight(qw_edq_t).view(torch.int32)

    # o_w = torchLinear.weight.data.reshape(N, K//32, 4,4,2).permute(0, 1, 3, 2, 4).reshape(N,K)
    # print(o_w)
    # assert (ori_weight - new_qw).sum() == 0
    for _ in range(20):
        # new_qw = permute_weight(qw_edq_t)
        # permute_by_lut(qw_edq_t, perm_list)
        output = torch.ones((N,K), device='cuda', dtype=torch.float16)
        edq_cuda_accel.dequant_interleaved_int4_to_fp16(output, qlinear_w4a16_awq.qweight, 
                            qlinear_w4a16_awq.scales, qlinear_w4a16_awq.scaled_zeros, 128)
        # output = output.reshape(N, K//32, 4,4,2).permute(0, 1, 3, 2, 4).reshape(N,K)
    n = 100
    output = torch.ones((N,K), device='cuda', dtype=torch.float16)
    with timer("kernel", n = n):
        for _ in range(n):
            edq_cuda_accel.dequant_interleaved_int4_to_fp16(output, qlinear_w4a16_awq.qweight, 
                                qlinear_w4a16_awq.scales, qlinear_w4a16_awq.scaled_zeros, 128)
            
    # with timer("permute", n = n):
    #     for _ in range(n):
    #         output = output.reshape(N, K//32, 4,4,2).permute(0, 1, 3, 2, 4).reshape(N,K)
            
    # with timer("k  +  p", n = n):
    #     for _ in range(n):
    #         edq_cuda_accel.dequant_interleaved_int4_to_fp16(output, qlinear_w4a16_awq.qweight, 
    #                             qlinear_w4a16_awq.scales, qlinear_w4a16_awq.scaled_zeros, 128)
    #         output = output.reshape(N, K//32, 4,4,2).permute(0, 1, 3, 2, 4).reshape(N,K)
    print('-'*20)

if __name__ == '__main__':
    w_shape = [(3584, 512), (3584, 3584), (3584, 18944), (18944, 3584)]
    # w_shape = [(18944, 3584)]
    for s in w_shape:
        test(s[0], s[1])

# dequant_interleaved_int4_to_fp16