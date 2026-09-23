import torch
import numpy
import tqdm
import random
import matplotlib.pyplot as plt

from utils.color_print import *
from utils.util import save_json, load_json
from utils.perf_eval import timer
from kairos.quantization import *
from kairos.dynamic_linear_v2 import DynamicLinear, COMP_TYPE_NAME

# COMP_TYPE_NAME = {'w4a16_gemm':0, 'w4a16_gemv':1, 'w8a8':2, \
#                   'w8a16_gemm':3, 'w8a16_gemv':4, 'fp16':5}     # W8A16 strategy is never used

torch.manual_seed(seed=10)
# print_flag = True
print_flag = False

try:
    # raise(ImportError)
    import LUT_module
    # Use cpp version
    LUT = LUT_module.LUT
except(ImportError):
    # Use python version
    class LUT:
        @torch.no_grad
        def __init__(self) -> None:
            # When m in (seg_list[i], seg_list[i+1]], the best is best_comp[i]; 
            # If m > seg_list[-1], the best is best_comp[-1] 
            self.seg_list = [0]
            self.best_comp = []
            self.pre_ask = 0; self.pre_res = 0

        def get(self, m):
            if m == self.pre_ask: return self.pre_res
            self.pre_ask = m
            l = 0; r = len(self.seg_list)-1
            if m > self.seg_list[r]: 
                self.pre_res = self.best_comp[-1]
            else:
                while r-l > 1:
                    mid = (l+r) >> 1
                    if self.seg_list[mid] < m: 
                        l = mid
                    else:        
                        r = mid
                self.pre_res = self.best_comp[l]
            return self.pre_res

        def record(self, r, comp_type_id):
            self.seg_list.append(r)
            self.best_comp.append(comp_type_id)

class Profiler:
    @torch.no_grad
    def __init__(self, shape_list) -> None:
        self.shape_list = shape_list
        self.seg_perf_list = []
        self.lut = {shape:LUT() for shape in self.shape_list}

    def record_best_comp(self, shape, r, comp_type):
        self.lut[shape].record(r, COMP_TYPE_NAME[comp_type])
    
    def profiling(self):
        for shape in self.shape_list:
            seg_perf_list = test_perf(str(shape), shape[1], shape[0], False)
            for t in seg_perf_list:
                self.record_best_comp(shape, t[0], t[1])
            self.seg_perf_list.append(seg_perf_list)

    def get_comp_type(self, shape, m):
        return self.lut[shape].get(m)

    def save(self, path = './prof/'):
        save_json(path+'seg_perf_list.pt', self.seg_perf_list)
    
    def load(self, path = './prof/'):
        self.seg_perf_list = load_json(path+'seg_perf_list.pt')
        for i in range(len(self.shape_list)):
            shape = self.shape_list[i]
            for t in self.seg_perf_list[i]:
                self.record_best_comp(tuple(shape), t[0], t[1])
 
@torch.no_grad
def test_perf(proj_name, K, N, bias):
    torchLinear = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    # Input Dimension M Sampling
    input_len = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 256, 512] + \
                [i for i in range(1024, 1024*8, 1024)] +\
                [i for i in range(1024*8, 1024*32, 2048)] 
    # input_len = \
    #     [i for i in range(1, 16, 1)] + \
    #     [i for i in range(16, 2048, 32)] + \
    #     [i for i in range(2048, 1024*16, 2048)] 

    n = 300 # Calculate the average performance of n executions.
    type_name_list = ['GEMM  FP16', 'GEMM W4A16', 'GEMV W4A16', 'GEMM W4A16', 'GEMM  W8A8', 'GEMM  W4A8']
    comp_type_list = ['fp16', 'w4a16_gemm', 'w4a16_gemv', 'w4a16_marlin', 'w8a8', 'w4a8_qserve']

    # Preheat Kernels
    preheat_time = 100
    # for m in input_len:
    #     input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
    #     for comp_type in comp_type_list:
    #         if m < 5 and 'gemm' in comp_type: continue
    #         elif m > 32 and 'gemv' in comp_type: continue
    #         for _ in range(preheat_time):
    #             out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])

    # Determine the default kernel
    B = [i for i in range(1, 9)]
    recordList_1 = [[] for _ in range(len(comp_type_list))]
    for midx in tqdm.tqdm(range(len(B)), desc=f"Determine Default Kernel for {proj_name}"):
        m  = B[midx]
        if print_flag:
            print(f'Sampled batch M: {m} K: {K} N: {N} Bias:{bias}')
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for i in range(len(comp_type_list)):
            comp_type = comp_type_list[i]
            if m < 5 and 'gemm' in comp_type: 
                recordList_1[i].append(1e5)
                continue
            # elif m < 4 and 'marlin' in comp_type: continue
            elif m > 32 and 'gemv' in comp_type: 
                recordList_1[i].append(1e5)
                continue
            torch.cuda.empty_cache()
            for _ in range(preheat_time): # Preheat Kernel
                out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])
            with timer(type_name_list[i], n=n, recordList=recordList_1[i], print_flag=print_flag):
                for _ in range(n):
                    out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])
    for i in range(len(recordList_1[0])):
        recordList_1[1][i] = min(recordList_1[1][i], recordList_1[2][i])
    awq_sum_latency = sum(recordList_1[1])
    mar_sum_latency = sum(recordList_1[3])
    if awq_sum_latency <= mar_sum_latency:
        default = 'awq_packing'
        skip_comp = [3]
    else: 
        default = 'mar_packing'
        skip_comp = [1, 2, 5]
    print(f"Default Kernel for ({N}, {K}) : {default}")


    recordList = [[] for _ in range(len(comp_type_list))]
    perf_list = []  # record the best comp_type for each m.
    for midx in tqdm.tqdm(range(len(input_len)), desc=f"Test Perf. for {proj_name}"):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N} Bias:{bias}')
        input_tensor = torch.randn((1, m, K), dtype=torch.float16, device = 'cuda')
        # TEST DYNAMIC LINEAR
        best_t = 1e10; best_comp = ''
        orig_t = 1e10; orig_comp = ''
        if print_flag: print('-' * 30)
        for i in range(len(comp_type_list)):
            if i in skip_comp: continue
            comp_type = comp_type_list[i]
            if m < 5 and 'gemm' in comp_type: continue
            # elif m < 4 and 'marlin' in comp_type: continue
            elif m > 32 and 'gemv' in comp_type: continue
            torch.cuda.empty_cache()
            for _ in range(preheat_time): # Preheat Kernel
                out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])
            with timer(type_name_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
                for _ in range(n):
                    out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])
            if best_t > recordList[i][-1]:
                best_t = recordList[i][-1]
                best_comp = comp_type
            if 'w4' in comp_type:
                if orig_t > recordList[i][-1]:
                    orig_t = recordList[i][-1]
                    orig_comp = comp_type
        # Check new comp is at least 5% better than orig comp.
        # if best_t * 1.05 > orig_t:
        #     best_comp = orig_comp
        perf_list.append((m, best_comp, (orig_t - best_t)*100/orig_t))
        if print_flag:
            # fastest_kid = numpy.argmin(recordList)
            # print(color_text('gre', 'BestKernel: '+kernel_name_list[fastest_kid]))
            # print(gre_prefix, end='')
            # print(default_color, end='')
            print('=' * 30)
    seg_perf_list = []
    pre_tp = perf_list[0]
    for t in perf_list:
        if t[1] != pre_tp[1]:
            seg_perf_list.append(pre_tp)
        pre_tp = t
    seg_perf_list.append(pre_tp)
    # for t in perf_list:
    #     print(f'm = {t[0]:5d} : {t[1]}')
    # print('-' * 30)
    print('-' * 7+f'N:{N:5d}  K:{K:5d}'+'-' * 7)
    for t in seg_perf_list:
        print(f'm: {t[0]:5d}   best: {t[1]:12s}  speed↑: {t[2]:4.2f}')
    print('-' * 30)
    if print_flag: print('#' * 30)
    return seg_perf_list
    
if __name__ == '__main__':
    prof = Profiler([(512, 3584), (3584, 3584), (3584, 18944), (18944, 3584)])
    prof.profiling()
    # prof.save()
    # prof.load()
    l = 100
    m_list = [random.randint(1, 1024*6) for _ in range(l)]
    shape_list = prof.shape_list
    """ Preheat Region """
    for shape in shape_list:
        for m in m_list:
            prof.get_comp_type(shape, m)
    """  Test  Region  """
    for shape in shape_list:
        with timer(f'{str(shape):14s}', n=l):
            for m in m_list:
                prof.get_comp_type(shape, m)
    # while 1:
    #     m = int(input())
    #     print(f'the best comp type is {prof.get_comp_type((512, 3584), m)}')
    # w_shape_proj = ['k_proj, v_proj', 'q_proj',
    #             'o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(3584, 512, True), (3584, 3584, True),
    #            (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    # w_shape_proj = ['k_proj, v_proj', 'q_proj, o_proj', 'gate_proj, up_proj', 'down_proj']
    # w_shape = [(3584, 512, False), (3584, 3584, False), (3584, 18944, False), (18944, 3584, False)]
    # for i in range(len(w_shape)):
    #     if print_flag: print(w_shape_proj[i])
    #     s = w_shape[i]
    #     test(w_shape_proj[i], s[0], s[1], s[2])
    #     print('#' * 30)
        # exit(0)