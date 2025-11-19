import torch
import numpy
import tqdm
import random
import matplotlib.pyplot as plt

from utils.color_print import *
from utils.util import save_json, load_json
from utils.perf_eval import timer
from kairos.quantization import *
from kairos.dynamic_linear import DynamicLinear, COMP_TYPE_NAME

torch.manual_seed(seed=10)
print_flag = True
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
    # M = 1024 + 1
    torchLinear   = torch.nn.Linear(in_features=K, out_features=N, bias=bias, dtype=torch.float16).cuda()
    dlinear  = DynamicLinear.from_module(torchLinear, 'cuda')
    input_len = [i for i in range(1, 16, 1)] + [i for i in range(16, 2048, 32)] + [i for i in range(2048, 1024*16, 2048)] 
    # input_len = [128, 512, 1024, 2048, 4096, 1024*8, 1024*16]
    # input_len = [1024*8, 1024*16, 1024*32]
    # input_len = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16*1024, 32*1024]
    # A = 32; B = 128; C = 1024; D = 4*1024
    # input_len = \
    #     [i for i in range(1, A, 1)] + \
    #     [i for i in range(A, B, 2)] + \
    #     [i for i in range(B, C, 16)] + \
    #     [i for i in range(C, D, 64)]
    # A = 1024; B = 4096; C = 16*1024; D = 48*1024; E = 128*1024
    # input_len = \
    #     [i for i in range(1, A, 128)] + \
    #     [i for i in range(A, B, 256)] + \
    #     [i for i in range(B, C, 512)] + \
    #     [i for i in range(C, D, 1024)] + \
    #     [i for i in range(D, E, 2024)]

    n = 100
    DL_type_list =  ['Dlinear  FP16', 'Dlinear W4A16', 'Dlinear W4A16', 'Dlinear  W8A8']
    comp_type_list = ['fp16', 'w4a16_gemm', 'w4a16_gemv', 'w8a8']

    # Preheat Kernels
    preheat_time = 20
    for m in [1, 16, 1024, 2048, 4096]:
        # Create input matrices
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        for comp_type in comp_type_list:
            if m > 32 and 'gemv' in comp_type: continue
            for _ in range(preheat_time):
                out = dlinear._forward(input_tensor, m, COMP_TYPE_NAME[comp_type])
            

    recordList = [[] for _ in range(len(comp_type_list))]
    perf_list = []  # record the best comp_type of m.
    for midx in tqdm.tqdm(range(len(input_len)), desc=f"Test Perf. for {proj_name}"):
        m = input_len[midx]
        if print_flag:
            print('input M: ', m)
            print(f'K: {K} N: {N} Bias:{bias}')
        input_tensor = torch.randn((m, K), dtype=torch.float16, device = 'cuda')
        # TEST DYNAMIC LINEAR
        best_t = 1e10; best_comp = ''
        orig_t = 1e10; orig_comp = ''
        if print_flag: print('-' * 30)
        for i in range(len(comp_type_list)):
            comp_type = comp_type_list[i]
            if m > 32 and 'gemv' in comp_type: continue
            torch.cuda.empty_cache()
            with timer(DL_type_list[i], n=n, recordList=recordList[i], print_flag=print_flag):
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
    plt.figure()
    # =============    ===============================
    # character        color
    # =============    ===============================
    # ``'b'``          blue
    # ``'g'``          green
    # ``'r'``          red
    # ``'c'``          cyan
    # ``'m'``          magenta
    # ``'y'``          yellow
    # ``'k'``          black
    # ``'w'``          white
    # =============    ===============================
    def showAll():
        plt.plot(input_len, recordList[0], color = 'r')
        plt.plot(input_len, recordList[1], color = 'g')
        plt.plot(input_len, recordList[2], color = 'c')
        plt.plot(input_len, recordList[3], color = 'm')
        # plt.plot(input_len, recordList[4], color = 'y')
        # plt.plot(input_len, recordList[5], color = 'k')
    def showByQuantConfig():
        plt.title(f'K:{K}, N:{N}')
        w4a16_lat = min(recordList[2], recordList[3])
        w8a8_lat = min(recordList[4], recordList[5])
        plt.plot(input_len, recordList[0], color = 'r', label = 'fp16')
        plt.plot(input_len, recordList[1], color = 'g', label = 'w4a8')
        plt.plot(input_len, w4a16_lat, color = 'y', label = 'w4a16')
        plt.plot(input_len, w8a8_lat, color = 'k', label = 'w8a8')
        plt.legend()
    showAll()
    plt.xlabel('m')
    plt.ylabel('latency')
    # plt.xticks(input_len)
    plt.grid(True)
    
if __name__ == '__main__':

    # for m in [i for i in range(1, 1000, 100)]:
    #     r = m
    #     ref_lut.record(r, m)
    #     cpp_lut.record(r, m)
    # l = 20
    # m_list = [random.randint(1, 1024*6) for _ in range(l)]
    # for m in m_list:
    #     c1 = ref_lut.get(m)
    #     c2 = cpp_lut.get(m)
    #     assert c1 == c2, print(c1, c2)

    # with timer('PY__LUT', n=l):
    #     for m in m_list:
    #         ref_lut.get(m)
    # with timer('CPP_LUT', n=l):
    #     for m in m_list:
    #         cpp_lut.get(m)

    prof = Profiler()
    prof.profiling()
    prof.save()
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