import torch
from utils.perf_eval import timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 预分配所有需要的张量
# x1 = torch.randn((64, 64), device=device)
# x2 = torch.randn((64, 64), device=device)

x1 = torch.randn((2048, 2048), device=device)
x2 = torch.randn((4096*2, 4096*2), device=device)

# 预分配输出张量
y1_result = torch.empty_like(x1)
y2_result = torch.empty_like(x2)

# 创建非阻塞流
stream1 = torch.cuda.Stream(device=device)
stream2 = torch.cuda.Stream(device=device)
event   = torch.cuda.Event(enable_timing=False)

@torch.no_grad()
def kernel1(output):
    # 使用原地操作减少内存分配
    # output.copy_(x1)
    # output[0][0].add_(10000)  # 使用原地操作
    # output.add_(1)
    output = x1 @ x1

@torch.no_grad()
def kernel2(output):
    # output[0][0].add_(10000)  # 使用原地操作
    output.add_(1)
    # output.cpu()
    # output = x2 @ x2

@torch.no_grad()
def sequential_execution():
    kernel1(y1_result)
    kernel2(y2_result)

@torch.no_grad()
def overlapped_execution():
    # 在非阻塞流中执行内核
    torch.cuda.synchronize()
    with torch.cuda.stream(stream1):
        kernel1(y1_result)
        event.record(stream1)
    kernel2(y2_result)
    torch.cuda.current_stream().wait_event(event)

if __name__ == '__main__':
    for _ in range(5):
        kernel1(y1_result)
        kernel2(y2_result)
        overlapped_execution()
        torch.cuda.synchronize()
        sequential_execution()
        
    with timer():
        sequential_execution()
    with timer():
        sequential_execution()
    with timer():
        sequential_execution()

    with timer():
        overlapped_execution()
    torch.cuda.synchronize()
    
    with timer():
        overlapped_execution()
    torch.cuda.synchronize()
    
    with timer():
        overlapped_execution()
    torch.cuda.synchronize()
    
exit(0)


import torch
from utils.perf_eval import timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# 使用更大的张量以便更好地利用GPU并行性
stream1 = torch.cuda.Stream(device=device)
stream2 = torch.cuda.Stream(device=device)
event1 = torch.cuda.Event(enable_timing=True)
event2 = torch.cuda.Event(enable_timing=True)

class test():
    def __init__(self):
        self.result = 1

    @torch.no_grad()
    def kernel1(self, x1):
        y = x1 @ x1
        self.result = y.amax()
        event1.record(stream1)

    @torch.no_grad()
    def kernel2(self, x2):
        x2 = x2 / 100 + 2

    @torch.no_grad()
    def kernel3(self, x2):
        x2 = x2 @ x2

    @torch.no_grad()
    def sequential_execution(self, x1, x2):
        self.kernel1(x1)
        self.kernel2(x2)
        # self.kernel3(x2)
        return x2.max()

    @torch.no_grad()
    def overlapped_execution(self, x1, x2):
        self.kernel1(x1)
        with torch.cuda.stream(stream2):
            self.kernel2(x2)
            event2.record(stream2)
        # event1.synchronize()
        # event2.synchronize()
        # kernel3(x2)
        return x2.max()
    
    def print_res(self):
        # with torch.cuda.stream(stream2):
        print(self.result.max())
    
    def test(self):
        with torch.cuda.stream(stream1):
            x1 = torch.randn((1024*16, 1024*16), device=device)
            self.kernel1(x1)
        # event1.synchronize()
        # exit(0)


if __name__ == '__main__':
    torch.manual_seed(10)

    t = test()
    res = 0
    with timer('test'):
        t.test()
    print(res)
    exit(0)

    # 预热
    for _ in range(1):
        sequential_execution(x1, x2)
        overlapped_execution(x1, x2)
        torch.cuda.synchronize()  # 确保时间记录准确
    
    for _ in range(1):
        # 清空缓存并计时
        torch.cuda.empty_cache()
        with timer('kernel1'):
            kernel1(x1)
        torch.cuda.empty_cache()
        with timer('kernel2'):
            kernel2(x2)
        torch.cuda.empty_cache()
        with timer('kernel3'):
            kernel3(x2)

        torch.cuda.empty_cache()
        with timer('Sequential execution'):
            y3max1 = sequential_execution(x1, x2)
        print(y3max1)
        
        torch.cuda.empty_cache()
        with timer('Overlapped execution'):
            y3max2 = overlapped_execution(x1, x2)
        print(y3max2)
        
        print('-' * 30)
exit(0)


import torch
from utils.perf_eval import timer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

stream1 = torch.cuda.Stream(device=device)
stream2 = torch.cuda.Stream(device=device)
event1 = torch.cuda.Event(enable_timing=False)
event2 = torch.cuda.Event(enable_timing=False)

@torch.no_grad
def kernel1():
    x1 = torch.randn((1024*2, 1024*2), device = device)
    y1 = x1
    for _ in range(10000):
        y1 += 1
    return y1

@torch.no_grad
def kernel2():
    x2 = torch.randn((1024*2, 1024*2), device = device)
    y2 = x2
    for _ in range(10000):
        y2 += 1
    return y2

@torch.no_grad
def overlapped_forward_with_event():
    with torch.cuda.stream(stream1):
        y1 = kernel1()
        event1.record()
    with torch.cuda.stream(stream2):
        y2 = kernel2()
        event2.record()
    event1.synchronize()
    event2.synchronize()
    y3 = y1
    return (y1, y2, y3)
 
@torch.no_grad
def ori_forward():
    y1 = kernel1()
    y2 = kernel2()
    y3 = y1
    return (y1, y2, y3)

@torch.no_grad
def overlapped_forward_with_default():
    with torch.cuda.stream(stream1):
        y1 = kernel1()
        event1.record()
    y2 = kernel2()
    event1.synchronize()
    y3 = y1
    return (y1, y2, y3)
 
 
if __name__ == '__main__':
    for _ in range(11):
        y1 = kernel1()
        y2 = kernel2()
        y1, y2, y3 = ori_forward()
        y1, y2, y3 = overlapped_forward_with_default()
        y1, y2, y3 = overlapped_forward_with_event()

    for _ in range(5):
        torch.cuda.empty_cache()
        with timer('kernel1'):
            y1 = kernel1()
        torch.cuda.empty_cache()
        with timer('kernel2'):
            y2 = kernel2()
        torch.cuda.empty_cache()
        with timer('w/o overlap'):
            y1, y2, y3 = ori_forward()
        # print(y1.max(), y2.max())
        torch.cuda.empty_cache()
        with timer('w/  overlap'):
            y1, y2, y3 = overlapped_forward_with_event()
        # print(y1.max(), y2.max())
        torch.cuda.empty_cache()
        with timer('w/  overlap'):
            y1, y2, y3 = overlapped_forward_with_default()
        # print(y1.max(), y2.max())
        print('-' * 30)

exit(0)

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
A = torch.rand(1000, 1000, device = 'cuda')
B = torch.rand(1000, 1000, device = 'cuda')
# Wait for the above tensors to initialise.
torch.cuda.synchronize()

for _ in range(10):
    C = torch.mm(A, A)
    E = C + A
    C = torch.mm(A, A)
    E = C + E
    D = torch.mm(B, B)
    F = D + B
    D = torch.mm(B, B)
    F = D + F
    
torch.cuda.synchronize()
with timer():
    C = torch.mm(A, A)
    E = C + A
    C = torch.mm(A, A)
    E = C + E
    D = torch.mm(B, B)
    F = D + B
    D = torch.mm(B, B)
    F = D + F
    torch.cuda.synchronize()
print(E.max())
print(F.max())
print('-' * 30)

torch.cuda.synchronize()
with timer():
    with torch.cuda.stream(s1):
        C = torch.mm(A, A)
        E = C + A
        C = torch.mm(A, A)
        E = C + E
    with torch.cuda.stream(s2):
        D = torch.mm(B, B)
        F = D + B
        D = torch.mm(B, B)
        F = D + F
    torch.cuda.synchronize()
print(E.max())
print(F.max())
# Wait for C and D to be computed



exit(0)

stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

M = 128
K = 128
N = 128

a1 = torch.randn((M, K * 6 * 8), device = 'cuda')
b1 = torch.randn((N, K * 6 * 8), device = 'cuda')
a2 = torch.randn((M * 8, K), device = 'cuda')
b2 = torch.randn((N * 8, K), device = 'cuda')

def kernel(a, b):
    c = a @ b.T
    return c

kernel_graphed1 = torch.cuda.make_graphed_callables(kernel, (a1, b1))
kernel_graphed2 = torch.cuda.make_graphed_callables(kernel, (a2, b2))

for _ in range(5):
    kernel(a1, b1)
    kernel(a2, b2)
    with torch.cuda.stream(stream1):
        kernel(a1, b1)
    with torch.cuda.stream(stream2):
        kernel(a2, b2)
    
g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    kernel(a1, b1)
    kernel(a2, b2)

# g2 = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g2):
#     with torch.cuda.stream(stream1):
#         kernel(a1, b1)
#     kernel(a2, b2)

# g3 = torch.cuda.CUDAGraph()
# with torch.cuda.graph(g3):
#     with torch.cuda.stream(stream1):
#         kernel(a1, b1)
#     with torch.cuda.stream(stream2):
#         kernel(a2, b2)

for _ in range(5):

    with timer('kernel1'):
        kernel(a1, b1)

    with timer('kernel2'):
        kernel(a2, b2)

    with timer('CUDAGR1'):
        g1.replay()

    # with timer('CUDAGR2'):
    #     g2.replay()

    with timer('CUDAGR3'):
        kernel_graphed1(a1,b1)
        kernel_graphed2(a2,b2)

    with timer('1 + 2  '):
        kernel(a1, b1)
        kernel(a2, b2)
    print('-'*30)

