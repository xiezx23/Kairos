import torch

N=4096; K=4096

stream = torch.cuda.Stream()

weight_tensor = torch.randn((N, K), dtype=torch.float16, device = 'cpu')
t1 = torch.randn((N, K), dtype=torch.float16, device = 'cpu')
with torch.cuda.stream(stream):
    weight_tensor.cuda()
t1.add(1)
t2 = weight_tensor+t1
print(t2.sum())