import torch


data = torch.tensor(16384, dtype=torch.float16)
data = data.numpy().tobytes()
# 以二进制展示
b_str = '|'.join([f'{i:08b}' for i in data])
# 默认是小端模式，每个字节从右往左
print(b_str)


data = torch.tensor(256, dtype=torch.float16)
data = data.numpy().tobytes()
# 以二进制展示
b_str = '|'.join([f'{i:08b}' for i in data])
# 默认是小端模式，每个字节从右往左
print(b_str)
