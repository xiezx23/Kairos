import torch

mask = 0b00001111

@torch.no_grad()
# @torch.compile()
def trans_int32(t):
    shape = t.shape
    t = t.reshape(-1, 4)
    la = t[:,0] & mask
    hb = (t[:,1] >> 4) & mask
    t[:,0] = (t[:,0] & ~mask) | hb
    t[:,1] = (t[:,1] & mask) | (la << 4)
    la = t[:,2] & mask
    hb = (t[:,3] >> 4)&mask
    t[:,2] = (t[:,2] & ~mask) | hb
    t[:,3] = (t[:,3] & mask) | (la << 4)
    t[:, [1, 2]] = t[:, [2, 1]]
    # NOTE: view(int32) to 4int8(a,b,c,d) => (dcba)
    # t[:, [0, 3]] = t[:, [3, 0]]
    return t.reshape(shape)
    # return t.reshape(shape).view(torch.int32)

def print_num(num):
    l =[]
    for _ in range(8):
        l.append(num&mask)
        num = num >> 4
    for i in range(8):
        print(l[7-i], end=' ')
    print('')

# [(1, 2), (3, 4)] => [(1, 3), (2, 4)]
# t= torch.tensor([(0<<4)+1, (2<<4)+3, (4<<4)+5, (6<<4)+7, 
#                  (1<<4)+2, (3<<4)+4, (5<<4)+6, (7<<4)+8], dtype=torch.int8)
# 10,9,15,6,5,1,9,0,
t= torch.tensor([(10<<4)+9, (15<<4)+6, (5<<4)+1, (9<<4)+0], dtype=torch.uint8)
t = t.view(dtype=torch.int8)

for i in range(4):
    print((t[i].item() >> 4)& 0b00001111, end=',')
    print(t[i].item() & 0b00001111, end=',')
print('')

t = trans_int32(t)

for i in range(4):
    print((t[i].item() >> 4)& 0b00001111, end=',')
    print(t[i].item() & 0b00001111, end=',')
print('')
# print_num(t[0].item())