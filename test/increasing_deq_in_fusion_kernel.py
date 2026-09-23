import math
import numpy
import matplotlib.pyplot as plt

def get_CTA_M(m):
    if (m > 128):
        return 128
    elif (m >= 64):
        return 64    
    return 32

alpha = 3
N = 4096
K = 4096
m_num = []
Compt = []
for M in range(1, 2048, 4):
    m_num.append(M)
    CTA_M = get_CTA_M(M)
    Compt.append(alpha*N*K*math.ceil(M/CTA_M))

# def showAll():
#     plt.plot(input_len, recordList[0], color = 'r')
#     plt.plot(input_len, recordList[1], color = 'g')
#     plt.plot(input_len, recordList[2], color = 'c')
#     plt.plot(input_len, recordList[3], color = 'm')
#     plt.plot(input_len, recordList[4], color = 'y')
#     plt.plot(input_len, recordList[5], color = 'k')
# def showByQuantConfig():
#     plt.title(f'K:{K}, N:{N}')
#     w4a16_lat = min(recordList[2], recordList[3])
#     w8a8_lat = min(recordList[4], recordList[5])
#     plt.plot(input_len, recordList[0], color = 'r', label = 'fp16')
#     plt.plot(input_len, recordList[1], color = 'g', label = 'w4a8')
#     plt.plot(input_len, w4a16_lat, color = 'y', label = 'w4a16')
#     plt.plot(input_len, w8a8_lat, color = 'k', label = 'w8a8')
#     plt.legend()

plt.figure()
plt.xlabel('m')
plt.ylabel('computation')
plt.plot(m_num, Compt)

# plt.xticks(input_len)
plt.grid(True)
plt.show()