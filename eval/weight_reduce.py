
import torch
import numpy
import matplotlib.pyplot as plt

torch.manual_seed(seed=10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

method = ['W8A8', 'W4A16', 'W4A8', 'Kairos']
result = [8.13, 5.35, 5.19, 5.56]
full = 14.23
for i in range(len(result)):
    result[i] = (full-result[i])/full    

# plt.xlabel('Input dimension M')
plt.ylabel('Weight compression ratio')
# plt.title(f'LLM linear')
# plt.grid(True)
# plt.plot(input_len, sum_lat_list[0], color = 'y', label = 'FP16')
plt.bar(method, result, 0.5, color=['k', 'c', 'b', 'r'], alpha=0.5)
# plt.plot(method, result[0], '--ok')
# plt.plot(method, result[2], 's-c')
# plt.plot(method, result[3], '--ob')
# plt.plot(method, result[4], '^-r')
plt.show()
exit(0)