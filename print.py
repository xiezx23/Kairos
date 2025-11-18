
from re import I
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, axis
import matplotlib.ticker as ticker

import numpy as np
import math


# matplotlib.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(10.8, 4))

x = np.arange(3)*0.5
total_width, n = 0.4, 4
width = total_width / n
dist = 1.1


# "Bert-Large": {
#             "Recomputation (Memory)": 95.92,
#             "Recomputation (Latency)": 105.16,
#             "Operator Fission (Memory)": 91.30,
#             "Operator Fission (Latency)": 107.93
#         },
#         "Qwen-1.8B": {
#             "Recomputation (Memory)": 92.13,
#             "Recomputation (Latency)": 113.98,
#             "Operator Fission (Memory)": 92.49,
#             "Operator Fission (Latency)": 107.37
#         },
#         "Llama-7B": {
#             "Recomputation (Memory)": 87.31,
#             "Recomputation (Latency)": 109.59,
#             "Operator Fission (Memory)": 84.55,
#             "Operator Fission (Latency)": 114.06
#         }
#     }


recompute_memory_usage = np.array([95.92, 92.13, 87.31])
recompute_latency = np.array([105.16, 113.98, 109.59])
fission_memory_usage = np.array([91.30, 92.49, 84.55])
fission_latency = np.array([107.93, 107.37, 114.06])


plt.grid(linestyle='--',linewidth=1.5, axis='y', zorder=0) 
ax=plt.gca()
ax.spines['bottom'].set_linewidth(1.75)
ax.spines['left'].set_linewidth(1.75)
ax.spines['right'].set_linewidth(1.75)
ax.spines['top'].set_linewidth(1.75)
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))


plt.bar(x + 0 * width*dist, recompute_memory_usage, width=width, color='#6baed6', zorder=100, label='Recomputation (Memory Usage)')
plt.bar(x + 1 * width*dist, recompute_latency, width=width, color='#6baed6',  hatch='x', zorder=100, label='Recomputation (Computation Latency)')
plt.bar(x + 2 * width*dist, fission_memory_usage, width=width, color='#a1d99b', zorder=100, label='Fission (Memory Usage)')
plt.bar(x + 3 * width*dist, fission_latency, width=width, color='#a1d99b', hatch='x',  zorder=100, label='Fission (Computation Latency)')
font = {'style': 'normal', 'weight': 'bold', 'size': 18}
# font = {'family': 'Times New Roman',  'style': 'normal', 'weight': 'bold', 'size': 18}
plt.legend(ncol=2, loc='center', fontsize=18, handletextpad=0.3, columnspacing=0.8, handlelength=1.2, prop=font, bbox_to_anchor=(0.5, 1.12), frameon=False)
plt.xticks([i+width*1.65 for i in x], ['Bert-Large', 'Qwen-1.8B', 'Llama-7B'], weight='bold')
plt.tick_params(axis='both', labelsize=18)
plt.ylim(80)
# plt.subplots_adjust(wspace=0.1, hspace=0.3)
# plt.savefig('./motivation_example.svg', format='svg', bbox_inches='tight', pad_inches=0.02)
plt.show()

