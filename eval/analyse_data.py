from utils.util import load_json
import matplotlib.pyplot as plt
import numpy as np
from utils.color_print import color_text

model = 'qwen' # batch_decode, prefill
quant_type = ['W8A8Linear', 'W4A16Linear', 'W4A16Linear_Marlin', 'W4A8Linear', 'W4A8Linear_QQQ', 'DynamicLinear', 'None']
result_list = [[] for _ in range(len(quant_type))]
TTFT = [[] for _ in range(len(quant_type))]
TPOT = [[] for _ in range(len(quant_type))]
VRAM = [[] for _ in range(len(quant_type))]


# Load Data
input_len = [1024, 1024*2, 1024*4, 1024*8, 1024*12, 1024*16]

            
# input_len = load_json('./data/'+test_case+'/m_list.json')
for i in range(len(quant_type)):
    result_list[i] = load_json('./result/end2end/'+model+'_'+quant_type[i]+'.json')
    for res in result_list[i]:
        TTFT[i].append(res[0])
        TPOT[i].append(res[1])
        VRAM[i].append(res[3])

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

plt.figure(figsize=(5.7,5))
plt.xlabel('Input Length')
categories = input_len
width = 0.12
x = np.arange(len(categories))
if 1:
    outputTokens = 50
    plt.ylabel('Total Time')
    for i in range(len(TTFT[0])):
        for j in range(len(quant_type)):
            TPOT[j][i] = outputTokens * TPOT[j][i]
    
    plt.bar(x-width*3,  TTFT[0], width=width,hatch='ooo', color = '#B6B4A7',edgecolor='k')
    plt.bar(x-width*2,  TTFT[1], width=width,hatch='ooo', color = '#4291F2',edgecolor='k')
    plt.bar(x-width,    TTFT[2], width=width,hatch='ooo', color = '#6baed6',edgecolor='k')
    plt.bar(x,          TTFT[3], width=width,hatch='ooo', color = 'c',edgecolor='k')
    plt.bar(x+width,    TTFT[4], width=width,hatch='ooo', color = '#3CBF8D',edgecolor='k')
    plt.bar(x+width*2,  TTFT[5], width=width,hatch='ooo', color = 'r',edgecolor='k')
    plt.bar(x+width*3,  TTFT[6], width=width,hatch='ooo', color = '#EADB52',edgecolor='k')
else: plt.ylabel('TPOT')
# plt.plot(input_len, TTFT[0], color = 'k', label = 'W8A8')
# plt.plot(input_len, TTFT[1], color = 'g', label = 'W4A16')
# plt.plot(input_len, TTFT[2], color = 'c', label = 'W4A8')
# plt.plot(input_len, TTFT[3], color = 'r', label = 'Dynamic')
# plt.plot(input_len, TTFT[4], color = 'y', label = 'FP16')

plt.bar(x-width*3,   TPOT[0], bottom= TTFT[0], width=width, color = '#B6B4A7',edgecolor='k', label = 'W8A8 by SmoothQuant')
plt.bar(x-width*2,   TPOT[1], bottom= TTFT[1], width=width, color = '#4291F2',edgecolor='k', label = 'W4A16 by AWQ')
plt.bar(x-width,     TPOT[2], bottom= TTFT[2], width=width, color = '#6baed6',edgecolor='k', label = 'W4A16 by Marlin')
plt.bar(x,           TPOT[3], bottom= TTFT[3], width=width, color = 'c',edgecolor='k', label = 'W4A8 by Qserve')
plt.bar(x+width,     TPOT[4], bottom= TTFT[4], width=width, color = '#3CBF8D',edgecolor='k', label = 'W4A8 by QQQ')
plt.bar(x+width*2,   TPOT[5], bottom= TTFT[5], width=width, color = 'r',edgecolor='k', label = 'Kairos')
plt.bar(x+width*3,   TPOT[6], bottom= TTFT[6], width=width, color = '#EADB52',edgecolor='k', label = 'FP16 by cuBLAS')
plt.xticks(x, categories)
plt.legend()
# plt.xticks(input_len)
# plt.grid(True)
plt.savefig(f'./{model}_end2end.svg', format='svg', dpi=300, bbox_inches='tight')
plt.show()
exit(0)

for i in range(len(input_len)):
    m = input_len[i]
    print(color_text('red',f'M: {m}'))
    print(f'W4A8    TTFT: {TTFT[2][i]:6.2f} ms  Peak Vram Usage: {VRAM[2][i]:6.2f} GB')
    print(f'W4A16   TTFT: {TTFT[1][i]:6.2f} ms  Peak Vram Usage: {VRAM[1][i]:6.2f} GB')
    print(f'W8A8    TTFT: {TTFT[0][i]:6.2f} ms  Peak Vram Usage: {VRAM[0][i]:6.2f} GB')
    print(f'Dynamic TTFT: {TTFT[3][i]:6.2f} ms  Peak Vram Usage: {VRAM[3][i]:6.2f} GB')
    if TTFT[1][i] < TTFT[2][i]:
        best_4bit_ttft = TTFT[1][i]
        ref_vram_usage = VRAM[1][i]
    else:
        best_4bit_ttft = TTFT[2][i]
        ref_vram_usage = VRAM[2][i]
    print(color_text('gre', f'Speedup* : {(best_4bit_ttft-TTFT[3][i])/best_4bit_ttft:4.2f} x      VRAM Increase: {(VRAM[3][i]-ref_vram_usage)/ref_vram_usage:.4f}'))
    print('=' * 55)
