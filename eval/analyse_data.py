from utils.util import load_json
import matplotlib.pyplot as plt
from utils.color_print import color_text

test_case = 'prefill' # batch_decode, prefill
quant_type = ['W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear', 'FullLinear']
result_list = [[] for _ in range(len(quant_type))]
TTFT = [[] for _ in range(len(quant_type))]
VRAM = [[] for _ in range(len(quant_type))]


# Load Data
input_len = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 1024*2] + [i for i in range(1024*4, 1024*96+1, 2048)]
if test_case == 'batch_decode':
    input_len = [i for i in range(1, 65)]
            
# input_len = load_json('./data/'+test_case+'/m_list.json')
for i in range(len(quant_type)):
    result_list[i] = load_json('./data/'+test_case+'/'+quant_type[i]+'.json')
    for res in result_list[i]:
        TTFT[i].append(res[0])
        VRAM[i].append(res[2])

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

window = -1
plt.title(f'End2end Test Result')
plt.xlabel('m')
if 0:
    plt.ylabel('Accelerate Ratio')
    for i in range(len(TTFT[0])):
        TTFT[0][i] = TTFT[4][i] / TTFT[0][i]
        TTFT[1][i] = TTFT[4][i] / TTFT[1][i]
        TTFT[2][i] = TTFT[4][i] / TTFT[2][i]
        TTFT[3][i] = TTFT[4][i] / TTFT[3][i]
        TTFT[4][i] = 1
else: plt.ylabel('TTFT')
plt.plot(input_len[0:window], TTFT[0][0:window], color = 'k', label = 'W8A8')
plt.plot(input_len[0:window], TTFT[1][0:window], color = 'g', label = 'W4A16')
plt.plot(input_len[0:window], TTFT[2][0:window], color = 'c', label = 'W4A8')
plt.plot(input_len[0:window], TTFT[3][0:window], color = 'r', label = 'Dynamic')
plt.plot(input_len[0:window], TTFT[4][0:window], color = 'y', label = 'FP16')
plt.legend()
# plt.xticks(input_len)
plt.grid(True)
# plt.show()

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
