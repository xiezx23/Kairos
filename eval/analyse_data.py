from utils.util import load_json
import matplotlib.pyplot as plt
import numpy as np
from utils.color_print import color_text

model = 'llama'
quant_type = ['W8A8Linear', 'W4A16Linear', 'W4A16Linear_Marlin', 'W4A8Linear', 'W4A8Linear_QQQ', 'DynamicLinear', 'None']
result_list = [[] for _ in range(len(quant_type))]
TTFT = [[] for _ in range(len(quant_type))]
TPOT = [[] for _ in range(len(quant_type))]
VRAM = [[] for _ in range(len(quant_type))]
TMP = [[] for _ in range(len(quant_type))]
TOTAL = [[] for _ in range(len(quant_type))]


# Load Data
input_len = [1024, 1024*2, 1024*4, 1024*8, 1024*12, 1024*16]

            
# input_len = load_json('./data/'+test_case+'/m_list.json')
for i in range(len(quant_type)):
    result_list[i] = load_json('./result/end2end/'+model+'_'+quant_type[i]+'.json')
    for res in result_list[i]:
        TTFT[i].append(res[0])
        TPOT[i].append(res[1])
        VRAM[i].append(res[3])
        TMP[i].append(0)
        TOTAL[i].append(0)

# for outputTokens in [10, 20, 30, 40, 50, 60, 70]:
#     # outputTokens = 50
#     for i in range(len(TTFT[0])):
#         for j in range(len(quant_type)):
#             TMP[j][i] = outputTokens * TPOT[j][i]

#     for i in range(len(input_len)):
#         l = input_len[i]
#         total = []
#         for j in range(len(quant_type)):
#             total.append(TTFT[j][i] + TMP[j][i])
    
#         if l == 8192:
#             print('=' * 30)
#             print('Input length:', l, '    ouput length:',outputTokens)
#             print(color_text('red', f'Accel Ratio to SmoothQ(W8A8): {(total[0]/total[5]):.2f}  {(TTFT[0][i]/TTFT[5][i]):.2f}  {(TPOT[0][i]/TPOT[5][i]):.2f}'))   
#             print(color_text('gre', f'Accel Ratio to AWQ   (W4A16): {(total[1]/total[5]):.2f}  {(TTFT[1][i]/TTFT[5][i]):.2f}  {(TPOT[1][i]/TPOT[5][i]):.2f}'))
#             print(color_text('red', f'Accel Ratio to Marlin(W4A16): {(total[2]/total[5]):.2f}  {(TTFT[2][i]/TTFT[5][i]):.2f}  {(TPOT[2][i]/TPOT[5][i]):.2f}'))
#             print(color_text('red', f'Accel Ratio to Qserve (W4A8): {(total[3]/total[5]):.2f}  {(TTFT[3][i]/TTFT[5][i]):.2f}  {(TPOT[3][i]/TPOT[5][i]):.2f}'))    
#             print(color_text('gre', f'Accel Ratio to QQQ    (W4A8): {(total[4]/total[5]):.2f}  {(TTFT[4][i]/TTFT[5][i]):.2f}  {(TPOT[4][i]/TPOT[5][i]):.2f}'))    
#             print(color_text('gre', f'Accel Ratio to Full prec.   : {(total[6]/total[5]):.2f}  {(TTFT[6][i]/TTFT[5][i]):.2f}  {(TPOT[6][i]/TPOT[5][i]):.2f}'))
#             print('=' * 30)

# outputTokens = 32
# for i in range(len(TTFT[0])):
#     for j in range(len(quant_type)):
#         TPOT[j][i] = outputTokens * TPOT[j][i]

# for i in range(len(input_len)):
#     l = input_len[i]
#     total = []
#     for j in range(len(quant_type)):
#         total.append(TTFT[j][i] + TPOT[j][i])
#     if l == 8192:
#         print('=' * 30)
#         print('Input length:', l, '    ouput length:',outputTokens)
#         print('      Acceleration          | total, ttft, tpot')
#         print(color_text('red', f'Accel Ratio to SmoothQ(W8A8): {(total[0]/total[5]):.2f}  {(TTFT[0][i]/TTFT[5][i]):.2f}  {(TPOT[0][i]/TPOT[5][i]):.2f}'))   
#         print(color_text('gre', f'Accel Ratio to AWQ   (W4A16): {(total[1]/total[5]):.2f}  {(TTFT[1][i]/TTFT[5][i]):.2f}  {(TPOT[1][i]/TPOT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Marlin(W4A16): {(total[2]/total[5]):.2f}  {(TTFT[2][i]/TTFT[5][i]):.2f}  {(TPOT[2][i]/TPOT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Qserve (W4A8): {(total[3]/total[5]):.2f}  {(TTFT[3][i]/TTFT[5][i]):.2f}  {(TPOT[3][i]/TPOT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to QQQ    (W4A8): {(total[4]/total[5]):.2f}  {(TTFT[4][i]/TTFT[5][i]):.2f}  {(TPOT[4][i]/TPOT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to Full prec.   : {(total[6]/total[5]):.2f}  {(TTFT[6][i]/TTFT[5][i]):.2f}  {(TPOT[6][i]/TPOT[5][i]):.2f}'))
#         print('=' * 30)

#         plt.figure(figsize=(5.7,5))
#         # plt.xlabel('Input Length')
#         plt.ylabel('Latency')
#         # plt.ylabel('Throughput')
#         width = 0.5
#         x = np.arange(len(quant_type)-1)
#         # plt.bar(0,  TTFT[0][i], width=width,hatch='ooo', color = '#B6B4A7',edgecolor='k')
#         # plt.bar(1,  TTFT[1][i], width=width,hatch='ooo', color = '#4291F2',edgecolor='k')
#         # plt.bar(2,    TTFT[2][i], width=width,hatch='ooo', color = '#6baed6',edgecolor='k')
#         # plt.bar(3,          TTFT[3][i], width=width,hatch='ooo', color = 'c',edgecolor='k')
#         # plt.bar(4,    TTFT[4][i], width=width,hatch='ooo', color = '#3CBF8D',edgecolor='k')
#         # plt.bar(5,  TTFT[5][i], width=width,hatch='ooo', color = 'r',edgecolor='k')
#         # plt.bar(6,  TTFT[6][i], width=width,hatch='ooo', color = '#EADB52',edgecolor='k')

#         # # plt.plot(input_len, TTFT[0][i], color = 'k', label = 'W8A8')
#         # # plt.plot(input_len, TTFT[1][i], color = 'g', label = 'W4A16')
#         # # plt.plot(input_len, TTFT[2][i], color = 'c', label = 'W4A8')
#         # # plt.plot(input_len, TTFT[3][i], color = 'r', label = 'Dynamic')
#         # # plt.plot(input_len, TTFT[4][i], color = 'y', label = 'FP16')

#         # plt.bar(0,   TPOT[0][i], bottom= TTFT[0][i], width=width, color = '#B6B4A7',edgecolor='k')
#         # plt.bar(1,   TPOT[1][i], bottom= TTFT[1][i], width=width, color = '#4291F2',edgecolor='k')
#         # plt.bar(2,     TPOT[2][i], bottom= TTFT[2][i], width=width, color = '#6baed6',edgecolor='k')
#         # plt.bar(3,           TPOT[3][i], bottom= TTFT[3][i], width=width, color = 'c',edgecolor='k')
#         # plt.bar(4,     TPOT[4][i], bottom= TTFT[4][i], width=width, color = '#3CBF8D',edgecolor='k')
#         # plt.bar(5,   TPOT[5][i], bottom= TTFT[5][i], width=width, color = 'r',edgecolor='k')
#         # plt.bar(6,   TPOT[6][i], bottom= TTFT[6][i], width=width, color = '#EADB52',edgecolor='k')

        
#         # plt.bar(0,  outputTokens*1000/total[0], width=width, color = '#B6B4A7',edgecolor='k')
#         plt.bar(0,  total[1], width=width, color = '#4291F2',edgecolor='k')
#         plt.bar(1,  total[2], width=width, color = '#6baed6',edgecolor='k')
#         plt.bar(2,  total[3], width=width, color = 'c',edgecolor='k')
#         plt.bar(3,  total[4], width=width, color = '#3CBF8D',edgecolor='k')
#         plt.bar(4,  total[5], width=width, color = 'r',edgecolor='k')
#         plt.bar(5,  total[6], width=width, color = '#EADB52',edgecolor='k')

#         label = ['AWQ\nW4A16', 'Marlin\nW4A16', 'Qserve\nW4A8', 'QQQ\nW4A8', 'Kairos', 'cuBLAS\nFP16']
#         plt.xticks(x, label)
#         # plt.grid(True)
#         plt.savefig(f'./{model}_end2end.svg', format='svg', dpi=300, bbox_inches='tight')
#         plt.show()
#         exit(0)


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

plt.figure(figsize=(16,4))
font = {'style': 'normal', 'weight': 'bold', 'size': 18}

plt.subplot(1,4,1)
plt.xlabel('Input Length', font=font)
plt.ylabel('Latency', font=font)
categories = input_len[3:6:2]
width = 0.13
break_width = 0.15
x = np.arange(len(categories))
# plt.bar(x-width*3,  TTFT[0], width=width,hatch='ooo', color = '#B6B4A7',edgecolor='k')
# plt.bar(x-width*2,  TTFT[1], width=width,hatch='ooo', color = '#4291F2',edgecolor='k')
# plt.bar(x-width,    TTFT[2], width=width,hatch='ooo', color = '#6baed6',edgecolor='k')
# plt.bar(x,          TTFT[3], width=width,hatch='ooo', color = 'c',edgecolor='k')
# plt.bar(x+width,    TTFT[4], width=width,hatch='ooo', color = '#3CBF8D',edgecolor='k')
# plt.bar(x+width*2,  TTFT[5], width=width,hatch='ooo', color = 'r',edgecolor='k')
# plt.bar(x+width*3,  TTFT[6], width=width,hatch='ooo', color = '#EADB52',edgecolor='k')

# plt.plot(input_len, TTFT[0], color = 'k', label = 'W8A8')
# plt.plot(input_len, TTFT[1], color = 'g', label = 'W4A16')
# plt.plot(input_len, TTFT[2], color = 'c', label = 'W4A8')
# plt.plot(input_len, TTFT[3], color = 'r', label = 'Dynamic')
# plt.plot(input_len, TTFT[4], color = 'y', label = 'FP16')

for i in range(len(TTFT)):
    for j in range(len(TTFT[i])):
        TOTAL[i][j] = TTFT[i][j] + 50 * TPOT[i][j]
plt.bar(x-break_width*3,   TOTAL[0][3:6:2],  width=width, color = '#033146',edgecolor='k', label = 'SmoothQuant W8A8')
plt.bar(x-break_width*2,   TOTAL[1][3:6:2],  width=width, color = '#006697',edgecolor='k', label = 'AWQ W4A16')
plt.bar(x-break_width,     TOTAL[2][3:6:2],  width=width, color = '#008CC4',edgecolor='k', label = 'MARLIN W4A16')
plt.bar(x,           TOTAL[3][3:6:2],  width=width, color = '#3EAEDB',edgecolor='k', label = 'Qserve W4A8')
plt.bar(x+break_width,     TOTAL[4][3:6:2],  width=width, color = '#B1DFF2',edgecolor='k', label = 'QQQ W4A8')
# plt.bar(x+break_width*3,   TOTAL[6][3:6:2],  width=width, color = '#3CBF8D',edgecolor='k', label = 'FP16 by cuBLAS')
plt.bar(x+break_width*2,   TOTAL[5][3:6:2],  width=width, color = 'r',edgecolor='k', label = 'Kairos')
plt.xticks(x, categories, fontsize=16)
plt.yticks(fontsize=16)
# plt.legend()
# plt.xticks(input_len)
# plt.grid(True)
# plt.ylim(1000)
plt.savefig(f'./{model}_end2end.svg', format='svg', dpi=300, bbox_inches='tight')
plt.legend(ncol=3, loc='center', fontsize=18, handletextpad=0.3, columnspacing=1., handlelength=1.6, bbox_to_anchor=(0.4, 1.1), frameon=False)
plt.show()


for i in range(len(input_len)):
    l = input_len[i]
    total = []
    for j in range(len(quant_type)):
        total.append(TTFT[j][i] + 50 * TPOT[j][i])
 
    if l == 8192 or l == 16*1024:
        print('=' * 30)
        print('Input length:', l)
        print(color_text('red', f'Accel Ratio to SmoothQ(W8A8): {(total[0]/total[5]):.2f}  {(TTFT[0][i]/TTFT[5][i]):.2f}'))   
        print(color_text('gre', f'Accel Ratio to AWQ   (W4A16): {(total[1]/total[5]):.2f}  {(TTFT[1][i]/TTFT[5][i]):.2f}'))
        print(color_text('red', f'Accel Ratio to Marlin(W4A16): {(total[2]/total[5]):.2f}  {(TTFT[2][i]/TTFT[5][i]):.2f}'))
        print(color_text('red', f'Accel Ratio to Qserve (W4A8): {(total[3]/total[5]):.2f}  {(TTFT[3][i]/TTFT[5][i]):.2f}'))    
        print(color_text('gre', f'Accel Ratio to QQQ    (W4A8): {(total[4]/total[5]):.2f}  {(TTFT[4][i]/TTFT[5][i]):.2f}'))    
        print(color_text('gre', f'Accel Ratio to Full prec.   : {(total[6]/total[5]):.2f}  {(TTFT[6][i]/TTFT[5][i]):.2f}'))
        print('=' * 30)


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
