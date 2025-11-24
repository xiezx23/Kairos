from utils.util import load_json
import matplotlib.pyplot as plt
import numpy as np
from utils.color_print import color_text

model_type = ['llama', 'qwen']
model_name = ['LLaMa3 8B', 'Qwen2.5 7B']
quant_type = ['W8A8Linear', 'W4A16Linear', 'W4A16Linear_Marlin', 'W4A8Linear', 'W4A8Linear_QQQ', 'DynamicLinear', 'None']


plt.figure(figsize=(16,8))
font = {'style': 'normal', 'weight': 'bold', 'size': 18}

for idx in range(len(model_type)):
    model = model_type[idx]
    result_list = [[] for _ in range(len(quant_type))]
    TTFT = [[] for _ in range(len(quant_type))]
    TPOT = [[] for _ in range(len(quant_type))]
    VRAM = [[] for _ in range(len(quant_type))]
    TMP = [[] for _ in range(len(quant_type))]
    TOTAL = [[] for _ in range(len(quant_type))]

    input_len = load_json('./result/end2end/input_len.json')
    for i in range(len(quant_type)):
        result_list[i] = load_json('./result/end2end/'+model+'_'+quant_type[i]+'.json')
        for res in result_list[i]:
            TTFT[i].append(res[0])
            TPOT[i].append(res[1])
            VRAM[i].append(res[3])
            TMP[i].append(0)
            TOTAL[i].append(0)
            
    plt.subplot(1,2,1+idx)
    plt.xlabel('Input Length\n'+model_name[idx], font=font)
    plt.ylabel('Latency', font=font)
    categories = input_len[3:6:2]
    width = 0.13
    break_width = 0.15
    x = np.arange(len(categories))

    for i in range(len(TTFT)):
        for j in range(len(TTFT[i])):
            TOTAL[i][j] = TTFT[i][j] + 50 * TPOT[i][j]
    plt.bar(x-break_width*3,   TOTAL[0][3:6:2],  width=width, color = '#033146',edgecolor='k', label = 'SmoothQuant W8A8')
    plt.bar(x-break_width*2,   TOTAL[1][3:6:2],  width=width, color = '#006697',edgecolor='k', label = 'AWQ W4A16')
    plt.bar(x-break_width,     TOTAL[2][3:6:2],  width=width, color = '#008CC4',edgecolor='k', label = 'MARLIN W4A16')
    plt.bar(x,                 TOTAL[3][3:6:2],  width=width, color = '#3EAEDB',edgecolor='k', label = 'Qserve W4A8')
    plt.bar(x+break_width,     TOTAL[4][3:6:2],  width=width, color = '#B1DFF2',edgecolor='k', label = 'QQQ W4A8')
    # plt.bar(x+break_width*3,   TOTAL[6][3:6:2],  width=width, color = '#3CBF8D',edgecolor='k', label = 'FP16 by cuBLAS')
    plt.bar(x+break_width*2,   TOTAL[5][3:6:2],  width=width, color = 'r',edgecolor='k', label = 'Kairos')
    plt.xticks(x, categories, fontsize=16)
    plt.yticks(fontsize=16)
    # plt.grid(True)
    plt.ylim(1000)
    # plt.savefig(f'./{model}_end2end.svg', format='svg', dpi=300, bbox_inches='tight')
plt.savefig(f'./end2end.svg', format='svg', dpi=300, bbox_inches='tight')
# plt.legend(ncol=6, loc='upper center', fontsize=18, frameon=False)
plt.legend(ncol=6, loc='center', fontsize=18, handletextpad=0.3, columnspacing=0.7, handlelength=1.2, bbox_to_anchor=(-0.15, 1.1), frameon=False)
plt.show()


# for i in range(len(input_len)):
#     l = input_len[i]
#     total = []
#     for j in range(len(quant_type)):
#         total.append(TTFT[j][i] + 50 * TPOT[j][i])
 
#     if l == 8192 or l == 16*1024:
#         print('=' * 30)
#         print('Input length:', l)
#         print(color_text('red', f'Accel Ratio to SmoothQ(W8A8): {(total[0]/total[5]):.2f}  {(TTFT[0][i]/TTFT[5][i]):.2f}'))   
#         print(color_text('gre', f'Accel Ratio to AWQ   (W4A16): {(total[1]/total[5]):.2f}  {(TTFT[1][i]/TTFT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Marlin(W4A16): {(total[2]/total[5]):.2f}  {(TTFT[2][i]/TTFT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Qserve (W4A8): {(total[3]/total[5]):.2f}  {(TTFT[3][i]/TTFT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to QQQ    (W4A8): {(total[4]/total[5]):.2f}  {(TTFT[4][i]/TTFT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to Full prec.   : {(total[6]/total[5]):.2f}  {(TTFT[6][i]/TTFT[5][i]):.2f}'))
#         print('=' * 30)
