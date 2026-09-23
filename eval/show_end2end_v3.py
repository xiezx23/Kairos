from utils.util import load_json
import matplotlib.pyplot as plt
import numpy as np
from utils.color_print import color_text

# model_type = ['qwen']
model_type = ['llama', 'qwen']
# model_name = ['Qwen2.5 7B']
model_name = ['LLaMa3 8B', 'Qwen2.5 7B']
quant_type = ['W8A8Linear', 'W4A16Linear', 'W4A16Linear_Marlin', 'W4A8Linear', 'W4A8Linear_QQQ', 'DynamicLinear', 'None']

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})

for idx in range(len(model_type)):
    plt.figure(figsize=(6,8))
    font = {'style': 'normal', 'weight': 'bold', 'size': 12}

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
    OUTPUT_TOKEN_NUMS = 128
    for i in range(len(TTFT)):
        for j in range(len(TTFT[i])):
            TOTAL[i][j] = int(OUTPUT_TOKEN_NUMS * 1000  / (TTFT[i][j] + (OUTPUT_TOKEN_NUMS-1) * TPOT[i][j]))
    for i in range(len(TTFT)):
        for j in range(len(TTFT[i])):
            TTFT[i][j] /= 1000
        
    plt.subplot(2,1,1)
    plt.xlabel('Sequence Length\n'+model_name[idx])
    plt.ylabel('TTFT (s)')
    width = 0.13
    break_width = 0.15
    categories = input_len[3:]
    x = np.arange(len(categories))

    plt.plot(x,   TTFT[0][3:], '-ok', linewidth=2, label = 'SmoothQuant W8A8')
    plt.plot(x,   TTFT[1][3:], '-o', linewidth=2, color = '#2E75B6', label = 'AWQ W4A16')
    plt.plot(x,   TTFT[2][3:], 's--', linewidth=2, color = '#2E75B6', label = 'MARLIN W4A16')
    plt.plot(x,   TTFT[3][3:], '-og', linewidth=2, label = 'Qserve W4A8')
    # plt.plot(x,   TTFT[4][3:], 's--g', linewidth=2, label = 'QQQ W4A8')
    # plt.plot(x,  TTFT[6][3:], '-oy', linewidth=2, label = 'FP16 by cuBLAS')
    # plt.plot(x,   TTFT[5][3:], '-or', linewidth=2, label = 'Kairos')
    plt.xticks(x, categories)
    plt.yticks()

    # plt.legend(ncol=6, loc='upper left', fontsize=14, handletextpad=0.3, columnspacing=0.7, handlelength=1, bbox_to_anchor=(0.15, 1.15), frameon=False)

        
    plt.subplot(2,1,2)
    plt.xlabel('Sequence Length\n'+model_name[idx])
    plt.ylabel('TPOT (ms)')
    categories = input_len[3:]
    width = 0.13
    break_width = 0.15
    x = np.arange(len(categories))

    plt.plot(x,   TPOT[0][3:], '-ok', linewidth=2, label = 'SmoothQuant W8A8')
    plt.plot(x,   TPOT[1][3:], '-o', linewidth=2, color = '#2E75B6', label = 'AWQ W4A16')
    plt.plot(x,   TPOT[2][3:], 's--', linewidth=2, color = '#2E75B6', label = 'MARLIN W4A16')
    plt.plot(x,   TPOT[3][3:], '-og', linewidth=2, label = 'Qserve W4A8')
    # plt.plot(x,   TPOT[4][3:], 's--g', linewidth=2, label = 'QQQ W4A8')
    # plt.plot(x,  TPOT[6][3:], '-oy', linewidth=2, label = 'FP16 by cuBLAS')
    # plt.plot(x,   TPOT[5][3:], '-or', linewidth=2, label = 'Kairos')
    plt.xticks(x, categories)
    plt.yticks()

    

    # x = np.arange(len(categories))
    # def div_list(a, b):
    #     for idx in range(len(a)):
    #         a[idx] = b[idx] / a[idx]
    # for i in range(6):
    #     div_list(TTFT[i], TTFT[6])
    #     div_list(TPOT[i], TPOT[6])

    # for i in range(len(TTFT)):
    #     for j in range(len(TTFT[i])):
    #         TOTAL[i][j] = TTFT[i][j] + 50 * TPOT[i][j]
    # plt.bar(x-break_width*3,   TTFT[0][3:-1],  width=width, color = '#033146',edgecolor='k', label = 'SmoothQuant W8A8')
    # plt.bar(x-break_width*2,   TTFT[1][3:-1],  width=width, color = '#006697',edgecolor='k', label = 'AWQ W4A16')
    # plt.bar(x-break_width,     TTFT[2][3:-1],  width=width, color = '#008CC4',edgecolor='k', label = 'MARLIN W4A16')
    # plt.bar(x,                 TTFT[3][3:-1],  width=width, color = '#3EAEDB',edgecolor='k', label = 'Qserve W4A8')
    # plt.bar(x+break_width,     TTFT[4][3:-1],  width=width, color = '#B1DFF2',edgecolor='k', label = 'QQQ W4A8')
    # # plt.bar(x+break_width*3,   TTFT[6][3:-1],  width=width, color = '#3CBF8D',edgecolor='k', label = 'FP16 by cuBLAS')
    # plt.bar(x+break_width*2,   TTFT[5][3:-1],  width=width, color = 'r',edgecolor='k', label = 'Kairos')
    # plt.xticks(x, categories, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylim(0.6)

    # plt.subplot(1,2,2)
    # plt.xlabel('Sequence Length\n'+model_name[idx], font=font)
    # plt.ylabel('Latency', font=font)

    # plt.bar(x-break_width*3,   TPOT[0][3:-1],  width=width, color = '#033146',edgecolor='k', label = 'SmoothQuant W8A8')
    # plt.bar(x-break_width*2,   TPOT[1][3:-1],  width=width, color = '#006697',edgecolor='k', label = 'AWQ W4A16')
    # plt.bar(x-break_width,     TPOT[2][3:-1],  width=width, color = '#008CC4',edgecolor='k', label = 'MARLIN W4A16')
    # plt.bar(x,                 TPOT[3][3:-1],  width=width, color = '#3EAEDB',edgecolor='k', label = 'Qserve W4A8')
    # plt.bar(x+break_width,     TPOT[4][3:-1],  width=width, color = '#B1DFF2',edgecolor='k', label = 'QQQ W4A8')
    # # plt.bar(x+break_width*3,   TPOT[6][3:-1],  width=width, color = '#3CBF8D',edgecolor='k', label = 'FP16 by cuBLAS')
    # plt.bar(x+break_width*2,   TPOT[5][3:-1],  width=width, color = 'r',edgecolor='k', label = 'Kairos')
    # plt.xticks(x, categories, fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.ylim(0.6)

    # plt.grid(True)
    # plt.ylim(1000)
    # plt.savefig(f'./{model}_end2end.svg', format='svg', dpi=300, bbox_inches='tight')
    
    # plt.savefig(f'./end2end.svg', format='svg', dpi=300, bbox_inches='tight')
    # plt.legend(ncol=6, loc='upper center', fontsize=18, frameon=False)
    # plt.legend(ncol=6, loc='center', fontsize=14, handletextpad=0.3, columnspacing=0.7, handlelength=1.2, bbox_to_anchor=(-0.15, 1.1), frameon=False)
    plt.show()


# for i in range(len(input_len)):
#     l = input_len[i]
#     total = []
#     for j in range(len(quant_type)):
#         total.append(TTFT[j][i] + 50 * TPOT[j][i])
 
#     if l == 8192 or l == 16*1024:
#         print('=' * 30)
#         print('Sequence length:', l)
#         print(color_text('red', f'Accel Ratio to SmoothQ(W8A8): {(total[0]/total[5]):.2f}  {(TTFT[0][i]/TTFT[5][i]):.2f}'))   
#         print(color_text('gre', f'Accel Ratio to AWQ   (W4A16): {(total[1]/total[5]):.2f}  {(TTFT[1][i]/TTFT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Marlin(W4A16): {(total[2]/total[5]):.2f}  {(TTFT[2][i]/TTFT[5][i]):.2f}'))
#         print(color_text('red', f'Accel Ratio to Qserve (W4A8): {(total[3]/total[5]):.2f}  {(TTFT[3][i]/TTFT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to QQQ    (W4A8): {(total[4]/total[5]):.2f}  {(TTFT[4][i]/TTFT[5][i]):.2f}'))    
#         print(color_text('gre', f'Accel Ratio to Full prec.   : {(total[6]/total[5]):.2f}  {(TTFT[6][i]/TTFT[5][i]):.2f}'))
#         print('=' * 30)
