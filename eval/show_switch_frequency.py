import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    # 'figure.figsize': (8, 5)
})

# ========== 1. 内核计数数据 ==========
kernel_data = {
    'BurstGPT': {'FP16 GEMM': 2632344, 'W8A8 GEMM': 2370540, 'W4A16 GEMM V2': 3258056, 'W4A16 GEMV': 1558926, 'W4A16 GEMM': 10192},
    'ShareGPT': {'FP16 GEMM': 287092, 'W4A16 GEMV': 546066, 'W4A16 GEMM V2': 679321, 'W8A8 GEMM': 135473, 'W4A16 GEMM': 52278},
    'LongBench': {'FP16 GEMM': 832, 'W8A8 GEMM': 2080}
}

# 所有内核名称（统一顺序）
all_kernels = ['FP16 GEMM', 'W8A8 GEMM', 'W4A16 GEMM V2', 'W4A16 GEMV', 'W4A16 GEMM']
datasets = list(kernel_data.keys())

# 构建计数矩阵，并归一化为比例
counts = []
for ds in datasets:
    row = [kernel_data[ds].get(k, 0) for k in all_kernels]
    counts.append(row)
counts = np.array(counts, dtype=float)
# 归一化（每行除以行和）
row_sums = counts.sum(axis=1, keepdims=True)
proportions = counts / row_sums 

import torch
lengths_data = dict()
datasets = ['BurstGPT', 'ShareGPT', 'LongBench']
# datasets = ['BurstGPT', 'ShareGPT', 'LongBench']
for dataset in datasets:
    path = './tmp/input_len'+dataset+'.pt'
    lengths_data[dataset] = torch.load(path, weights_only=False)

# ========== 3. 绘图 ==========
plt.figure(figsize=(16, 4))
# ax_left = plt.subplot2grid((1, 8), (0, 0), colspan=4)
# # ---- 左图：堆叠柱状图（比例） ----
# bottom = np.zeros(len(datasets))
# colors = ['#00ccff', '#B2DF8A', '#FDBF6F', '#FB9A99', '#CAB2D6']
# for i, kernel in enumerate(all_kernels):
#     ax_left.bar(datasets, proportions[:, i], bottom=bottom, label=kernel, color=colors[i])
#     bottom += proportions[:, i]

# ax_left.set_ylabel('Proportion of Kernel Calls')
# # ax_left.set_xlabel('(a) Kernel Selection Proportion')
# # ax_left.legend(loc='upper right', fontsize=12)
# ax_left.legend(ncol=5, loc='upper left', fontsize=12, handletextpad=0.3, columnspacing=0.7, handlelength=1, bbox_to_anchor=(0.15, 1.15), frameon=False)
# ax_left.grid(axis='y', linestyle='--', alpha=0.5)
# # 在柱子上标注百分比（可选）
# for i, ds in enumerate(datasets):
#     y_pos = 0
#     for j, val in enumerate(proportions[i]):
#         if val > 0.05:  # 只标注大于5%的
#             ax_left.text(i, y_pos + val/2, f'{val*100:.1f}%', ha='center', va='center', fontsize=12)
#         y_pos += val


# ---- 右图：箱线图（Boxplot） ----
# 构建 DataFrame

ax_right = plt.subplot2grid((1, 8), (0, 5), colspan=2)
df_list = []
for ds in datasets:
    df_list.append(pd.DataFrame({'length': lengths_data[ds], 'dataset': ds}))
df = pd.concat(df_list, ignore_index=True)

# 绘制箱线图（不显示异常值以避免图形过于拥挤，可根据需要调整）
sns.boxplot(data=df, x='dataset', y='length', ax=ax_right, palette='Set2', showfliers=False)
ax_right.set_ylabel('Input Length (tokens)')
# ax_right.set_xlabel('(b) Input Length Distribution')
ax_right.grid(axis='y', linestyle='--', alpha=0.5)
ax_right.set_yscale('log')

plt.tight_layout()
# plt.show()
plt.savefig(f'./switch_frequency.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'./tmp.png', format='png', dpi=300, bbox_inches='tight')

