import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置全局字体和样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'text.usetex': False,  # 如果系统支持 LaTeX 可改为 True
    'figure.dpi': 150,
    'figure.figsize': (8, 5)
})

# ======================= 1. 单 GEMM 加速比曲线 =======================
# 模拟数据：M 值 (对数采样) 和 加速比 (Kairos / best static)
M_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
# 对于形状 (4096,4096) 在 A100 上的加速比 (相对最佳静态内核)
speedup = [1.00, 1.00, 1.00, 1.02, 1.05, 1.10, 1.20, 1.35, 1.55, 1.70, 1.80, 1.85, 1.88]

plt.figure(figsize=(8,5))
plt.plot(M_values, speedup, 'o-', color='#2c7bb6', linewidth=2, markersize=6, label='Kairos vs. Best Static')
plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
plt.xscale('log')
plt.xlabel('Input dimension $M$', fontsize=12)
plt.ylabel('Speedup over best static kernel', fontsize=12)
plt.title('Single GEMM Performance (LLaMa-3 8B, shape=4096×4096, A100)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fig_single_gemm_speedup.png', dpi=150)
plt.show()

# ======================= 2. 复合 GEMM 柱状图 =======================
# 模拟数据：不同 M 下的总延迟 (相对值)
M_compound = [1, 16, 64, 256, 1024]
# 各策略的归一化延迟 (越小越好)
static_w4a16 = [1.00, 1.20, 2.00, 3.00, 4.00]   # 小M好，大M差
static_w4a8  = [1.50, 1.50, 1.50, 1.80, 2.50]   # 平衡
static_w8a8  = [3.00, 2.50, 1.50, 1.20, 1.00]   # 大M好
kairos       = [1.00, 1.20, 1.50, 1.20, 1.00]   # 动态切换

x = np.arange(len(M_compound))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - 1.5*width, static_w4a16, width, label='W4A16 (static)', color='#d7191c')
ax.bar(x - 0.5*width, static_w4a8,  width, label='W4A8 (static)',  color='#fdae61')
ax.bar(x + 0.5*width, static_w8a8,  width, label='W8A8 (static)',  color='#abd9e9')
ax.bar(x + 1.5*width, kairos,       width, label='Kairos',         color='#2c7bb6')

ax.set_xlabel('Input dimension $M$', fontsize=12)
ax.set_ylabel('Normalized Latency (lower is better)', fontsize=12)
ax.set_title('Compound GEMM (3 layers) Performance', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([str(m) for m in M_compound])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_compound_gemm.png', dpi=150)
plt.show()

# ======================= 3. 端到端延迟对比 =======================
# 模拟数据：不同输入长度下的端到端延迟 (ms)
seq_lengths = [128, 512, 2048, 4096]
# LLaMa-3 8B on A100
trt_fp16 = [45, 180, 720, 1440]
trt_w4a16 = [30, 110, 440, 880]
trt_w8a8 = [35, 120, 400, 800]
marlin = [32, 115, 450, 900]
qserve = [34, 118, 430, 850]
kairos = [28, 100, 380, 760]

plt.figure(figsize=(10, 6))
plt.plot(seq_lengths, trt_fp16, 's-', label='TensorRT-LLM FP16', color='gray')
plt.plot(seq_lengths, trt_w4a16, '^-', label='TensorRT-LLM W4A16', color='#d7191c')
plt.plot(seq_lengths, trt_w8a8,  'v-', label='TensorRT-LLM W8A8',  color='#abd9e9')
plt.plot(seq_lengths, marlin,    'o-', label='MARLIN (W4A16)',      color='#fdae61')
plt.plot(seq_lengths, qserve,    'd-', label='Qserve (W4A8)',       color='#e6ab02')
plt.plot(seq_lengths, kairos,    '*-', label='Kairos',              color='#2c7bb6', linewidth=2, markersize=10)

plt.xlabel('Input Sequence Length', fontsize=12)
plt.ylabel('End-to-End Latency (ms)', fontsize=12)
plt.title('LLaMa-3 8B End-to-End Inference on A100', fontsize=14)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fig_e2e_latency.png', dpi=150)
plt.show()

# ======================= 4. 加速比总结柱状图 =======================
# 对比 Kairos 相对于各静态方法的平均加速比
methods = ['MARLIN', 'Qserve', 'TRT-W4A16', 'TRT-W8A8']
speedup_avg = [1.57, 1.63, 1.48, 1.25]  # 从论文摘要中取值

plt.figure(figsize=(6, 5))
bars = plt.bar(methods, speedup_avg, color=['#d7191c', '#fdae61', '#abd9e9', 'gray'])
plt.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
plt.ylabel('Average Speedup over Baseline', fontsize=12)
plt.title('Kairos vs. State-of-the-Art (LLaMa-3 8B, A100)', fontsize=14)
for bar, val in zip(bars, speedup_avg):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}×', ha='center', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig_speedup_summary.png', dpi=150)
plt.show()