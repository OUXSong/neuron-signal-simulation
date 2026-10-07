"""
神经元黑盒仿真 - 带阈值标注版本
版本：English titles + Chinese comments
关键改进：所有图都显示阈值线，便于观察膜电位何时超过阈值
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 时间参数
# ============================================================
dt = 0.1  # 时间步长 (ms)
T = 500.0  # 总时间长度 (ms)
t = np.arange(0, T, dt)

# ============================================================
# 神经元参数（黑盒模型）
# ============================================================
V_rest = -0.2      # 静息电位
V_reset = -0.2     # 复位电位
threshold = 1.0    # 阈值
leak = 0.15        # 漏电系数（增大以加速衰减）
refractory = 3.0   # 不应期 (ms)

print("=" * 70)
print("NEURON BLACK-BOX SIMULATION - WITH THRESHOLD VISUALIZATION")
print("=" * 70)
print(f"Parameters:")
print(f"  V_rest = {V_rest}")
print(f"  V_reset = {V_reset}")
print(f"  threshold = {threshold} (red dashed line in plots)")
print(f"  leak = {leak}")
print(f"  refractory = {refractory} ms")
print("=" * 70)

# ============================================================
# 输入信号生成函数
# ============================================================

def random_pulse_train(t, strength, seed=0):
    """
    随机脉冲输入：
    随机时刻出现脉冲，幅度与强度成正比
    """
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    p = 0.08 + 0.10 * strength
    active = rng.random(len(t)) < p
    x[active] = strength
    return x


def triangle_wave(t, strength):
    """
    三角波输入：
    周期 200 ms
    """
    period = 200.0
    phase = np.mod(t, period) / period
    x = strength * (1 - 2 * np.abs(phase - 0.5))
    return x


def sine_wave(t, strength):
    """
    正弦波输入：
    周期 120 ms
    """
    period = 120.0
    x = strength * np.sin(2 * np.pi * t / period)
    return x


def step_signal(t, strength):
    """
    阶跃信号输入：
    150 ms 后保持固定强度
    """
    x = np.zeros_like(t)
    x[t >= 150.0] = strength
    return x


# ============================================================
# 神经元模型：LIF (Leaky Integrate-and-Fire)
# ============================================================

def simulate_neuron(input_signal, t):
    """
    神经元仿真：
    
    方程：dV/dt = (input - leak * V) / tau
    
    步骤：
    1. 膜电位积分（输入推高，漏电衰减）
    2. 判断是否达到阈值
    3. 如果达到，发放脉冲并复位
    4. 进入不应期（无法再发放）
    """
    V = np.zeros_like(t)
    V[0] = V_rest
    spikes = np.zeros_like(t, dtype=bool)
    last_spike_time = -1e9

    for i in range(1, len(t)):
        # 1. 检查不应期
        if t[i] - last_spike_time < refractory:
            V[i] = V_reset
            continue

        # 2. 膜电位动态：积分 + 漏电
        V[i] = V[i - 1] + (input_signal[i] - leak * V[i - 1]) * dt

        # 3. 阈值判断
        if V[i] >= threshold:
            spikes[i] = True
            last_spike_time = t[i]
            V[i] = V_reset

    return V, spikes


# ============================================================
# 绘图函数 - 显示输入+膜电位+阈值+脉冲
# ============================================================

def plot_comprehensive(signal_name, signal_fn, strengths):
    """
    每个强度一行，显示：
    左图：输入信号 + 阈值线
    右图：膜电位 + 阈值线 + 脉冲点
    """
    fig, axes = plt.subplots(len(strengths), 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"{signal_name} - Neuron Response with Threshold", 
                 fontsize=14, fontweight="bold")

    for i, strength in enumerate(strengths):
        # 生成输入
        if signal_name == "Random Pulse Train":
            x = signal_fn(t, strength, seed=100 + i * 20)
        else:
            x = signal_fn(t, strength)

        # 仿真神经元
        V, spikes = simulate_neuron(x, t)

        # ---- 左图：输入信号 + 阈值线 ----
        ax_in = axes[i, 0]
        ax_in.plot(t, x, color="royalblue", linewidth=1.8, label="Input Signal")
        ax_in.fill_between(t, 0, x, alpha=0.2, color="royalblue")
        
        # 添加阈值线（便于对比）
        ax_in.axhline(threshold, color="red", linestyle="--", linewidth=2, 
                     label=f"Threshold = {threshold}")
        ax_in.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        
        ax_in.set_title(f"Input - Strength = {strength}", fontsize=11, fontweight="bold")
        ax_in.set_xlabel("Time (ms)")
        ax_in.set_ylabel("Signal Level")
        ax_in.grid(alpha=0.3, linestyle="--")
        ax_in.legend(loc="upper right", fontsize=9)
        ax_in.set_xlim(0, T)

        # ---- 右图：膜电位 + 阈值线 + 脉冲点 ----
        ax_out = axes[i, 1]
        
        # 膜电位曲线
        ax_out.plot(t, V, color="darkorange", linewidth=1.8, label="Membrane Voltage")
        
        # 阈值线（关键！）
        ax_out.axhline(threshold, color="red", linestyle="--", linewidth=2, 
                      label=f"Threshold = {threshold}")
        ax_out.axhline(V_reset, color="gray", linestyle=":", linewidth=0.8, 
                      alpha=0.5, label=f"Reset = {V_reset}")
        
        # 脉冲点
        spike_times = t[spikes]
        num_spikes = len(spike_times)
        if num_spikes > 0:
            ax_out.scatter(spike_times, V[spikes], color="crimson", s=100, 
                          marker="*", label=f"Spikes ({num_spikes})", zorder=10)
            # 用竖线标记脉冲时刻
            ax_out.vlines(spike_times, V_reset, threshold + 0.2, 
                         color="crimson", linewidth=1.5, alpha=0.6, linestyle=":")
        
        ax_out.set_title(f"Membrane Voltage - {num_spikes} Spikes", fontsize=11, fontweight="bold")
        ax_out.set_xlabel("Time (ms)")
        ax_out.set_ylabel("Voltage (V)")
        ax_out.grid(alpha=0.3, linestyle="--")
        ax_out.legend(loc="upper right", fontsize=9)
        ax_out.set_xlim(0, T)

    plt.tight_layout()
    plt.show()


# ============================================================
# 统计分析函数
# ============================================================

def analyze_spikes(signal_name, strength, spikes, t, x, V):
    """
    分析脉冲列的统计特性，以及输入-输出关系
    """
    spike_times = t[spikes]
    num_spikes = len(spike_times)
    
    # 计算输入信号的统计信息
    x_mean = np.mean(x)
    x_max = np.max(x)
    x_min = np.min(x)
    
    print(f"\n[{signal_name}] Strength = {strength}")
    print(f"  Input statistics: min={x_min:.3f}, mean={x_mean:.3f}, max={x_max:.3f}")
    print(f"  Threshold = {threshold}")
    print(f"  Total spikes: {num_spikes}")
    
    if num_spikes > 1:
        isi = np.diff(spike_times)
        print(f"  Mean ISI: {np.mean(isi):.2f} ms (std={np.std(isi):.2f})")
        print(f"  Firing rate: {1000.0 / np.mean(isi):.2f} Hz")
    elif num_spikes == 1:
        print(f"  Only 1 spike at t={spike_times[0]:.1f} ms")
    else:
        print(f"  No spikes - membrane voltage could not reach threshold")


# ============================================================
# 主程序
# ============================================================

strengths = [0.5, 1.0, 2.0]

# 1. 随机脉冲输入
print("\n" + "=" * 70)
print("1. RANDOM PULSE TRAIN")
print("=" * 70)
plot_comprehensive("Random Pulse Train", random_pulse_train, strengths)
for s in strengths:
    x = random_pulse_train(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Random Pulse", s, spikes, t, x, V)

# 2. 三角波输入
print("\n" + "=" * 70)
print("2. TRIANGLE WAVE")
print("=" * 70)
plot_comprehensive("Triangle Wave", triangle_wave, strengths)
for s in strengths:
    x = triangle_wave(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Triangle Wave", s, spikes, t, x, V)

# 3. 正弦波输入
print("\n" + "=" * 70)
print("3. SINE WAVE")
print("=" * 70)
plot_comprehensive("Sine Wave", sine_wave, strengths)
for s in strengths:
    x = sine_wave(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Sine Wave", s, spikes, t, x, V)

# 4. 阶跃信号输入
print("\n" + "=" * 70)
print("4. STEP SIGNAL")
print("=" * 70)
plot_comprehensive("Step Signal", step_signal, strengths)
for s in strengths:
    x = step_signal(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Step Signal", s, spikes, t, x, V)

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
