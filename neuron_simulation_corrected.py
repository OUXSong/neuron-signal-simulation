"""
神经元黑盒仿真 - 修正版
确保：只有当膜电位达到阈值时才放电
版本：English titles + Chinese comments

关键改进：
1. 漏电系数增大 (leak = 0.15) -> 膜电位衰减更快
2. 不应期增加 (refractory = 3.0 ms) -> 防止过度连续发放
3. 详细日志输出 -> 便于调试检查膜电位变化
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
print("NEURON BLACK-BOX SIMULATION - CORRECTED VERSION")
print("=" * 70)
print(f"Parameters:")
print(f"  V_rest = {V_rest}")
print(f"  V_reset = {V_reset}")
print(f"  threshold = {threshold}")
print(f"  leak = {leak} (larger -> faster decay)")
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
    p = 0.08 + 0.10 * strength  # 强度越大，脉冲出现概率越大
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

def simulate_neuron(input_signal, t, verbose=False):
    """
    神经元仿真：
    
    方程：dV/dt = (input - leak * V) / tau
    
    步骤：
    1. 膜电位积分（输入推高，漏电衰减）
    2. 判断是否达到阈值
    3. 如果达到，发放脉冲并复位
    4. 进入不应期（无法再发放）
    
    返回：
      V - 膜电位轨迹
      spikes - 脉冲时刻布尔数组
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
        # V[i] = V[i-1] + dt * (input - leak * V[i-1])
        V[i] = V[i - 1] + (input_signal[i] - leak * V[i - 1]) * dt

        # 3. 阈值判断
        if V[i] >= threshold:
            spikes[i] = True
            last_spike_time = t[i]
            V[i] = V_reset
            if verbose and i % 100 == 0:
                print(f"  Spike at t={t[i]:.1f} ms, V reached {V[i-1]:.3f}")

    return V, spikes


# ============================================================
# 绘图函数
# ============================================================

def plot_case(signal_name, signal_fn, strengths):
    """
    画每种信号在 3 个强度下的输入-输出对比
    """
    fig, axes = plt.subplots(len(strengths), 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"{signal_name} - Input and Output Spikes", fontsize=14, fontweight="bold")

    for i, strength in enumerate(strengths):
        # 生成输入
        if signal_name == "Random Pulse Train":
            x = signal_fn(t, strength, seed=100 + i * 20)
        else:
            x = signal_fn(t, strength)

        # 仿真神经元
        V, spikes = simulate_neuron(x, t, verbose=False)

        # 左：输入信号
        ax_in = axes[i, 0]
        ax_in.plot(t, x, color="royalblue", linewidth=1.5)
        ax_in.fill_between(t, 0, x, alpha=0.2, color="royalblue")
        ax_in.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax_in.set_title(f"Strength = {strength}", fontsize=11)
        ax_in.set_xlabel("Time (ms)")
        ax_in.set_ylabel("Input Signal")
        ax_in.grid(alpha=0.3, linestyle="--")
        ax_in.set_xlim(0, T)

        # 右：输出脉冲
        ax_out = axes[i, 1]
        spike_times = t[spikes]
        num_spikes = len(spike_times)
        
        if num_spikes > 0:
            ax_out.vlines(spike_times, 0, 1, color="crimson", linewidth=2.5)
            ax_out.scatter(spike_times, np.ones_like(spike_times), 
                          color="crimson", s=50, zorder=5)
        
        ax_out.set_title(f"Output Spikes = {num_spikes}", fontsize=11)
        ax_out.set_xlabel("Time (ms)")
        ax_out.set_ylabel("Spike")
        ax_out.set_ylim(-0.3, 1.5)
        ax_out.grid(alpha=0.3, linestyle="--")
        ax_out.set_xlim(0, T)

    plt.show()


def plot_detailed(signal_name, signal_fn, strengths):
    """
    详细图：输入 + 膜电位 + 阈值线 + 脉冲点
    这样可以清楚看到膜电位为什么会发放脉冲
    """
    fig, axes = plt.subplots(len(strengths), 1, figsize=(13, 10), constrained_layout=True)
    
    if len(strengths) == 1:
        axes = [axes]
    
    fig.suptitle(f"{signal_name} - Detailed Analysis (Input + Membrane Voltage)", 
                 fontsize=14, fontweight="bold")

    for i, strength in enumerate(strengths):
        if signal_name == "Random Pulse Train":
            x = signal_fn(t, strength, seed=100 + i * 20)
        else:
            x = signal_fn(t, strength)

        V, spikes = simulate_neuron(x, t)

        ax = axes[i]
        
        # 画输入信号
        ax.plot(t, x, color="royalblue", linewidth=1.5, label="Input", alpha=0.7)
        
        # 画膜电位
        ax.plot(t, V, color="darkorange", linewidth=1.5, label="Membrane Voltage", alpha=0.8)
        
        # 画阈值线
        ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, 
                  label=f"Threshold = {threshold}")
        
        # 标记脉冲发放点
        spike_times = t[spikes]
        if len(spike_times) > 0:
            ax.scatter(spike_times, V[spikes], color="crimson", s=100, 
                      marker="*", label=f"Spikes (N={len(spike_times)})", zorder=10)
        
        ax.set_title(f"Strength = {strength}", fontsize=12)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Voltage")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(0, T)

    plt.show()


# ============================================================
# 统计分析函数
# ============================================================

def analyze_spikes(signal_name, strength, spikes, t):
    """
    分析脉冲列的统计特性
    """
    spike_times = t[spikes]
    num_spikes = len(spike_times)
    
    print(f"\n[{signal_name}] Strength = {strength}")
    print(f"  Total spikes: {num_spikes}")
    
    if num_spikes > 1:
        isi = np.diff(spike_times)  # 脉冲间隔
        print(f"  Mean ISI (inter-spike interval): {np.mean(isi):.2f} ms")
        print(f"  Firing rate: {1000.0 / np.mean(isi):.2f} Hz")
    elif num_spikes == 1:
        print(f"  Only 1 spike at t={spike_times[0]:.1f} ms")
    else:
        print(f"  No spikes detected")


# ============================================================
# 主程序
# ============================================================

strengths = [0.5, 1.0, 2.0]

# 1. 随机脉冲输入
print("\n" + "=" * 70)
print("1. RANDOM PULSE TRAIN")
print("=" * 70)
plot_case("Random Pulse Train", random_pulse_train, strengths)
for s in strengths:
    x = random_pulse_train(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Random Pulse", s, spikes, t)

# 2. 三角波输入
print("\n" + "=" * 70)
print("2. TRIANGLE WAVE")
print("=" * 70)
plot_case("Triangle Wave", triangle_wave, strengths)
for s in strengths:
    x = triangle_wave(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Triangle Wave", s, spikes, t)

# 3. 正弦波输入
print("\n" + "=" * 70)
print("3. SINE WAVE")
print("=" * 70)
plot_case("Sine Wave", sine_wave, strengths)
for s in strengths:
    x = sine_wave(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Sine Wave", s, spikes, t)

# 4. 阶跃信号输入
print("\n" + "=" * 70)
print("4. STEP SIGNAL")
print("=" * 70)
plot_case("Step Signal", step_signal, strengths)
for s in strengths:
    x = step_signal(t, s)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Step Signal", s, spikes, t)

# 5. 详细分析（重点看阶跃信号，最能看出膜电位变化）
print("\n" + "=" * 70)
print("5. DETAILED VIEW: Step Signal")
print("=" * 70)
print("This shows input, membrane voltage, threshold, and spike times together.")
plot_detailed("Step Signal", step_signal, strengths)

print("\nDone!")
