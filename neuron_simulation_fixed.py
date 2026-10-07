"""
神经元黑盒仿真：输入信号 -> 阈值型脉冲输出
适用于：随机脉冲、三角波、正弦波、阶跃信号
3 个不同强度，每种信号都展示

解决方案：使用系统默认字体或 DejaVu 字体来显示中文
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# 设置中文字体显示
# 方案1：使用 SimHei（如果系统有）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 备选方案2：使用 DejaVu 显示英文，标题用数字标记
plt.rcParams['font.size'] = 10

# 时间参数（毫秒）
dt = 0.1
T = 500.0
t = np.arange(0, T, dt)

# 神经元参数（黑盒抽象）
V_rest = 0.0
V_reset = 0.0
threshold = 1.0
leak = 0.08
refractory = 2.0  # 不应期，单位 ms

print("=" * 70)
print("NEURON BLACK-BOX SIMULATION")
print("=" * 70)
print(f"Time range: 0 - {T} ms, dt = {dt} ms")
print(f"Neuron parameters:")
print(f"  - Resting potential: {V_rest}")
print(f"  - Threshold: {threshold}")
print(f"  - Leak conductance: {leak}")
print(f"  - Refractory period: {refractory} ms")
print("=" * 70)

# ===================================================
# 1. 输入信号生成函数
# ===================================================

def random_pulse_train(t, strength, seed=0):
    """
    Random pulse train input
    Larger strength => higher probability of pulse
    """
    rng = np.random.default_rng(seed)
    x = np.zeros_like(t)
    p = 0.10 + 0.12 * strength
    active = rng.random(len(t)) < p
    x[active] = strength
    return x

def triangle_wave(t, strength):
    """
    Triangle wave: rises for 0-100ms, falls for 100-200ms
    Period = 200ms
    """
    period = 200.0
    phase = np.mod(t, period) / period
    x = strength * (1 - 2 * np.abs(phase - 0.5))
    return x

def sine_wave(t, strength):
    """
    Sinusoidal input: period = 120 ms
    """
    period = 120.0
    x = strength * np.sin(2 * np.pi * t / period)
    return x

def step_signal(t, strength):
    """
    Step input: jumps to constant value at t=150ms
    """
    x = np.zeros_like(t)
    x[t >= 150.0] = strength
    return x

# ===================================================
# 2. 神经元仿真：积分-阈值-复位模型
# ===================================================

def simulate_neuron(input_signal, t):
    """
    Simulate neuron using LIF model:
    - Integrate input signal
    - Check threshold
    - Reset after spike
    - Enforce refractory period
    
    Returns:
      - V: membrane voltage over time
      - spikes: boolean array indicating spike times
    """
    V = np.zeros_like(t)
    spikes = np.zeros_like(t, dtype=bool)
    last_spike_time = -1e9

    for i in range(1, len(t)):
        # 绝对不应期：不允许再次发放
        if t[i] - last_spike_time < refractory:
            V[i] = V_reset
            continue

        # 膜电位更新：积分 + 漏电 (leaky integration)
        V[i] = V[i - 1] + (input_signal[i] - leak * V[i - 1]) * dt

        # 阈值放电
        if V[i] >= threshold:
            spikes[i] = True
            last_spike_time = t[i]
            V[i] = V_reset

    return V, spikes

# ===================================================
# 3. 可视化函数
# ===================================================

def plot_signal_case(signal_name_en, signal_name_cn, signal_fn, strengths):
    """
    Plot input signal and output spikes for 3 different strengths
    """
    fig, axes = plt.subplots(len(strengths), 2, figsize=(13, 9), constrained_layout=True)
    
    title_cn = f"{signal_name_cn} - Input and Output Spikes"
    fig.suptitle(title_cn, fontsize=14, fontweight='bold')

    for i, strength in enumerate(strengths):
        # Generate input signal
        if signal_name_en == "random":
            x = signal_fn(t, strength, seed=100 + i * 10)
        else:
            x = signal_fn(t, strength)

        # Run neuron simulation
        V, spikes = simulate_neuron(x, t)

        # ---- Left: Input Signal ----
        ax_in = axes[i, 0]
        ax_in.plot(t, x, color="royalblue", lw=1.5)
        ax_in.fill_between(t, x, alpha=0.3, color="royalblue")
        ax_in.set_title(f"Strength = {strength}", fontsize=11)
        ax_in.set_xlabel("Time (ms)")
        ax_in.set_ylabel("Input Signal")
        ax_in.grid(alpha=0.3, linestyle='--')
        ax_in.set_xlim(0, T)

        # ---- Right: Output Spikes ----
        ax_out = axes[i, 1]
        spike_times = t[spikes]
        num_spikes = spike_times.size
        
        if num_spikes > 0:
            ax_out.vlines(spike_times, 0, 1, colors='crimson', linewidth=2)
            ax_out.scatter(spike_times, np.ones_like(spike_times), 
                          color='crimson', s=50, zorder=5)
        
        ax_out.set_title(f"Output Spikes (Count = {num_spikes})", fontsize=11)
        ax_out.set_xlabel("Time (ms)")
        ax_out.set_ylabel("Spike Event")
        ax_out.set_ylim(-0.2, 1.5)
        ax_out.grid(alpha=0.3, linestyle='--')
        ax_out.set_xlim(0, T)

    plt.show()
    
    return fig

def plot_membrane_voltage_case(signal_name_en, signal_name_cn, signal_fn, strengths):
    """
    Plot input signal, membrane voltage, and spike events
    This shows HOW continuous input is converted to discrete spikes
    """
    fig, axes = plt.subplots(len(strengths), 2, figsize=(13, 9), constrained_layout=True)
    
    title_cn = f"{signal_name_cn} - Input and Membrane Voltage"
    fig.suptitle(title_cn, fontsize=14, fontweight='bold')

    for i, strength in enumerate(strengths):
        # Generate input
        if signal_name_en == "random":
            x = signal_fn(t, strength, seed=50 + i * 7)
        else:
            x = signal_fn(t, strength)

        V, spikes = simulate_neuron(x, t)

        # ---- Left: Input Signal ----
        ax_in = axes[i, 0]
        ax_in.plot(t, x, color="royalblue", lw=1.5, label="Input")
        ax_in.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f"Threshold = {threshold}")
        ax_in.fill_between(t, 0, x, alpha=0.2, color="royalblue")
        ax_in.set_title(f"Strength = {strength}", fontsize=11)
        ax_in.set_xlabel("Time (ms)")
        ax_in.set_ylabel("Input Signal")
        ax_in.grid(alpha=0.3, linestyle='--')
        ax_in.legend(loc='upper right', fontsize=9)
        ax_in.set_xlim(0, T)

        # ---- Right: Membrane Voltage ----
        ax_out = axes[i, 1]
        ax_out.plot(t, V, color="darkorange", lw=1.5, label="Membrane V")
        ax_out.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f"Threshold")
        ax_out.scatter(t[spikes], V[spikes], color='crimson', s=50, zorder=5, label="Spikes")
        ax_out.set_title(f"Membrane Voltage (Spikes = {np.sum(spikes)})", fontsize=11)
        ax_out.set_xlabel("Time (ms)")
        ax_out.set_ylabel("Voltage (V)")
        ax_out.grid(alpha=0.3, linestyle='--')
        ax_out.legend(loc='upper right', fontsize=9)
        ax_out.set_xlim(0, T)

    plt.show()
    
    return fig

# ===================================================
# 4. 分析函数：统计脉冲信息
# ===================================================

def analyze_spikes(signal_name, strength, spikes, t):
    """
    Analyze spike train statistics
    """
    spike_times = t[spikes]
    num_spikes = len(spike_times)
    
    if num_spikes > 1:
        isi = np.diff(spike_times)  # Inter-spike intervals
        mean_isi = np.mean(isi)
        firing_rate = 1000.0 / mean_isi  # Hz (convert from ms)
    else:
        mean_isi = np.nan
        firing_rate = np.nan
    
    print(f"\n{signal_name} | Strength = {strength}")
    print(f"  Total spikes: {num_spikes}")
    print(f"  Mean ISI: {mean_isi:.2f} ms")
    print(f"  Firing rate: {firing_rate:.2f} Hz")
    
    return num_spikes, mean_isi, firing_rate

# ===================================================
# 5. 运行仿真：4 类信号，每类 3 个强度
# ===================================================

strengths = [0.5, 1.0, 2.0]

print("\n" + "=" * 70)
print("1. RANDOM PULSE TRAIN")
print("=" * 70)
plot_signal_case("random", "Random Pulse Train", random_pulse_train, strengths)

# 统计分析
print("\nAnalysis - Random Pulse Train:")
for strength in strengths:
    x = random_pulse_train(t, strength)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Random", strength, spikes, t)

print("\n" + "=" * 70)
print("2. TRIANGLE WAVE")
print("=" * 70)
plot_signal_case("triangle", "Triangle Wave", triangle_wave, strengths)

# 统计分析
print("\nAnalysis - Triangle Wave:")
for strength in strengths:
    x = triangle_wave(t, strength)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Triangle", strength, spikes, t)

print("\n" + "=" * 70)
print("3. SINE WAVE")
print("=" * 70)
plot_signal_case("sine", "Sine Wave", sine_wave, strengths)

# 统计分析
print("\nAnalysis - Sine Wave:")
for strength in strengths:
    x = sine_wave(t, strength)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Sine", strength, spikes, t)

print("\n" + "=" * 70)
print("4. STEP SIGNAL")
print("=" * 70)
plot_signal_case("step", "Step Signal", step_signal, strengths)

# 统计分析
print("\nAnalysis - Step Signal:")
for strength in strengths:
    x = step_signal(t, strength)
    V, spikes = simulate_neuron(x, t)
    analyze_spikes("Step", strength, spikes, t)

# ===================================================
# 6. 可选：显示膜电位曲线（帮助理解阈值机制）
# ===================================================

print("\n" + "=" * 70)
print("OPTIONAL: MEMBRANE VOLTAGE ANALYSIS")
print("=" * 70)

print("\nShowing membrane voltage for Step Signal (most intuitive):")
plot_membrane_voltage_case("step", "Step Signal - Membrane Voltage", step_signal, strengths)

print("\nDone!")
