"""
Model 1: Leaky Integrate-and-Fire (LIF) - 最简单的脉冲神经元模型
===================================================================

最简单的积分-阈值-复位模型

方程：
  dV/dt = (I - leak * V) / tau

特点：
- 计算快速
- 参数少
- 能展示基本阈值特性
- 不能产生复杂的放电模式
"""

import numpy as np
import matplotlib.pyplot as plt

# ==================== LIF 神经元 ====================

class LIF_Neuron:
    """
    Leaky Integrate-and-Fire 神经元
    
    参数：
      - tau: 时间常数
      - leak: 漏电系数
      - threshold: 阈值
      - V_reset: 复位电位
      - refractory_period: 不应期
    """
    
    def __init__(self, tau=10.0, leak=1.0, threshold=1.0, V_reset=-0.2, refractory_period=2.0):
        self.tau = tau
        self.leak = leak
        self.threshold = threshold
        self.V_reset = V_reset
        self.refractory_period = refractory_period
        self.V = 0.0
        self.last_spike_time = -1e9
    
    def step(self, I, t, dt):
        """
        单步积分
        I: 输入电流
        t: 当前时间
        dt: 时间步长
        """
        # 不应期检查
        if t - self.last_spike_time < self.refractory_period:
            self.V = self.V_reset
            return False
        
        # 膜电位更新
        self.V += (I - self.leak * self.V) * dt / self.tau
        
        # 阈值判断
        spike = False
        if self.V >= self.threshold:
            spike = True
            self.last_spike_time = t
            self.V = self.V_reset
        
        return spike


# ==================== 仿真 ====================

dt = 0.1
T = 500.0
t = np.arange(0, T, dt)

def simulate_lif(I_input, dt, tau=10.0):
    neuron = LIF_Neuron(tau=tau)
    V_trace = []
    spikes = []
    
    for i, t_i in enumerate(t):
        spike = neuron.step(I_input[i], t_i, dt)
        V_trace.append(neuron.V)
        if spike:
            spikes.append(t_i)
    
    return np.array(V_trace), np.array(spikes)


# ==================== 对比不同输入强度 ====================

def plot_lif_comparison():
    # 生成阶跃输入
    I = np.zeros_like(t)
    
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    fig.suptitle("LIF Model - Step Input with Different Strengths", fontsize=14, fontweight="bold")
    
    strengths = [0.5, 1.0, 2.0]
    
    for idx, strength in enumerate(strengths):
        I = np.zeros_like(t)
        I[t >= 150.0] = strength
        
        V_trace, spikes = simulate_lif(I, dt)
        
        # 左图：输入和膜电位
        ax1 = axes[idx, 0]
        ax1.plot(t, I, color="royalblue", linewidth=1.5, label="Input Current")
        ax1_2 = ax1.twinx()
        ax1_2.plot(t, V_trace, color="darkorange", linewidth=1.5, label="Membrane Voltage")
        ax1_2.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Threshold")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current", color="royalblue")
        ax1_2.set_ylabel("Voltage", color="darkorange")
        ax1.set_title(f"LIF - Strength = {strength}", fontweight="bold")
        ax1.grid(alpha=0.3)
        ax1.set_xlim(0, T)
        
        # 右图：脉冲
        ax2 = axes[idx, 1]
        if len(spikes) > 0:
            ax2.vlines(spikes, 0, 1, color="crimson", linewidth=2)
            ax2.scatter(spikes, np.ones_like(spikes), color="crimson", s=50)
        ax2.set_title(f"Output Spikes: {len(spikes)}", fontweight="bold")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Spike")
        ax2.set_ylim(-0.2, 1.5)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, T)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("LIF (Leaky Integrate-and-Fire) Model")
    print("=" * 70)
    plot_lif_comparison()
    print("Done!")
