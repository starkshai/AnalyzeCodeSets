import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端, 不弹窗
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数设置 ====================
Fs = 40e6          # ADC采样率 40MHz
Fsignal = 9.92e6   # 输入信号频率 9.92MHz (低于奈奎斯特频率20MHz, 无混叠)
N = 128            # 采样点数 (符合需求 N=128)
Vpp_range = 1.0    # 动态范围 ±0.5V, 峰峰值1V
ADC_bits = 16      # 16位ADC
R_load = 50        # 负载阻抗 50Ω (dBm标准)
noise_level = 0.01 # 1% 噪声(幅度)
phi0 = np.pi / 4   # 正弦信号初始相位 (rad), 可改变: 0, pi/4, pi/2 等
phase_noise_std = 0.02  # 相位噪声标准差 (rad), 每采样点独立相位抖动

# ==================== 1. 产生采样信号 ====================
# 采样时间序列
t = np.arange(N) / Fs

# 信号幅度: 动态范围±0.5V, 正弦波幅度设为0.5V(满量程)
# 这样可以充分利用ADC量程, 实际可根据需要调整
A_signal = 0.5

# 相位噪声建模: 随机游走(布朗噪声) + 白相位抖动
# 随机游走模拟振荡器频率源的长期相位漂移(1/f^2 噪声)
# 白相位抖动模拟采样时刻的瞬时相位扰动
phase_walk = np.cumsum(np.random.randn(N) * phase_noise_std * 0.3)  # 随机游走
phase_jitter = np.random.randn(N) * phase_noise_std                   # 白相位抖动
phase_noise = phase_walk + phase_jitter

# 带初始相位和相位噪声的正弦信号
# 9.92MHz < Fs/2 = 20MHz, 满足奈奎斯特采样定理, 不发生混叠
sine_theory = A_signal * np.sin(2 * np.pi * Fsignal * t + phi0 + phase_noise)

# 信号频率与奈奎斯特频率的对比 (无混叠, F_alias 即信号频率本身)
F_alias = abs(Fsignal - round(Fsignal / Fs) * Fs)
print(f"信号频率: {Fsignal/1e6:.2f} MHz (奈奎斯特频率 {Fs/2/1e6:.1f} MHz, 无混叠)")
print(f"初始相位 phi0 = {phi0/np.pi:.3f}π rad")
print(f"相位噪声标准差 = {phase_noise_std} rad ({np.degrees(phase_noise_std):.2f}°)")
print(f"相位噪声峰峰值 ≈ {np.ptp(phase_noise):.4f} rad ({np.degrees(np.ptp(phase_noise)):.2f}°)")

# ==================== 2. ADC量化 (16位, ±0.5V) ====================
LSB = Vpp_range / (2**ADC_bits)  # 量化步长
print(f"LSB (量化步长): {LSB*1e3:.4f} mV")

# 量化到16位ADC码值, 再转回电压
adc_codes = np.round(sine_theory / LSB)
adc_codes = np.clip(adc_codes, -(2**(ADC_bits-1)), 2**(ADC_bits-1) - 1)
signal_quantized = adc_codes * LSB

# ==================== 3. 加入1%噪声 ====================
# 噪声幅度为信号幅度的1%
noise = noise_level * A_signal * np.random.randn(N)
signal_noisy = signal_quantized + noise

# ==================== 4. 计算VRMS和dBm ====================
# VRMS = sqrt(mean(v^2))
Vrms = np.sqrt(np.mean(signal_noisy**2))
print(f"VRMS = {Vrms*1e3:.4f} mV")

# dBm: P = Vrms^2 / R, dBm = 10*log10(P/1mW)
P_watts = Vrms**2 / R_load
P_dBm = 10 * np.log10(P_watts / 1e-3)
print(f"信号功率 = {P_dBm:.2f} dBm (R={R_load}Ω)")

# ==================== 5. 绘制时域波形 ====================
plt.figure(figsize=(12, 9))

# 子图1: 时域波形
plt.subplot(3, 1, 1)
plt.plot(t * 1e9, signal_noisy, 'b-o', markersize=3, label='带噪声量化信号')
plt.plot(t * 1e9, signal_quantized, 'r--', linewidth=1, label='量化信号(无加性噪声)')
plt.xlabel('时间 (ns)')
plt.ylabel('电压 (V)')
plt.title(f'ADC采样时域波形 (Fs={Fs/1e6:.1f}MHz, Fsignal={Fsignal/1e6:.2f}MHz, '
          f'N={N}, phi0={phi0/np.pi:.2f}π)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([-0.6, 0.6])

# 子图2: 相位噪声
plt.subplot(3, 1, 2)
plt.plot(t * 1e9, np.degrees(phase_noise), 'm-', linewidth=1.5,
         label=f'相位噪声 (σ={np.degrees(phase_noise_std):.1f}°)')
plt.xlabel('时间 (ns)')
plt.ylabel('相位 (°)')
plt.title('相位噪声 (随机游走 + 白相位抖动)')
plt.legend()
plt.grid(True, alpha=0.3)

# ==================== 6. FFT频谱分析 ====================
# 使用汉宁窗减少频谱泄漏 (9.92MHz非整周期采样, 泄漏更明显)
window = np.hanning(N)
signal_windowed = signal_noisy * window

# 计算FFT
fft_result = np.fft.fft(signal_windowed)
fft_mag = np.abs(fft_result) / N  # 归一化幅度

# 频率轴 (0 ~ Fs)
freq_axis = np.fft.fftfreq(N, d=1/Fs)

# 只取正频率部分 (0 ~ Fs/2)
half_N = N // 2
freq_pos = freq_axis[:half_N]
fft_mag_pos = fft_mag[:half_N]

# 由于使用了汉宁窗, 需要进行窗补偿 (coherent gain = 0.5 for Hann)
coherent_gain = np.mean(window)
fft_mag_pos_calibrated = fft_mag_pos / coherent_gain

# 将频谱幅度转换为dBm
# 对于实信号, FFT幅度M对应正频率分量幅度, 信号总功率P = 2*M^2 / R
# (因子2是因为实信号功率均分在正负频率)
fft_mag_dBm = 10 * np.log10((2 * fft_mag_pos_calibrated**2) / (R_load * 1e-3) + 1e-30)

# ==================== 7. 查找9.92MHz处的峰值 ====================
# 找到最接近9.92MHz的频点
target_freq = 9.92e6
freq_resolution = Fs / N  # 频率分辨率
print(f"频率分辨率: {freq_resolution/1e3:.1f} kHz")

# 9.92MHz对应的频点索引 (9.92MHz/312.5kHz = 31.74, 不在bin中心)
idx_target = int(round(target_freq / freq_resolution))
print(f"9.92MHz对应FFT索引: {idx_target} (对应频率 {freq_pos[idx_target]/1e6:.3f} MHz)")

# 在目标bin邻域内搜索真实峰值 (9.92MHz不在bin中心, 必须邻域搜索)
search_range = 2
lo = max(0, idx_target - search_range)
hi = min(len(fft_mag_dBm) - 1, idx_target + search_range)
idx_peak = lo + int(np.argmax(fft_mag_dBm[lo:hi + 1]))
peak_mag_dBm = fft_mag_dBm[idx_peak]

print(f"\n========== 9.92MHz处频谱峰值 ==========")
print(f"峰值bin索引: {idx_peak}")
print(f"峰值频率: {freq_pos[idx_peak]/1e6:.3f} MHz")
print(f"峰值功率: {peak_mag_dBm:.2f} dBm")

# 全频谱最大值
idx_max = np.argmax(fft_mag_dBm)
print(f"\n全频谱最大峰值:")
print(f"频率: {freq_pos[idx_max]/1e6:.3f} MHz")
print(f"功率: {fft_mag_dBm[idx_max]:.2f} dBm")

# ==================== 8. 绘制频谱 ====================
plt.subplot(3, 1, 3)
freq_mhz = freq_pos / 1e6
plt.plot(freq_mhz, fft_mag_dBm, 'g-', linewidth=1.5)
plt.plot(freq_mhz[idx_peak], peak_mag_dBm, 'ro', markersize=10,
         label=f'9.92MHz峰值: {peak_mag_dBm:.2f} dBm @ {freq_pos[idx_peak]/1e6:.3f} MHz')
plt.xlabel('频率 (MHz)')
plt.ylabel('功率 (dBm)')
plt.title(f'FFT频谱 (输入信号 {Fsignal/1e6:.2f}MHz, 无混叠, N={N}, 分辨率{freq_resolution/1e3:.1f}kHz)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, Fs/2/1e6])

plt.tight_layout()
plt.savefig('adc_simulation_v2_result.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图像已保存为: adc_simulation_v2_result.png")
