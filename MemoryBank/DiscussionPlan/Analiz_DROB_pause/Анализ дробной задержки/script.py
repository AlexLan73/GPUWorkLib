
import numpy as np

# =============================================
# SIMULATION: Comparison of fractional delay estimation methods for LFM radar
# =============================================

fs = 12e6          # sampling frequency
f0 = 9.0e9         # start frequency (example X-band)
f1 = 9.5e9         # end frequency
BW = f1 - f0       # 500 MHz bandwidth
T_pulse = 100e-6   # pulse duration 100 us
chirp_rate = BW / T_pulse  # Hz/s

N = int(fs * T_pulse)  # 1200 samples per pulse
Ts = 1.0 / fs

# True fractional delay in samples
true_delay_ns = 23.7  # nanoseconds
true_delay_s = true_delay_ns * 1e-9
true_delay_samples = true_delay_s * fs  # in samples

print(f"Parameters:")
print(f"  fs = {fs/1e6:.0f} MHz, BW = {BW/1e6:.0f} MHz, T = {T_pulse*1e6:.0f} us")
print(f"  N = {N} samples/pulse")
print(f"  True delay = {true_delay_ns} ns = {true_delay_samples:.6f} samples")
print(f"  Chirp rate k = {chirp_rate:.3e} Hz/s")
print()

# Generate reference LFM signal (analytic)
n = np.arange(N)
t_ref = n / fs
phase_ref = 2 * np.pi * (f0 * t_ref + 0.5 * chirp_rate * t_ref**2)
s_ref = np.exp(1j * phase_ref)

# Generate delayed LFM signal (analytic - exact)
t_del = (n / fs) - true_delay_s
phase_del = 2 * np.pi * (f0 * t_del + 0.5 * chirp_rate * t_del**2)
s_delayed = np.exp(1j * phase_del)

# Add noise
SNR_dB = 20
noise_power = 10**(-SNR_dB/10)
noise = np.sqrt(noise_power/2) * (np.random.randn(N) + 1j * np.random.randn(N))
s_noisy = s_delayed + noise

print("=" * 70)
print("METHOD 1: Spectral (Analytic) Phase Method")
print("=" * 70)
# Since we know the LFM model: compute exact phase at each sample
# and compare with received signal phase
phase_received = np.angle(s_noisy)
phase_model_nodelay = 2 * np.pi * (f0 * t_ref + 0.5 * chirp_rate * t_ref**2)
phase_model_nodelay_wrapped = np.angle(np.exp(1j * phase_model_nodelay))

# Phase difference
dphi = np.angle(s_noisy * np.conj(s_ref))

# For LFM with delay tau:
# delta_phase = 2*pi*(f0*tau + chirp_rate * t * tau - 0.5*chirp_rate*tau^2)
# This is LINEAR in t: delta_phase = a + b*t
# where b = 2*pi*chirp_rate*tau, a = 2*pi*(f0*tau - 0.5*chirp_rate*tau^2)
# Least squares fit to find tau from slope b

# Unwrap phase difference
dphi_unwrap = np.unwrap(dphi)

# Linear fit: dphi = a + b*t
A_mat = np.vstack([t_ref, np.ones(N)]).T
result = np.linalg.lstsq(A_mat, dphi_unwrap, rcond=None)
b_fit, a_fit = result[0]

# From b = 2*pi*chirp_rate*tau => tau = b / (2*pi*chirp_rate)
tau_from_slope = b_fit / (2 * np.pi * chirp_rate)

# From a = 2*pi*(f0*tau - 0.5*k*tau^2) ≈ 2*pi*f0*tau for small tau
tau_from_intercept = a_fit / (2 * np.pi * f0)

print(f"  Phase slope b = {b_fit:.6f} rad/s")
print(f"  tau from slope  = {tau_from_slope*1e9:.6f} ns (true: {true_delay_ns} ns)")
print(f"  tau from intercept = {tau_from_intercept*1e9:.6f} ns")
print(f"  Error (slope)   = {abs(tau_from_slope - true_delay_s)*1e12:.3f} ps")
print(f"  Error (intercept) = {abs(tau_from_intercept - true_delay_s)*1e12:.3f} ps")
print()

print("=" * 70)
print("METHOD 2: Cross-correlation + Parabolic Interpolation")
print("=" * 70)
xcorr = np.correlate(s_noisy, s_ref, mode='full')
lags = np.arange(-N+1, N)
peak_idx = np.argmax(np.abs(xcorr))

# Parabolic interpolation around peak
y_m1 = np.abs(xcorr[peak_idx - 1])
y_0  = np.abs(xcorr[peak_idx])
y_p1 = np.abs(xcorr[peak_idx + 1])
delta_parabolic = 0.5 * (y_m1 - y_p1) / (y_m1 - 2*y_0 + y_p1)
delay_parabolic_samples = lags[peak_idx] + delta_parabolic
delay_parabolic_s = delay_parabolic_samples / fs

print(f"  Peak at lag = {lags[peak_idx]}")
print(f"  Parabolic delta = {delta_parabolic:.6f} samples")
print(f"  Estimated delay = {delay_parabolic_s*1e9:.6f} ns (true: {true_delay_ns} ns)")
print(f"  Error = {abs(delay_parabolic_s - true_delay_s)*1e9:.6f} ns = {abs(delay_parabolic_s - true_delay_s)*1e12:.3f} ps")
print()

print("=" * 70)
print("METHOD 3: FFT Zero-Padding (sub-bin interpolation)")
print("=" * 70)
# Heterodyne (dechirp): multiply received by conjugate of reference
beat = s_noisy * np.conj(s_ref)

# Zero-pad and FFT
ZP = 64  # zero-padding factor
N_fft = N * ZP
Beat_fft = np.fft.fft(beat, N_fft)
freqs = np.fft.fftfreq(N_fft, Ts)

# Peak detection
peak_bin = np.argmax(np.abs(Beat_fft[:N_fft//2]))
f_beat = freqs[peak_bin]

# For dechirp: f_beat = chirp_rate * tau
tau_fft = f_beat / chirp_rate
print(f"  Zero-pad factor = {ZP}x (N_fft = {N_fft})")
print(f"  Beat frequency = {f_beat:.3f} Hz")
print(f"  Estimated delay = {tau_fft*1e9:.6f} ns (true: {true_delay_ns} ns)")
print(f"  Error = {abs(tau_fft - true_delay_s)*1e9:.6f} ns = {abs(tau_fft - true_delay_s)*1e12:.3f} ps")
print()

print("=" * 70)
print("METHOD 4: Phase of Cross-Spectral Density (GCC-PHAT like)")
print("=" * 70)
S1 = np.fft.fft(s_ref, 2*N)
S2 = np.fft.fft(s_noisy, 2*N)
cross_spectrum = S2 * np.conj(S1)

# Phase slope in frequency domain
freqs_cs = np.fft.fftfreq(2*N, Ts)
phase_cs = np.unwrap(np.angle(cross_spectrum))

# Use central part of spectrum (avoid edges)
valid = (np.abs(freqs_cs) > 0.05 * fs) & (np.abs(freqs_cs) < 0.45 * fs)
A_cs = np.vstack([2 * np.pi * freqs_cs[valid], np.ones(np.sum(valid))]).T
result_cs = np.linalg.lstsq(A_cs, phase_cs[valid], rcond=None)
tau_cs = -result_cs[0][0]  # negative because phase = -2*pi*f*tau

print(f"  Cross-spectral phase slope fit")
print(f"  Estimated delay = {tau_cs*1e9:.6f} ns (true: {true_delay_ns} ns)")
print(f"  Error = {abs(tau_cs - true_delay_s)*1e9:.6f} ns = {abs(tau_cs - true_delay_s)*1e12:.3f} ps")
print()

print("=" * 70)
print("METHOD 5: MNK (Least Squares) Phase Fitting to Model")
print("=" * 70)
# Direct model fitting: minimize |s_noisy - s_model(tau)|^2
# Gradient-based: use Newton's method
# For LFM: ds/dtau = -j*2*pi*(f0 + chirp_rate*t) * s_model(tau)

tau_est = 0.0  # initial guess
for iteration in range(20):
    t_model = t_ref - tau_est
    phase_model = 2 * np.pi * (f0 * t_model + 0.5 * chirp_rate * t_model**2)
    s_model = np.exp(1j * phase_model)
    
    # Residual
    residual = s_noisy - s_model
    
    # Gradient: d(residual)/d(tau) 
    inst_freq = f0 + chirp_rate * t_model
    ds_dtau = 1j * 2 * np.pi * inst_freq * s_model  # note sign
    
    # Newton step: delta_tau = Re(sum(residual * conj(ds_dtau))) / Re(sum(|ds_dtau|^2))
    numerator = np.real(np.sum(residual * np.conj(-ds_dtau)))
    denominator = np.real(np.sum(np.abs(ds_dtau)**2))
    delta_tau = numerator / denominator
    
    tau_est += delta_tau
    
    if abs(delta_tau) < 1e-15:
        break

print(f"  Converged in {iteration+1} iterations")
print(f"  Estimated delay = {tau_est*1e9:.6f} ns (true: {true_delay_ns} ns)")
print(f"  Error = {abs(tau_est - true_delay_s)*1e9:.6f} ns = {abs(tau_est - true_delay_s)*1e12:.3f} ps")
print()

# =============================================
# CRB Computation
# =============================================
print("=" * 70)
print("Cramér-Rao Bound (CRB)")
print("=" * 70)
# For time delay estimation of known signal in AWGN:
# CRB(tau) = 1 / (SNR * (2*pi)^2 * beta_rms^2 * N)
# where beta_rms is the RMS bandwidth of the signal

# For LFM: beta_rms^2 = f0^2 + f0*BW + BW^2/3 (approx for baseband chirp)
# More precisely: beta_rms^2 = (1/T) * integral of (f_inst - f_mean)^2 dt
f_mean = f0 + BW/2
# instantaneous freq: f(t) = f0 + chirp_rate * t
# variance of inst freq over pulse duration
beta_rms_sq = (BW**2) / 12  # variance of uniform distribution over [0, BW]
beta_rms = np.sqrt(beta_rms_sq)

SNR_linear = 10**(SNR_dB/10)
CRB_tau = 1.0 / (8 * np.pi**2 * beta_rms_sq * N * SNR_linear)
CRB_tau_ns = np.sqrt(CRB_tau) * 1e9

print(f"  RMS bandwidth beta_rms = {beta_rms/1e6:.2f} MHz")
print(f"  CRB(tau) variance = {CRB_tau:.4e} s^2")
print(f"  CRB std(tau) = {CRB_tau_ns:.6f} ns = {np.sqrt(CRB_tau)*1e12:.3f} ps")
print()

print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
methods = [
    ("1. Spectral Phase (MNK slope)", abs(tau_from_slope - true_delay_s)),
    ("2. Cross-corr + Parabolic", abs(delay_parabolic_s - true_delay_s)),
    ("3. Dechirp + FFT ZP (64x)", abs(tau_fft - true_delay_s)),
    ("4. Cross-spectral phase slope", abs(tau_cs - true_delay_s)),
    ("5. ML/Newton (model fitting)", abs(tau_est - true_delay_s)),
]

print(f"{'Method':<35} {'Error (ps)':>12} {'Error (ns)':>12} {'vs CRB':>10}")
print("-" * 72)
for name, err in methods:
    print(f"{name:<35} {err*1e12:>12.3f} {err*1e9:>12.6f} {err/np.sqrt(CRB_tau):>10.2f}x")
print(f"{'CRB std(tau)':<35} {np.sqrt(CRB_tau)*1e12:>12.3f} {CRB_tau_ns:>12.6f} {'1.00x':>10}")
