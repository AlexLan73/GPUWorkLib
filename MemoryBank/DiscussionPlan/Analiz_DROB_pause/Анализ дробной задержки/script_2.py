
import numpy as np

# The problem: with 4 MHz BW and 12 MHz sampling, the beat frequency for 23.7 ns delay 
# is only ~948 Hz - less than 1 FFT bin (10 kHz for N=1200). 
# We need a wideband scenario. Let me use proper wideband LFM parameters.
# In real radar: BW = 500 MHz, but Fs=12 MHz can't handle that directly.
# The user works with ALREADY DECHIRPED data (stretch processing).
# After dechirp, beat frequency = chirp_rate * delay is low frequency.
# Let's use correct scenario.

# Real radar parameters
BW_rf = 500e6      # RF bandwidth 500 MHz
T_pulse = 20e-6    # pulse 20 us  
chirp_rate = BW_rf / T_pulse  # 2.5e13 Hz/s
fs = 12e6          # ADC sampling rate (after dechirp)
N = int(fs * T_pulse)  # 240 samples per pulse
Ts = 1.0 / fs

# For phased array: the delay comes from element spacing
# d = lambda/2 at 10 GHz => d = 0.015 m
# Max delay = d*sin(theta_max)/c ~ 0.015/3e8 = 50 ps for single element
# For 256 antennas traversing beam, delays range 0 to ~dozen ns

# Let's pick a meaningful fractional delay
true_delay_ns = 3.75   # nanoseconds (sub-sample: 0.045 samples at 12 MHz)  
true_delay_s = true_delay_ns * 1e-9
true_delay_samples = true_delay_s * fs

# Beat frequency for this delay (after dechirp)
f_beat_true = chirp_rate * true_delay_s
print(f"=== Radar Scenario (after dechirp/stretch processing) ===")
print(f"  RF BW = {BW_rf/1e6:.0f} MHz, T_pulse = {T_pulse*1e6:.0f} us")
print(f"  Chirp rate = {chirp_rate:.3e} Hz/s")
print(f"  fs = {fs/1e6:.0f} MHz, N = {N} samples")
print(f"  True delay = {true_delay_ns} ns = {true_delay_samples:.6f} samples")
print(f"  Beat freq = {f_beat_true:.3f} Hz")
print(f"  Beat freq / bin_width = {f_beat_true / (fs/N):.4f} bins")
print()

rng = np.random.default_rng(123)
n_arr = np.arange(N)
t = n_arr / fs

# After dechirp, signal is approximately:
# s_beat(t) = A * exp(j*2*pi*f_beat*t + j*phi0)
# where f_beat = chirp_rate * tau, phi0 = phase terms

# Let's model more realistically using the KNOWN analytic signal approach:
# Reference signal (no delay)
f0_bb = 0  # baseband center
s_ref = np.exp(1j * 2 * np.pi * (f0_bb * t + 0.5 * chirp_rate * t**2))

# Delayed signal 
t_delayed = t - true_delay_s
s_del = np.exp(1j * 2 * np.pi * (f0_bb * t_delayed + 0.5 * chirp_rate * t_delayed**2))

# Dechirped (heterodyne) signal
beat_clean = s_del * np.conj(s_ref)
# = exp(j*2*pi*(-chirp_rate*tau*t + 0.5*chirp_rate*tau^2))
# = exp(j*2*pi*f_beat*t) * exp(j*phi0)  where f_beat = -chirp_rate*tau

# Verify
beat_phase_clean = np.unwrap(np.angle(beat_clean))
slope_check = np.polyfit(t, beat_phase_clean, 1)
tau_check = -slope_check[0] / (2 * np.pi * chirp_rate)
print(f"  Verification (clean): tau = {tau_check*1e9:.6f} ns ✓")
print()

# Add noise
SNR_dB_values = [10, 20, 30, 40]
n_trials = 500

# CRB for dechirped signal
# After dechirp, we have a single tone of known frequency structure
# CRB for time delay of known wideband signal:
# var(tau) >= 1 / (8*pi^2 * beta_rms^2 * N * SNR)
# beta_rms for LFM with BW 500 MHz: beta_rms = BW/sqrt(12)
beta_rms = BW_rf / np.sqrt(12)
print(f"  RMS bandwidth = {beta_rms/1e6:.2f} MHz")

results_all = {}

for SNR_dB in SNR_dB_values:
    SNR_lin = 10**(SNR_dB/10)
    CRB_var = 1.0 / (8 * np.pi**2 * beta_rms**2 * N * SNR_lin)
    CRB_std_ps = np.sqrt(CRB_var) * 1e12
    
    errors = {
        'phase_slope': [],
        'xcorr_parabola': [],
        'fft_zp': [],
        'fft_parabola': [],
        'ml_newton': [],
    }
    
    for trial in range(n_trials):
        noise_power = 10**(-SNR_dB/10)
        noise = np.sqrt(noise_power/2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        s_rx = s_del + noise
        beat = s_rx * np.conj(s_ref)
        
        # ---- Method 1: Phase slope (МНК наклона фазы beat) ----
        beat_ph = np.unwrap(np.angle(beat))
        coeffs = np.polyfit(t, beat_ph, 1)
        tau_m1 = -coeffs[0] / (2 * np.pi * chirp_rate)
        errors['phase_slope'].append(tau_m1 - true_delay_s)
        
        # ---- Method 2: Cross-correlation + parabola ----
        # Use FFT-based correlation for speed
        S_rx = np.fft.fft(s_rx, 2*N)
        S_ref = np.fft.fft(s_ref, 2*N)
        xcorr = np.fft.ifft(S_rx * np.conj(S_ref))
        mag_xc = np.abs(xcorr)
        pk = np.argmax(mag_xc)
        
        # Handle circular indexing
        if pk > N:
            pk_lag = pk - 2*N
        else:
            pk_lag = pk
            
        ym1 = mag_xc[(pk-1) % (2*N)]
        y0  = mag_xc[pk]
        yp1 = mag_xc[(pk+1) % (2*N)]
        denom = ym1 - 2*y0 + yp1
        if abs(denom) > 1e-12:
            dp = 0.5 * (ym1 - yp1) / denom
        else:
            dp = 0
        tau_m2 = (pk_lag + dp) / fs
        errors['xcorr_parabola'].append(tau_m2 - true_delay_s)
        
        # ---- Method 3: FFT zero-padding (256x) ----
        ZP = 256
        N_fft = N * ZP
        Beat_fft = np.fft.fft(beat, N_fft)
        freqs_fft = np.fft.fftfreq(N_fft, Ts)
        pk_fft = np.argmax(np.abs(Beat_fft))
        f_beat_est = freqs_fft[pk_fft]
        tau_m3 = -f_beat_est / chirp_rate  # negative because beat = exp(-j*2pi*k*tau*t)
        errors['fft_zp'].append(tau_m3 - true_delay_s)
        
        # ---- Method 4: FFT(8x) + parabolic refinement ----
        N_fft4 = N * 8
        Beat_fft4 = np.fft.fft(beat, N_fft4)
        freqs_fft4 = np.fft.fftfreq(N_fft4, Ts)
        mag4 = np.abs(Beat_fft4)
        pk4 = np.argmax(mag4)
        ym1_4 = mag4[(pk4-1) % N_fft4]
        y0_4  = mag4[pk4]
        yp1_4 = mag4[(pk4+1) % N_fft4]
        denom4 = ym1_4 - 2*y0_4 + yp1_4
        if abs(denom4) > 1e-12:
            dp4 = 0.5 * (ym1_4 - yp1_4) / denom4
        else:
            dp4 = 0
        f_ref4 = freqs_fft4[pk4] + dp4 * (fs / N_fft4)
        tau_m4 = -f_ref4 / chirp_rate
        errors['fft_parabola'].append(tau_m4 - true_delay_s)
        
        # ---- Method 5: ML Newton ----
        # Initialize from FFT ZP
        tau_ml = tau_m3
        for it in range(30):
            t_m = t - tau_ml
            ph_m = 2 * np.pi * (f0_bb * t_m + 0.5 * chirp_rate * t_m**2)
            s_m = np.exp(1j * ph_m)
            res = s_rx - s_m
            inst_f = f0_bb + chirp_rate * t_m
            ds_dtau = -1j * 2 * np.pi * inst_f * s_m
            num = np.real(np.sum(res * np.conj(ds_dtau)))
            den = np.real(np.sum(np.abs(ds_dtau)**2))
            if den < 1e-20:
                break
            dtau = -num / den
            tau_ml += dtau
            if abs(dtau) < 1e-16:
                break
        errors['ml_newton'].append(tau_ml - true_delay_s)
    
    # Compute RMSE for each method
    results_all[SNR_dB] = {}
    for method_name, err_list in errors.items():
        err_arr = np.array(err_list)
        rmse = np.sqrt(np.mean(err_arr**2))
        bias = np.mean(err_arr)
        std = np.std(err_arr)
        results_all[SNR_dB][method_name] = {
            'rmse_ps': rmse * 1e12,
            'bias_ps': bias * 1e12,
            'std_ps': std * 1e12,
            'crb_ps': CRB_std_ps,
        }

# Print results
method_names_ru = {
    'phase_slope': '1. МНК фазового наклона',
    'xcorr_parabola': '2. Кросс-корр + парабола',
    'fft_zp': '3. FFT ZP (256x)',
    'fft_parabola': '4. FFT(8x) + парабола',
    'ml_newton': '5. ML/Newton',
}

print("\n" + "=" * 90)
print("РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ (500 реализаций)")
print("=" * 90)

for SNR_dB in SNR_dB_values:
    CRB_ps = results_all[SNR_dB]['phase_slope']['crb_ps']
    print(f"\n  SNR = {SNR_dB} dB, КРБ std = {CRB_ps:.2f} ps")
    print(f"  {'Метод':<30} {'RMSE (ps)':>10} {'Bias (ps)':>10} {'Std (ps)':>10} {'RMSE/CRB':>10}")
    print("  " + "-" * 75)
    for key in errors.keys():
        r = results_all[SNR_dB][key]
        print(f"  {method_names_ru[key]:<30} {r['rmse_ps']:>10.2f} {r['bias_ps']:>10.2f} {r['std_ps']:>10.2f} {r['rmse_ps']/r['crb_ps']:>10.2f}")
