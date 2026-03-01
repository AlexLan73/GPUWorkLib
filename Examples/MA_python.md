Давайте сделаем обновленный скрипт под ваш вариант: 20 нулей, затем 50 единиц подряд, затем 50 нулей (итог 120 точек). Я предоставлю обновленный Python-код, который вычисляет SMA(20), EMA(20), MMA(20), DEMA(20), TEMA(20), KAMA(20) и строит график на одной панели с подписями. Затем можно будет получить PNG прямо здесь.

Вот готовый скрипт:

python
import numpy as np
import matplotlib.pyplot as plt

def SMA(x, n):
    return np.convolve(x, np.ones(n)/n, mode='same')

def EMA(x, n):
    a = 2/(n+1)
    ema = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        if i == 0:
            ema[i] = x[i]
        else:
            ema[i] = a*x[i] + (1-a)*ema[i-1]
    return ema

def MMA(x, n):
    # Обычно MMA трактуют как SMA
    return SMA(x, n)

def DEMA(x, n):
    ema1 = EMA(x, n)
    ema2 = EMA(ema1, n)
    return 2*ema1 - ema2

def TEMA(x, n):
    ema1 = EMA(x, n)
    ema2 = EMA(ema1, n)
    ema3 = EMA(ema2, n)
    return 3*ema1 - 3*ema2 + ema3

def KAMA(x, n):
    # Упрощенная версия KAMA (для наглядности)
    kama = np.zeros_like(x, dtype=float)
    if len(x) == 0:
        return kama
    kama[0] = x[0]
    for i in range(1, len(x)):
        delta = abs(x[i] - kama[i-1])
        alpha = min(1.0, delta / (n if n > 0 else 1))
        kama[i] = kama[i-1] + (x[i] - kama[i-1]) * alpha
    return kama

def plot_all(signal, n=20):
    t = np.arange(len(signal))
    sma = SMA(signal, n)
    ema = EMA(signal, n)
    mma = MMA(signal, n)
    dema = DEMA(signal, n)
    tema = TEMA(signal, n)
    kama = KAMA(signal, n)

    plt.figure(figsize=(10,6))
    plt.plot(t, signal, label='Original', color='black', linewidth=1.5)
    plt.plot(t, sma, label=f'SMA({n})')
    plt.plot(t, ema, label=f'EMA({n})')
    plt.plot(t, mma, label=f'MMA({n})')
    plt.plot(t, dema, label=f'DEMA({n})')
    plt.plot(t, tema, label=f'TEMA({n})')
    plt.plot(t, kama, label=f'KAMA({n})')
    plt.title('Synthetic signal with moving averages (VariantA)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('signal_ma_variantA.png', dpi=150)
    plt.close()

def main():
    # 120 точек: 20 нулей, 50 единиц, 50 нулей
    signal = np.zeros(120, dtype=float)
    signal[20:70] = 1.0  # 50 единиц подряд
    plot_all(signal, 20)

if __name__ == '__main__':
    main()