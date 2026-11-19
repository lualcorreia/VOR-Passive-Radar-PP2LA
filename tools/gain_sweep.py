"""Varredura de ganho: saturação do ADC e potência de banda larga por ganho.

Uso: .venv\\Scripts\\python.exe tools\\gain_sweep.py [freq_MHz]
Sintoniza o LO em f_VOR + fs/4 (como o app faz) e mede, para cada ganho,
o pico do ADC, a fração de amostras clipadas e a potência média (dBFS).
"""
import sys
import numpy as np
from rtlsdr import RtlSdr

f_vor = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 115.9e6
fs = 1.024e6
sdr = RtlSdr()
sdr.sample_rate = fs
sdr.center_freq = f_vor + fs / 4
print("set_bias_tee disponível:", hasattr(sdr, "set_bias_tee"))
sdr.read_samples(65536)
for g in [0.0, 9.7, 19.7, 25.4, 29.7, 33.8, 37.2, 40.2, 44.5, 49.6]:
    sdr.gain = g
    sdr.read_samples(65536)  # descarta transitório
    x = sdr.read_samples(512 * 1024)
    pk = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))
    clip = np.mean((np.abs(x.real) > 0.99) | (np.abs(x.imag) > 0.99)) * 100
    # potência no canal VOR (±10 kHz em torno de -fs/4) vs mediana da banda
    X = np.fft.fftshift(np.abs(np.fft.fft(x[:262144] * np.hanning(262144))) ** 2)
    f = np.fft.fftshift(np.fft.fftfreq(262144, 1 / fs))
    ch = (np.abs(f + fs / 4) < 10e3)
    rel = 10 * np.log10(np.max(X[ch]) / np.median(X))
    print(f"ganho {sdr.gain:5.1f} dB | pico {pk:.3f} | clip {clip:6.3f}% | "
          f"P {10*np.log10(np.mean(np.abs(x)**2)):6.1f} dBFS | "
          f"pico canal VOR vs mediana: {rel:5.1f} dB")
sdr.close()
