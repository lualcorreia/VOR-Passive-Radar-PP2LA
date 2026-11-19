"""Diagnóstico: procura vestígios do VOR (portadora, ±30 Hz, ±9960 Hz, ident 1020 Hz) em ±16 kHz.

Uso: .venv\\Scripts\\python.exe tools\\vor_scan.py [freq_MHz] [segundos] [ganho_dB]
Usa a mesma cadeia do app (LO = f + fs/4 → DDC → 32 kS/s) e faz periodograma
médio (Welch, ~1 Hz de resolução). Lista os picos mais proeminentes.
"""
import sys
import time

import numpy as np
from scipy import signal
from rtlsdr import RtlSdr

sys.path.insert(0, ".")
from vorradar.dsp import DDC  # noqa: E402

f_vor = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 115.9e6
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
gain = float(sys.argv[3]) if len(sys.argv) > 3 else 29.7
fs = 1.024e6
sdr = RtlSdr()
sdr.sample_rate = fs
sdr.center_freq = f_vor + fs / 4
sdr.gain = gain
sdr.read_samples(131072)
ddc = DDC(fs)
wide = []
n_need = int(dur * fs)
n = 0
t0 = time.time()
while n < n_need:
    x = sdr.read_samples(262144)
    n += len(x)
    wide.append(ddc.full_to_wide(x.astype(np.complex64)))
sdr.close()
w = np.concatenate(wide)
f, P = signal.welch(w, fs=32000, nperseg=32768, noverlap=16384, return_onesided=False, window="hann")
i = np.argsort(f)
f, P = f[i], P[i]
Pd = 10 * np.log10(P)
med = signal.medfilt(Pd, 301)
prom = Pd - med
print(f"Captura: {dur:.0f} s, ganho {sdr.gain if False else gain} dB, resolução {f[1] - f[0]:.2f} Hz")
print(f"Piso mediano: {np.median(Pd):.1f} dBFS/Hz")
pk, _ = signal.find_peaks(prom, height=6, distance=5)
pk = pk[np.argsort(-prom[pk])][:15]
print("Picos mais proeminentes (offset em relação a 115,9 MHz nominal):")
for j in sorted(pk, key=lambda k: f[k]):
    print(f"  {f[j]:+10.2f} Hz   {prom[j]:5.1f} dB acima do piso local")


def band(lo, hi):
    s = (f >= lo) & (f <= hi)
    k = np.argmax(prom[s])
    return f[s][k], prom[s][k]


for name, lo, hi in (("portadora (±300 Hz)", -300, 300), ("subportadora +9960 (±500)", 9460, 10460),
                     ("subportadora −9960 (±500)", -10460, -9460), ("ident +1020 (±300)", 720, 1320)):
    fx, px = band(lo, hi)
    print(f"{name:28s}: máx {px:5.1f} dB em {fx:+.1f} Hz")
