"""Teste rÃ¡pido de captura IQ do RTL-SDR (abertura, taxa efetiva e perdas).

Uso:  .venv\\Scripts\\python.exe tools\\iq_capture_test.py [freq_MHz] [segundos]

DetecÃ§Ã£o de perdas: lÃª blocos pelo modo assÃ­ncrono do librtlsdr e compara o
total de amostras recebidas com o esperado pelo relÃ³gio de parede (fs * t).
RazÃ£o < ~0,995 indica buffers USB descartados. TambÃ©m verifica saturaÃ§Ã£o do ADC.
"""
import sys
import time
import numpy as np
from rtlsdr import RtlSdr

freq = float(sys.argv[1]) * 1e6 if len(sys.argv) > 1 else 115.9e6
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
fs = 1.024e6
blk = 256 * 1024  # ~0,25 s por callback

from rtlsdr.librtlsdr import librtlsdr; print(f"Dispositivos encontrados: {librtlsdr.rtlsdr_get_device_count()}")
sdr = RtlSdr()
sdr.sample_rate = fs
sdr.center_freq = freq
try:
    sdr.freq_correction = 1
except Exception as e:
    print("freq_correction:", e)
sdr.gain = 40.2
print(f"Tuner: {sdr.get_tuner_type()}  fs={sdr.sample_rate:.0f}  fc={sdr.center_freq:.0f}  "
      f"ganho={sdr.gain} dB")
print(f"Ganhos vÃ¡lidos: {sdr.valid_gains_db}")

state = {"total": 0, "n": 0, "peak": 0.0, "pwr": [], "t0": None}


def cb(samples, ctx):
    if state["t0"] is None:
        state["t0"] = time.perf_counter()
        return  # descarta o primeiro bloco (transitÃ³rio de partida)
    state["total"] += len(samples)
    state["n"] += 1
    state["peak"] = max(state["peak"], float(np.max(np.abs(samples.real))),
                        float(np.max(np.abs(samples.imag))))
    state["pwr"].append(float(np.mean(np.abs(samples) ** 2)))
    if time.perf_counter() - state["t0"] >= dur:
        sdr.cancel_read_async()


sdr.read_samples_async(cb, num_samples=blk)
elapsed = time.perf_counter() - state["t0"]
sdr.close()

# o 1Âº bloco contado chega ~1 bloco apÃ³s t0; corrige a janela de tempo
expected = fs * elapsed
ratio = state["total"] / expected
print(f"DuraÃ§Ã£o: {elapsed:.2f} s   blocos: {state['n']}")
print(f"Amostras recebidas: {state['total']}   esperadas: {expected:.0f}   razÃ£o: {ratio:.4f}")
print(f"Pico ADC: {state['peak']:.3f}   potÃªncia mÃ©dia: "
      f"{10*np.log10(np.mean(state['pwr'])):.1f} dBFS")
print("RESULTADO:", "OK â€” sem perdas detectÃ¡veis" if ratio > 0.99
      else f"ATENÃ‡ÃƒO â€” ~{(1-ratio)*100:.2f}% perdidas")
if state["peak"] > 0.98:
    print("AVISO: ADC prÃ³ximo de saturar; reduza o ganho.")
