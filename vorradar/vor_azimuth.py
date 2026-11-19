"""[EXPERIMENTAL] Radial da aeronave em relação ao VOR a partir do próprio eco.

O sinal VOR que ilumina a aeronave carrega a fase de 30 Hz correspondente à
radial em que ela está. Um receptor de bordo mede:
    radial = φ(30 Hz da FM na subportadora 9960 Hz) − φ(30 Hz da AM)
(CVOR: referência = FM, variável = AM. No DVOR os papéis e o sentido de
rotação se invertem, de modo que o mesmo cálculo dá a mesma radial.)

Aqui o eco é isolado deslocando o fluxo largo (32 kS/s) pela frequência da
trilha (com a inclinação df/dt), recuperando a portadora do eco com um passa-
baixas de 5 Hz e fazendo detecção AM síncrona. Depois:
  * AM 30 Hz → lock-in em 30 Hz;
  * subportadora 9960 ± 700 Hz → sinal analítico → frequência instantânea →
    lock-in em 30 Hz.
Todos os filtros são de fase zero (filtfilt), então não há atraso de grupo
diferencial entre os dois ramos (1 ms de erro = 10,8°).

LIMITAÇÕES (por isso é experimental):
  * precisa de C/N0 alto no eco (≳ 35–40 dB-Hz) — o limiar da FM é severo;
  * se o sinal direto estiver presente, a subportadora dele (±9960 Hz, desvio
    ±480 Hz) se sobrepõe à do eco (o Doppler é de só dezenas a centenas de Hz),
    contaminando a fase da FM. Funciona melhor exatamente no caso sem visada;
  * a radial é magnética conforme o alinhamento da estação; use a calibração.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from .config import FS_WIDE

F_SC = 9960.0
F_MOD = 30.0
OFF_FREQS = (21.0, 25.0, 35.0, 39.0)


def _lockin(x: np.ndarray, t: np.ndarray, f: float, w: np.ndarray) -> complex:
    return complex(np.sum(w * x * np.exp(-2j * math.pi * f * t)) / np.sum(w))


def _tone_snr(x, t, w) -> tuple[complex, float]:
    p = _lockin(x, t, F_MOD, w)
    off = [abs(_lockin(x, t, fo, w)) for fo in OFF_FREQS]
    rms_off = math.sqrt(np.mean(np.square(off))) or 1e-30
    return p, 20 * math.log10(max(abs(p), 1e-30) / rms_off)


class VorAzimuthEstimator:
    def __init__(self, fs: float = FS_WIDE):
        self.fs = fs
        self.sos_sc = signal.butter(4, [F_SC - 700, F_SC + 700], btype="bandpass", fs=fs, output="sos")
        self.dec = 32
        self.sos_c = signal.butter(2, 5.0, fs=fs / self.dec, output="sos")

    def estimate(self, x: np.ndarray, f_echo_hz: float, slope_hz_s: float = 0.0,
                 cal_deg: float = 0.0, min_snr_db: float = 10.0) -> dict:
        """x: fluxo largo complexo; f_echo_hz: freq. do eco no centro do trecho (eixo do fluxo)."""
        n = len(x)
        if n < int(self.fs):
            return {"valid": False, "msg": "trecho curto"}
        t = np.arange(n) / self.fs
        tm = t - t[n // 2]
        y = x.astype(np.complex128) * np.exp(-2j * math.pi * (f_echo_hz * tm + 0.5 * slope_hz_s * tm * tm))
        # portadora do eco (≤5 Hz), estimada a 1 kS/s e interpolada
        m = self.dec
        nb = n // m
        yd = y[: nb * m].reshape(nb, m).mean(axis=1)
        cd = signal.sosfiltfilt(self.sos_c, yd)
        tc = (np.arange(nb) * m + (m - 1) / 2) / self.fs
        c = np.interp(t, tc, cd.real) + 1j * np.interp(t, tc, cd.imag)
        z = y * np.conj(c) / np.maximum(np.abs(c), 1e-15)
        r = z.real - np.mean(z.real)
        # ramo AM 30 Hz
        trim = slice(int(0.05 * n), int(0.95 * n))
        tt = t[trim]
        w = np.hanning(len(tt))
        p_am, snr_am = _tone_snr(r[trim], tt, w)
        # ramo FM (subportadora)
        sc = signal.sosfiltfilt(self.sos_sc, r)
        ph = np.unwrap(np.angle(signal.hilbert(sc)))
        finst = np.gradient(ph) * self.fs / (2 * math.pi) - F_SC
        fi = finst[trim] - np.mean(finst[trim])
        p_fm, snr_fm = _tone_snr(fi, tt, w)
        radial = (math.degrees(math.atan2(p_fm.imag, p_fm.real) - math.atan2(p_am.imag, p_am.real))
                  + cal_deg) % 360.0
        valid = snr_am >= min_snr_db and snr_fm >= min_snr_db
        return {"valid": bool(valid), "radial_deg": radial, "snr_am_db": snr_am, "snr_fm_db": snr_fm,
                "fm_dev_hz": 2 * abs(p_fm), "am_depth": 2 * abs(p_am) / max(np.mean(np.abs(c)), 1e-15),
                "msg": "ok" if valid else "SNR insuficiente"}
