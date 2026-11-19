"""Pipeline DSP: conversão digital (DDC), análise espectral, fundo, referência e CFAR.

Cadeia (entrada típica 1,024 MS/s, LO sintonizado em f_VOR + fs/4):
    mistura por +fs/4 (multiplicação por j^n, sem custo)  → VOR em 0 Hz
    FIR ↓8   → 128 kS/s
    FIR ↓4   →  32 kS/s  (fluxo "largo": gravação IQ e módulo de azimute)
    NCO fino (offset em Hz configurável)
    FIR ↓8   →   4 kS/s  (fluxo "estreito": análise Doppler ±2 kHz)
Todos os FIR são de fase linear (Kaiser, ~70 dB de rejeição) e implementados
como decimadores polifásicos com estado entre blocos (nenhuma amostra perdida).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal, stats

from .config import FS_WIDE, FS_NB

_J4 = np.array([1, 1j, -1, -1j], dtype=np.complex64)


def design_lowpass(fs: float, f_pass: float, f_stop: float, atten_db: float = 70.0) -> np.ndarray:
    numtaps, beta = signal.kaiserord(atten_db, (f_stop - f_pass) / (fs / 2))
    numtaps |= 1  # ímpar → atraso de grupo inteiro
    return signal.firwin(numtaps, (f_pass + f_stop) / 2, window=("kaiser", beta), fs=fs)


class FirDecimator:
    """FIR + decimação por D, calculando só as saídas necessárias, com estado."""

    def __init__(self, taps: np.ndarray, decim: int):
        self.taps = np.asarray(taps, dtype=np.float64)
        self.h = self.taps[::-1].astype(np.complex64)
        self.L = len(self.taps)
        self.D = int(decim)
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.L - 1, np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        xx = np.concatenate((self.hist, np.asarray(x, dtype=np.complex64)))
        n_win = len(xx) - self.L + 1
        if n_win <= 0:
            self.hist = xx
            return np.zeros(0, np.complex64)
        n_out = (n_win - 1) // self.D + 1
        win = sliding_window_view(xx, self.L)[: (n_out - 1) * self.D + 1: self.D]
        y = win @ self.h
        self.hist = xx[n_out * self.D:]
        return y.astype(np.complex64, copy=False)


class DDC:
    """Conversor digital descendente multiestágio (ver docstring do módulo)."""

    def __init__(self, fs_in: float = 1.024e6):
        self.fs_in = fs_in
        d12 = fs_in / FS_WIDE
        if abs(d12 - round(d12)) > 1e-9 or int(round(d12)) % 4:
            raise ValueError(f"fs_in={fs_in} precisa ser múltiplo de 4×{FS_WIDE:.0f}")
        d12 = int(round(d12))
        d1 = d12 // 4
        f1 = fs_in / d1
        self.st1 = FirDecimator(design_lowpass(fs_in, 11e3, f1 - 11e3), d1)
        self.st2 = FirDecimator(design_lowpass(f1, 11e3, FS_WIDE - 11e3), 4)
        self.st3 = FirDecimator(design_lowpass(FS_WIDE, 1.6e3, FS_NB - 1.6e3), int(FS_WIDE // FS_NB))
        self.reset()

    def reset(self) -> None:
        for st in (self.st1, self.st2, self.st3):
            st.reset()
        self._mix_k = 0
        self._nco_phase = 0.0

    def full_to_wide(self, x: np.ndarray) -> np.ndarray:
        """1,024 MS/s (VOR em −fs/4) → 32 kS/s (VOR em 0 Hz)."""
        n = len(x)
        k = (self._mix_k + np.arange(n)) & 3
        self._mix_k = (self._mix_k + n) & 3
        return self.st2.process(self.st1.process(x.astype(np.complex64, copy=False) * _J4[k]))

    def nco(self, x: np.ndarray, shift_hz: float) -> np.ndarray:
        """Desloca o fluxo largo por −shift_hz (fase contínua entre blocos)."""
        if shift_hz == 0.0 or len(x) == 0:
            return x
        w = -2.0 * math.pi * shift_hz / FS_WIDE
        ph = self._nco_phase + w * np.arange(len(x))
        self._nco_phase = float((ph[-1] + w) % (2 * math.pi))
        return (x * np.exp(1j * ph)).astype(np.complex64)

    def wide_to_nb(self, w: np.ndarray) -> np.ndarray:
        return self.st3.process(w)


class SpectrumAnalyzer:
    """FFT longa com janela, overlap e integração incoerente de N espectros.

    Saída em densidade espectral (unid. de FS²/Hz). Para um tom de amplitude a,
    o pico no bin vale ≈ a²/ENBW; para ruído de densidade N0, a média é N0.
    """

    def __init__(self, fs: float, nfft: int, overlap: float, window: str, n_integrate: int):
        self.fs = fs
        self.nfft = int(nfft)
        self.hop = max(1, int(round(self.nfft * (1.0 - overlap))))
        name = "blackmanharris" if window.lower().startswith("black") else "hann"
        w = signal.get_window(name, self.nfft)
        self.win = w.astype(np.float32)
        self.norm = 1.0 / (fs * np.sum(w ** 2))
        self.enbw_hz = fs * np.sum(w ** 2) / np.sum(w) ** 2
        self.bin_hz = fs / self.nfft
        self.freqs = np.fft.fftshift(np.fft.fftfreq(self.nfft, 1.0 / fs))
        self.dt = self.hop / fs
        self.n_integrate = max(1, int(n_integrate))
        self.hist: deque = deque(maxlen=self.n_integrate)
        self.buf = np.zeros(0, np.complex64)
        self.n_consumed = 0

    def push(self, x: np.ndarray) -> list[tuple[float, np.ndarray]]:
        self.buf = np.concatenate((self.buf, x))
        out = []
        while len(self.buf) >= self.nfft:
            X = np.fft.fftshift(np.fft.fft(self.buf[: self.nfft] * self.win))
            p = (X.real.astype(np.float64) ** 2 + X.imag.astype(np.float64) ** 2) * self.norm
            self.hist.append(p)
            pint = np.mean(self.hist, axis=0) if len(self.hist) > 1 else p
            # centro temporal do conjunto integrado (s desde o início do fluxo)
            tc = (self.n_consumed + self.nfft / 2) / self.fs - (len(self.hist) - 1) * self.dt / 2
            out.append((tc, pint))
            self.buf = self.buf[self.hop:]
            self.n_consumed += self.hop
        return out


class Background:
    """Fundo espectral de longa duração (EMA) com atualização bloqueada nos alvos.

    Normalizar o espectro por este fundo remove linhas estáticas (portadora
    residual, clutter de terreno, bandas laterais AM 30 Hz, espúrios) e
    branqueia o piso de ruído, deixando visíveis só os traços que se movem.
    """

    def __init__(self, tau_s: float, dt: float):
        self.tau_s = tau_s
        self.dt = dt
        self.value: np.ndarray | None = None
        self.k = 0

    @property
    def age_s(self) -> float:
        return self.k * self.dt

    def update(self, p: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.k += 1
        a = max(1.0 / self.k, min(1.0, self.dt / max(self.tau_s, 1e-3)))
        if self.value is None:
            self.value = p.copy()
            return
        if mask is None:
            self.value += a * (p - self.value)
        else:  # nos bins com alvo, atualização 20× mais lenta
            aa = np.where(mask, a / 20.0, a)
            self.value += aa * (p - self.value)


class ReferenceEstimator:
    """Estima a frequência de 0 Hz Doppler pela linha estática dominante.

    Sem visada direta, o sinal direto chega por difração/clutter e aparece como
    uma linha fixa (Doppler nulo). Sua posição mede o erro conjunto do LO do
    RTL-SDR (TCXO 1 ppm ≈ ±116 Hz) e do transmissor VOR, que de outra forma
    contaminaria diretamente o Δf medido.
    """

    def __init__(self, freqs: np.ndarray, dt: float, tau_s: float, search_hz: float,
                 min_prom_db: float):
        self.freqs = freqs
        self.dt = dt
        self.tau_s = tau_s
        self.search_hz = search_hz
        self.min_prom_db = min_prom_db
        self.avg: np.ndarray | None = None
        self.k = 0
        self.f_ref: float | None = None
        self.locked = False
        self.prominence_db = 0.0

    def update(self, p: np.ndarray) -> None:
        self.k += 1
        a = max(1.0 / self.k, min(1.0, self.dt / self.tau_s))
        self.avg = p.copy() if self.avg is None else self.avg + a * (p - self.avg)
        if self.k * self.dt < 8.0:
            return
        sel = np.flatnonzero(np.abs(self.freqs) <= self.search_hz)
        if len(sel) < 8:
            return
        sub = self.avg[sel]
        i = int(np.argmax(sub))
        self.prominence_db = 10 * math.log10(sub[i] / max(np.median(sub), 1e-30))
        if self.prominence_db >= self.min_prom_db and 0 < i < len(sub) - 1:
            d = parabolic_offset(10 * np.log10(sub[i - 1: i + 2]))
            f_est = float(self.freqs[sel[i]] + d * (self.freqs[1] - self.freqs[0]))
            self.f_ref = f_est if self.f_ref is None else self.f_ref + 0.2 * (f_est - self.f_ref)
            self.locked = True
        else:
            self.locked = False


def parabolic_offset(y3: np.ndarray) -> float:
    """Deslocamento sub-bin do pico por interpolação parabólica (em dB)."""
    y0, y1, y2 = (float(v) for v in y3)
    den = y0 - 2 * y1 + y2
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))


class Cfar:
    """CFAR 1D: OS (mediana das células de referência) ou CA (média)."""

    def __init__(self, guard: int, ref: int, mode: str, n_integrate: int):
        self.G = max(0, int(guard))
        self.R = max(2, int(ref))
        self.mode = mode.upper()
        k = max(1, int(n_integrate))
        # mediana de uma Gama(k) normalizada (média 1): corrige o viés do OS
        self.med_corr = float(stats.gamma(a=k).median() / k)

    def noise(self, s: np.ndarray) -> np.ndarray:
        pad = self.G + self.R
        sp = np.pad(s, pad, mode="reflect")
        win = sliding_window_view(sp, 2 * pad + 1)
        refs = np.concatenate((win[:, : self.R], win[:, -self.R:]), axis=1)
        if self.mode == "OS":
            return np.median(refs, axis=1) / self.med_corr
        return refs.mean(axis=1)


@dataclass
class Peak:
    idx: int
    f_abs: float
    snr_db: float


def detect_peaks(s: np.ndarray, noise: np.ndarray, freqs: np.ndarray, f_ref: float,
                 thr_db: float, max_doppler_hz: float, exclude_hz: float,
                 exclude_am30: bool, min_sep_bins: int = 3, max_peaks: int = 12) -> list[Peak]:
    snr = s / np.maximum(noise, 1e-30)
    cand = snr > 10 ** (thr_db / 10)
    cand[1:-1] &= (s[1:-1] >= s[:-2]) & (s[1:-1] >= s[2:])
    cand[0] = cand[-1] = False
    df = np.abs(freqs - f_ref)
    cand &= (df <= max_doppler_hz) & (df > exclude_hz)
    if exclude_am30:
        cand &= np.abs(df - 30.0) > 1.5
    idx = np.flatnonzero(cand)
    if len(idx) == 0:
        return []
    order = idx[np.argsort(-snr[idx])]
    taken: list[int] = []
    for i in order:
        if all(abs(int(i) - j) >= min_sep_bins for j in taken):
            taken.append(int(i))
            if len(taken) >= max_peaks:
                break
    binw = freqs[1] - freqs[0]
    peaks = []
    for i in taken:
        d = parabolic_offset(10 * np.log10(np.maximum(s[i - 1: i + 2], 1e-30)))
        peaks.append(Peak(i, float(freqs[i] + d * binw), float(10 * np.log10(snr[i]))))
    return peaks
