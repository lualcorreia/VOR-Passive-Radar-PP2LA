"""Estimadores físicos: velocidade bistática, df/dt, CPA (curva em "S") e geometria.

Convenções
----------
Δf = f_eco − f_ref.  Para o trajeto bistático R_b = R_T + R_R (VOR→aeronave→receptor):
    Δf = −(1/λ)·dR_b/dt      →      v_b = dR_b/dt = −λ·Δf
Δf > 0  ⇒  v_b < 0  ⇒  trajeto encurtando ("aproximando").
|v_b| ≤ 2·|v_aeronave| sempre; a velocidade da aeronave NÃO é observável só pelo Δf.

Modelo de passagem (CPA)
------------------------
Para voo retilíneo e uniforme, se uma das pernas domina a variação (ex.: VOR
longe, receptor perto da rota), a perna dominante vale R(t) = sqrt(d² + v²(t−t0)²)
e a outra contribui com um termo quase constante:
    Δf(t) ≈ −A·(t−t0)/sqrt(τ² + (t−t0)²) + c ,   A = v/λ ,  τ = d/v
Ajustando (A, t0, τ, c):
    t0  → CPA da perna dominante (ponto de inflexão do "S"; Doppler cruza c)
    t_zero (Δf = 0) → mínimo do trajeto bistático R_T+R_R; difere de t0 quando c ≠ 0
    −A/τ → inclinação máxima do Doppler (Hz/s)
    hipótese "perna única":      v ≈ A·λ ,    d_CPA ≈ v·τ
    hipótese "quase monostática" (VOR e RX na mesma direção): v ≈ A·λ/2, d_CPA ≈ v·τ
O resultado real fica entre esses limites; sem medir atraso (sinal CW, sem
referência direta) não há distância absoluta — por isso os valores são
rotulados como estimativas geométricas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from .config import C, KT

R_EARTH = 6_371_008.8


def wavelength(f_hz: float) -> float:
    return C / f_hz


def bistatic_rate(df_hz, lam: float):
    """Taxa de variação do trajeto bistático (m/s); negativa = encurtando."""
    return -np.asarray(df_hz) * lam


def ms_to_kt(v):
    return np.asarray(v) / KT


def ms_to_kmh(v):
    return np.asarray(v) * 3.6


def max_plausible_doppler(v_max_ms: float, lam: float) -> float:
    return 2.0 * v_max_ms / lam


def local_slope(t: np.ndarray, f: np.ndarray, span_s: float = 5.0) -> tuple[float, float]:
    """Regressão linear dos pontos nos últimos span_s segundos → (df/dt, f no último instante)."""
    if len(t) < 3:
        return 0.0, float(f[-1]) if len(f) else 0.0
    t = np.asarray(t)
    sel = t >= t[-1] - span_s
    if sel.sum() < 3:
        sel = slice(-3, None)
    tt = t[sel] - t[-1]
    ff = np.asarray(f)[sel]
    p = np.polyfit(tt, ff, 1)
    return float(p[0]), float(p[1])


def s_curve(t, A, t0, tau, c):
    u = t - t0
    return -A * u / np.sqrt(tau * tau + u * u) + c


@dataclass
class CpaFit:
    ok: bool = False
    message: str = ""
    A_hz: float = math.nan
    t0: float = math.nan            # epoch (s)
    t0_sigma: float = math.nan
    tau_s: float = math.nan
    c_hz: float = math.nan
    slope_cpa: float = math.nan     # Hz/s no cruzamento
    r2: float = math.nan
    rms_hz: float = math.nan
    observed: bool = False          # t0 dentro do intervalo observado
    t_zero: float = math.nan        # epoch em que Δf = 0 (mínimo do trajeto bistático, dR_b/dt = 0)
    zero_observed: bool = False
    v_single: float = math.nan      # m/s (hipótese perna única)
    v_mono: float = math.nan        # m/s (hipótese quase monostática)
    d_single_km: float = math.nan
    d_mono_km: float = math.nan
    plausible: bool = True

    def curve(self, t):
        return s_curve(np.asarray(t) - 0.0, self.A_hz, self.t0, self.tau_s, self.c_hz)


def fit_cpa(t, f, lam: float, v_max_ms: float = 340.0, min_points: int = 15) -> CpaFit:
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if len(t) < min_points or t[-1] - t[0] < 8.0:
        return CpaFit(message="poucos pontos")
    T0 = t[0]
    tt = t - T0
    k = min(7, len(f) // 3 * 2 + 1)
    fs_ = np.convolve(f, np.ones(k) / k, mode="same") if k >= 3 else f
    fmax, fmin = np.percentile(f, 95), np.percentile(f, 5)
    c0 = 0.5 * (fmax + fmin)
    A0 = max(0.5 * (fmax - fmin), 1.0)
    cross = np.flatnonzero((fs_[:-1] - c0 > 0) & (fs_[1:] - c0 <= 0))
    grad = np.gradient(fs_, tt)
    if len(cross):
        t00 = tt[cross[len(cross) // 2]]
    else:
        t00 = tt[int(np.argmax(np.abs(grad)))]
    slope0 = max(float(np.max(np.abs(grad))), 0.02)
    tau0 = float(np.clip(A0 / slope0, 0.5, 600.0))
    a_max = max_plausible_doppler(v_max_ms, lam) * 1.5
    lo = [0.2, tt[0] - 900.0, 0.3, -a_max]
    hi = [a_max, tt[-1] + 900.0, 1800.0, a_max]
    p0 = [min(A0, a_max * 0.99), t00, tau0, float(np.clip(c0, -a_max * 0.99, a_max * 0.99))]
    try:
        p, cov = curve_fit(s_curve, tt, f, p0=p0, bounds=(lo, hi), maxfev=6000)
    except Exception as e:  # noqa: BLE001
        return CpaFit(message=f"ajuste falhou: {e.__class__.__name__}")
    A, t0, tau, c = (float(v) for v in p)
    res = f - s_curve(tt, *p)
    ss_tot = float(np.sum((f - f.mean()) ** 2)) or 1e-12
    r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot
    perr = np.sqrt(np.clip(np.diag(cov), 0, None)) if np.all(np.isfinite(cov)) else [math.nan] * 4
    fit = CpaFit(ok=True, A_hz=A, t0=T0 + t0, t0_sigma=float(perr[1]), tau_s=tau, c_hz=c,
                 slope_cpa=-A / tau, r2=r2, rms_hz=float(np.sqrt(np.mean(res ** 2))),
                 observed=bool(tt[0] <= t0 <= tt[-1]))
    if abs(c) < A:   # raiz fechada de s_curve = 0:  u/sqrt(τ²+u²) = c/A
        q = c / A
        fit.t_zero = T0 + t0 + tau * q / math.sqrt(1.0 - q * q)
        fit.zero_observed = bool(t[0] <= fit.t_zero <= t[-1])
    fit.v_single = A * lam
    fit.v_mono = A * lam / 2.0
    fit.d_single_km = fit.v_single * tau / 1000.0
    fit.d_mono_km = fit.v_mono * tau / 1000.0
    fit.plausible = fit.v_mono <= v_max_ms * 1.05 and fit.r2 > 0.6
    if not fit.observed:
        fit.message = "CPA extrapolado (fora da janela observada)"
    elif not fit.plausible:
        fit.message = "ajuste pouco confiável (R² baixo ou v implausível)"
    else:
        fit.message = "ok"
    return fit


# ----------------------------------------------------------------- geometria
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def enu_offset(lat0, lon0, lat, lon) -> tuple[float, float]:
    """Deslocamento leste/norte (m) de (lat,lon) em relação a (lat0,lon0) — aproximação local."""
    e = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    n = math.radians(lat - lat0) * R_EARTH
    return e, n


def geo_valid(*vals) -> bool:
    return all(v is not None and np.isfinite(v) for v in vals)
