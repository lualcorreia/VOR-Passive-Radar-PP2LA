"""Simulador físico de IQ: VOR residual + ecos bistáticos de aeronaves + ruído.

Geometria em coordenadas locais ENU (m) com o receptor na origem e o VOR em
`tx`. Cada aeronave voa em linha reta a velocidade constante. Para cada
amostra calcula-se R_T = |p − tx|, R_R = |p| e a fase do eco
    φ(t) = −2π (R_T + R_R)/λ + 2π f_erro t
o que produz exatamente a curva em "S" bistática. A amplitude segue a
equação do radar (∝ 1/(R_T·R_R)) normalizada para o C/N0 de pico escolhido.
O eco carrega a modulação VOR (CVOR) com a fase de 30 Hz da radial em que a
aeronave está — permitindo validar o módulo experimental de azimute.

C/N0 (dB-Hz) = 10·log10(a²·fs/σ²): é a medida física independente da FFT.
Para FFT Hann com ENBW ≈ 1,47 Hz, SNR por bin ≈ C/N0 − 1,7 dB.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .config import AppConfig, C
from .physics import enu_offset, geo_valid


@dataclass
class SimAircraft:
    uid: int
    t_spawn: float      # tempo de simulação (s)
    p0: np.ndarray      # posição em t_spawn (ENU, m)
    vel: np.ndarray     # m/s
    t_end: float
    a_ref: float        # amplitude × (R_T·R_R)
    t_cpa: float
    d_cpa_m: float
    speed: float

    def pos(self, t):
        return self.p0[None, :] + self.vel[None, :] * (np.atleast_1d(t) - self.t_spawn)[:, None]

    def doppler(self, t, tx, lam):
        """Doppler bistático verdadeiro (Hz) — sem erro de LO."""
        p = self.pos(t)
        rr = np.linalg.norm(p, axis=1)
        pt = p - tx[None, :]
        rt = np.linalg.norm(pt, axis=1)
        drdt = (p @ self.vel) / rr + (pt @ self.vel) / rt
        return -drdt / lam


def vor_composite(t: np.ndarray, radial_deg: float, ident: bool = True) -> np.ndarray:
    """Envelope AM de um CVOR: 30 Hz variável (30%), subportadora 9960 Hz FM ±480 Hz (30%), ident 1020 Hz."""
    th = math.radians(radial_deg)
    w30 = 2 * math.pi * 30.0 * t
    m = 0.3 * np.cos(w30 - th) + 0.3 * np.cos(2 * math.pi * 9960.0 * t + 16.0 * np.sin(w30))
    if ident:
        gate = ((t % 12.0) < 3.0) & ((np.floor(t * 6.0) % 2) == 0)
        m = m + 0.1 * gate * np.cos(2 * math.pi * 1020.0 * t)
    return 1.0 + m


class VorSimulator:
    def __init__(self, cfg: AppConfig, fs: float = 1.024e6, block_s: float = 0.256,
                 t0_epoch: float | None = None, seed: int | None = None, spawn: bool = True):
        self.cfg = cfg
        self.fs = float(fs)
        self.lam = C / cfg.f_carrier_hz
        # LO do RTL-SDR em f_VOR + fs/4 → VOR aparece em −fs/4 no fluxo de taxa total
        self.f_bb = -self.fs / 4 if self.fs > 100e3 else 0.0
        self.n_block = int(round(block_s * self.fs / 4)) * 4
        self.sigma = 0.05
        g = cfg.geo
        if geo_valid(g.rx_lat, g.rx_lon, g.vor_lat, g.vor_lon):
            e, n = enu_offset(g.rx_lat, g.rx_lon, g.vor_lat, g.vor_lon)
        else:
            e, n = 25e3 * math.sin(math.radians(60)), 25e3 * math.cos(math.radians(60))
        self.tx = np.array([e, n, 0.0])
        self.L = float(np.linalg.norm(self.tx))
        self.rng = np.random.default_rng(seed if seed else None)
        self.t = 0.0
        self.t0_epoch = time.time() if t0_epoch is None else t0_epoch
        self.aircraft: list[SimAircraft] = []
        self.spawn_enabled = spawn
        self.next_spawn = 3.0
        self.next_uid = 1
        self.messages: list[str] = []
        self.phi_d = self.rng.uniform(0, 2 * math.pi)

    def amp(self, cn0_dbhz: float) -> float:
        return self.sigma * math.sqrt(10 ** (cn0_dbhz / 10) / self.fs)

    def add_aircraft(self, speed: float, d_cpa: float, heading_deg: float, alt: float,
                     t_to_cpa: float, life_after: float, cn0_peak: float) -> SimAircraft:
        h = math.radians(heading_deg)
        u = np.array([math.sin(h), math.cos(h), 0.0])
        nrm = np.array([math.cos(h), -math.sin(h), 0.0])
        p_cpa = d_cpa * nrm + np.array([0, 0, alt])
        p0 = p_cpa - u * speed * t_to_cpa
        ac = SimAircraft(self.next_uid, self.t, p0, u * speed, self.t + t_to_cpa + life_after, 0.0,
                         self.t + t_to_cpa, float(np.linalg.norm(p_cpa)), speed)
        ts = np.linspace(self.t, ac.t_end, 400)
        p = ac.pos(ts)
        prod = np.linalg.norm(p, axis=1) * np.linalg.norm(p - self.tx, axis=1)
        ac.a_ref = self.amp(cn0_peak) * float(prod.min())
        self.next_uid += 1
        self.aircraft.append(ac)
        f = ac.doppler(ts, self.tx, self.lam)
        t_cpa_utc = time.strftime("%H:%M:%S", time.gmtime(self.t0_epoch + ac.t_cpa))
        self.messages.append(
            f"SIM: aeronave #{ac.uid} v={speed:.0f} m/s ({speed / 0.514444:.0f} kt), "
            f"rumo {heading_deg:.0f}°, alt {alt / 1e3:.1f} km, CPA(RX) {ac.d_cpa_m / 1e3:.1f} km "
            f"às {t_cpa_utc} UTC, Doppler verdadeiro {f.max():+.0f}…{f.min():+.0f} Hz, "
            f"C/N0 pico {cn0_peak:.0f} dB-Hz")
        return ac

    def _spawn_random(self) -> None:
        s = self.cfg.sim
        r = self.rng
        self.add_aircraft(speed=r.uniform(120, 240), d_cpa=r.uniform(2e3, 25e3),
                          heading_deg=r.uniform(0, 360), alt=r.uniform(3e3, 11e3),
                          t_to_cpa=r.uniform(60, 110), life_after=r.uniform(60, 110),
                          cn0_peak=s.echo_cn0_dbhz + r.uniform(-6, 3))
        self.next_spawn = self.t + r.uniform(s.spawn_min_s, s.spawn_max_s)

    def next_block(self) -> tuple[float, np.ndarray]:
        s = self.cfg.sim
        if self.spawn_enabled and self.t >= self.next_spawn:
            self._spawn_random()
        n = self.n_block
        t = self.t + np.arange(n) / self.fs
        x = (self.rng.standard_normal(n) + 1j * self.rng.standard_normal(n)) * (self.sigma / math.sqrt(2))
        w_off = 2 * math.pi * (self.f_bb + s.lo_error_hz)
        if s.direct_cn0_dbhz > 0:
            th_rx = math.degrees(math.atan2(-self.tx[0], -self.tx[1])) % 360
            x += self.amp(s.direct_cn0_dbhz) * vor_composite(t - self.L / C, th_rx) * \
                np.exp(1j * (w_off * t + self.phi_d))
        alive = []
        for ac in self.aircraft:
            if self.t > ac.t_end:
                continue
            alive.append(ac)
            p = ac.pos(t)
            rr = np.linalg.norm(p, axis=1)
            rt = np.linalg.norm(p - self.tx[None, :], axis=1)
            rb = rr + rt
            pm = p[n // 2] - self.tx
            th = math.degrees(math.atan2(pm[0], pm[1])) % 360   # radial VOR→aeronave
            ph = -2 * math.pi * rb / self.lam + w_off * t
            x += (ac.a_ref / (rr * rt)) * vor_composite(t - rb / C, th) * np.exp(1j * ph)
        self.aircraft = alive
        t_first = self.t0_epoch + self.t
        self.t += n / self.fs
        return t_first, x.astype(np.complex64)
