"""Rastreador de traços Doppler (vizinho mais próximo global + ciclo de vida M/N).

Cada quadro espectral gera detecções (Δf, SNR). O rastreador:
  * remove "fantasmas" AM 30 Hz (bandas laterais do próprio eco, ±30 Hz e
    ≥6 dB abaixo de um pico mais forte);
  * associa detecções a trilhas por atribuição ótima (Hungarian) dentro de uma
    janela (gate) em torno da predição linear da trilha;
  * confirma trilhas com hits suficientes e duração mínima; encerra trilhas
    sem atualização por max_miss_s e devolve as concluídas para resumo/log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import TrackerConfig
from .physics import local_slope, max_plausible_doppler, CpaFit


@dataclass
class Detection:
    t: float          # epoch UTC (s)
    f: float          # Δf relativo à referência (Hz)
    f_abs: float      # posição no eixo do espectro (Hz, relativo à nominal + offset fino)
    snr_db: float
    p_db: float


class Track:
    def __init__(self, det: Detection):
        self.t: list[float] = [det.t]
        self.f: list[float] = [det.f]
        self.fa: list[float] = [det.f_abs]
        self.snr: list[float] = [det.snr_db]
        self.pdb: list[float] = [det.p_db]
        self.confirmed = False
        self.uid: int | None = None
        self.fit: CpaFit | None = None
        self.fit_t = 0.0
        self.radial: dict | None = None
        self.iq_file: str | None = None
        self.implausible = False

    @property
    def t0(self) -> float:
        return self.t[0]

    @property
    def last_t(self) -> float:
        return self.t[-1]

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0]

    def add(self, d: Detection) -> None:
        self.t.append(d.t)
        self.f.append(d.f)
        self.fa.append(d.f_abs)
        self.snr.append(d.snr_db)
        self.pdb.append(d.p_db)

    def slope(self, span_s: float = 5.0) -> tuple[float, float]:
        return local_slope(np.array(self.t), np.array(self.f), span_s)

    def slope_abs(self, span_s: float = 5.0) -> tuple[float, float]:
        return local_slope(np.array(self.t), np.array(self.fa), span_s)

    def predict(self, t: float) -> tuple[float, float]:
        s, f0 = self.slope(6.0)
        s = float(np.clip(s, -60.0, 60.0))
        return f0 + s * (t - self.last_t), s

    def status(self, thr_hz: float = 1.5) -> str:
        s, f0 = self.slope(5.0)
        t = np.array(self.t)
        f = np.array(self.f)
        rec = f[t >= t[-1] - 6.0]
        if len(rec) >= 3 and rec.min() < -thr_hz / 2 and rec.max() > thr_hz / 2:
            return "cruzamento (CPA)"
        if f0 > thr_hz:
            return "aproximação"
        if f0 < -thr_hz:
            return "afastamento"
        return "cruzamento (CPA)"


def suppress_am30(dets: list[Detection], tol_hz: float = 2.0, margin_db: float = 6.0) -> list[Detection]:
    keep = []
    for d in dets:
        ghost = any(
            o is not d and abs(abs(d.f_abs - o.f_abs) - 30.0) <= tol_hz and o.snr_db >= d.snr_db + margin_db
            for o in dets)
        if not ghost:
            keep.append(d)
    return keep


class Tracker:
    def __init__(self, cfg: TrackerConfig, wavelength: float):
        self.cfg = cfg
        self.lam = wavelength
        self.tracks: list[Track] = []
        self.next_uid = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, t: float, dets: list[Detection], frame_dt: float,
               dead_zone=None) -> tuple[list[Track], list[Track]]:
        """Retorna (trilhas recém-confirmadas, trilhas encerradas e válidas).

        dead_zone(f) → True se Δf cai numa zona onde o detector está cego
        (em torno de 0 Hz e das bandas AM ±30 Hz). Trilhas confirmadas cuja
        predição está numa zona cega seguem por inércia até 5× max_miss_s.
        """
        cfg = self.cfg
        if cfg.suppress_am30_ghosts and len(dets) > 1:
            dets = suppress_am30(dets)
        fmax_phys = max_plausible_doppler(cfg.v_max_ms, self.lam)
        assigned_det: set[int] = set()
        if self.tracks and dets:
            cost = np.full((len(self.tracks), len(dets)), 1e6)
            for i, tr in enumerate(self.tracks):
                fp, s = tr.predict(t)
                gate = cfg.gate_hz + 0.5 * abs(s) * max(frame_dt, t - tr.last_t)
                for j, d in enumerate(dets):
                    e = abs(d.f - fp)
                    if e <= gate:
                        cost[i, j] = e - (0.5 if tr.confirmed else 0.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e5:
                    self.tracks[i].add(dets[j])
                    assigned_det.add(j)
        for j, d in enumerate(dets):
            if j not in assigned_det and abs(d.f) <= fmax_phys * 1.1:
                self.tracks.append(Track(d))

        newly_confirmed, finished, keep = [], [], []
        for tr in self.tracks:
            since = t - tr.last_t
            if not tr.confirmed:
                frames = (tr.last_t - tr.t0) / max(frame_dt, 1e-3) + 1
                hits = len(tr.t)
                if hits >= cfg.confirm_hits and tr.duration >= cfg.confirm_min_s and hits >= 0.6 * frames:
                    tr.confirmed = True
                    tr.uid = self.next_uid
                    self.next_uid += 1
                    newly_confirmed.append(tr)
                elif since > 2.5 * frame_dt or tr.duration > 4 * max(cfg.confirm_min_s, 1.0):
                    continue  # trilha tentativa descartada
            coasting = False
            if tr.confirmed and since > cfg.max_miss_s and dead_zone is not None:
                fp, _ = tr.predict(t)
                coasting = bool(dead_zone(fp)) and since <= 5 * cfg.max_miss_s
            if tr.confirmed and since > cfg.max_miss_s and not coasting:
                if tr.duration >= cfg.min_duration_s:
                    finished.append(tr)
                continue
            if tr.confirmed:
                tr.implausible = abs(tr.f[-1]) > fmax_phys
            keep.append(tr)
        self.tracks = keep
        return newly_confirmed, finished

    def flush(self) -> list[Track]:
        out = [tr for tr in self.tracks if tr.confirmed and tr.duration >= self.cfg.min_duration_s]
        self.tracks = []
        return out
