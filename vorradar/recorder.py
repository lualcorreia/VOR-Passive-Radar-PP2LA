"""Gravação: trechos IQ em SigMF (fluxo largo 32 kS/s) e logs CSV de eventos/pontos."""
from __future__ import annotations

import csv
import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import FS_WIDE, PROJECT_DIR


def utc_iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def local_str(t: float) -> str:
    return datetime.fromtimestamp(t).astimezone().strftime("%H:%M:%S")


def utc_str(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%H:%M:%S")


def _data(base: Path) -> Path:
    return Path(str(base) + ".sigmf-data")


def resolve_dir(d: str) -> Path:
    p = Path(d)
    if not p.is_absolute():
        p = PROJECT_DIR / p
    p.mkdir(parents=True, exist_ok=True)
    return p


class IQRecorder:
    """Mantém um pré-buffer circular do fluxo largo e grava SigMF sob demanda."""

    def __init__(self, out_dir: str, prebuffer_s: float = 10.0, post_s: float = 5.0,
                 max_s: float = 120.0, fs: float = FS_WIDE):
        self.dir = resolve_dir(out_dir) / "iq"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fs = fs
        self.pre_n = int(prebuffer_s * fs)
        self.post_s = post_s
        self.max_n = int(max_s * fs)
        self.ring: deque = deque()
        self.ring_n = 0
        self.active: dict | None = None

    def push(self, t_first: float, x: np.ndarray) -> str | None:
        """Alimenta o recorder; devolve o caminho se uma gravação terminou neste bloco."""
        self.ring.append((t_first, x))
        self.ring_n += len(x)
        while self.ring and self.ring_n - len(self.ring[0][1]) >= self.pre_n:
            self.ring_n -= len(self.ring.popleft()[1])
        a = self.active
        if a is None:
            return None
        x.astype(np.complex64).tofile(a["fh"])
        a["n"] += len(x)
        t_end = t_first + len(x) / self.fs
        if a["n"] >= self.max_n or (a["stop_at"] is not None and t_end >= a["stop_at"]):
            return self._close()
        return None

    def start(self, label: str, f_center_hz: float, fine_offset_hz: float, ref_hz: float,
              extra: dict | None = None) -> str | None:
        if self.active is not None or not self.ring:
            return None
        t0 = self.ring[0][0]
        stamp = datetime.fromtimestamp(t0, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"vor_{f_center_hz / 1e6:.3f}".replace(".", "p") + f"MHz_{stamp}_{label}"
        base = self.dir / name
        fh = open(_data(base), "wb")
        n = 0
        for _, blk in self.ring:
            blk.astype(np.complex64).tofile(fh)
            n += len(blk)
        self.active = {"fh": fh, "base": base, "n": n, "t0": t0, "label": label, "stop_at": None,
                       "f": f_center_hz + fine_offset_hz, "fine": fine_offset_hz, "ref": ref_hz,
                       "extra": extra or {}}
        return str(_data(base))

    def request_stop(self, t_now: float) -> None:
        if self.active is not None and self.active["stop_at"] is None:
            self.active["stop_at"] = t_now + self.post_s

    def _close(self) -> str:
        a = self.active
        self.active = None
        a["fh"].close()
        meta = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": self.fs,
                "core:version": "1.0.0",
                "core:recorder": "vorradar",
                "core:hw": "RTL-SDR Blog V3",
                "core:description": "Fluxo largo (32 kS/s) centrado na portadora VOR; eco Doppler de aeronave",
                "vorradar:fine_offset_hz": a["fine"],
                "vorradar:ref_offset_hz": a["ref"],
                **{f"vorradar:{k}": v for k, v in a["extra"].items()},
            },
            "captures": [{"core:sample_start": 0, "core:frequency": a["f"], "core:datetime": utc_iso(a["t0"])}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": a["n"], "core:label": a["label"]}],
        }
        Path(str(a["base"]) + ".sigmf-meta").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return str(_data(a["base"]))

    def close(self) -> str | None:
        return self._close() if self.active is not None else None


EVENT_FIELDS = [
    "id", "inicio_utc", "fim_utc", "duracao_s", "n_pontos", "f_vor_mhz", "ref_hz", "ref_estado",
    "df_max_hz", "df_min_hz", "vb_inicio_kt", "vb_fim_kt", "dfdt_max_abs_hz_s",
    "snr_pico_db", "t_snr_pico_utc",
    "cpa_utc", "cpa_local", "cpa_sigma_s", "cpa_observado", "doppler_nulo_utc", "doppler_nulo_local",
    "ajuste_A_hz", "ajuste_tau_s",
    "ajuste_c_hz", "dfdt_cpa_hz_s", "ajuste_r2", "ajuste_rms_hz",
    "v_perna_unica_kt", "v_monostatico_kt", "d_cpa_perna_unica_km", "d_cpa_monostatico_km",
    "ajuste_msg", "radial_exp_deg", "radial_snr_am_db", "radial_snr_fm_db", "arquivo_iq",
]
POINT_FIELDS = ["id", "t_utc", "epoch_s", "df_hz", "f_eixo_hz", "vb_ms", "vb_kt", "snr_db", "p_dbfs_hz"]


class CsvLogger:
    def __init__(self, log_dir: str):
        self.dir = resolve_dir(log_dir)

    def _append(self, name: str, fields: list[str], rows: list[dict]) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        p = self.dir / f"{name}_{day}.csv"
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)
        return p

    def log_event(self, ev: dict) -> Path:
        return self._append("eventos", EVENT_FIELDS, [ev])

    def log_points(self, rows: list[dict]) -> Path:
        return self._append("pontos", POINT_FIELDS, rows)

    @staticmethod
    def export(path: str, rows: list[dict], fields: list[str] = EVENT_FIELDS) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
