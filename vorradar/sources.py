"""Fontes de IQ: RTL-SDR ao vivo, simulação e replay de arquivo (SigMF / .cu8).

Todas entregam na fila tuplas (t_primeira_amostra_epoch, amostras complex64, fs, tipo):
    tipo "full": 1,024 MS/s com o VOR em −fs/4 (LO = f_VOR + fs/4)
    tipo "wide": 32 kS/s com o VOR em 0 Hz (gravações do próprio app)
"""
from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import AppConfig, FS_WIDE
from .simulator import VorSimulator

R820T_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7, 22.9,
               25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5,
               48.0, 49.6]


class Source:
    name = "fonte"
    wide_offset_applied_hz = 0.0   # offset fino já aplicado nas amostras (replay de fluxo largo)

    def __init__(self):
        self.q: queue.Queue | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.finished = False
        self.stats = {"blocks": 0, "samples": 0, "queue_drops": 0, "loss_pct": 0.0, "fs": 0.0}
        self.messages: list[str] = []

    def start(self, q: queue.Queue) -> None:
        self.q = q
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_guarded, name=self.name, daemon=True)
        self._thread.start()

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001
            self.error = f"{e.__class__.__name__}: {e}"
        finally:
            self.finished = True

    def _run(self):
        raise NotImplementedError

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _put(self, item, block: bool) -> None:
        try:
            if block:
                while not self._stop.is_set():
                    try:
                        self.q.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
            else:
                self.q.put_nowait(item)
            self.stats["blocks"] += 1
            self.stats["samples"] += len(item[1])
        except queue.Full:
            self.stats["queue_drops"] += 1

    # controles de RF (sem efeito em simulação/replay)
    def set_freq(self, f_vor_hz: float): ...
    def set_gain(self, gain_db: float, agc: bool): ...
    def set_ppm(self, ppm: int): ...
    def set_bias_tee(self, on: bool): ...


class RtlSdrSource(Source):
    name = "RTL-SDR"

    def __init__(self, cfg: AppConfig, device_index: int = 0, block: int = 256 * 1024):
        super().__init__()
        self.cfg = cfg
        self.idx = device_index
        self.block = block
        self.sdr = None
        self._lock = threading.Lock()
        self.stats["fs"] = cfg.rf.sample_rate

    def start(self, q):
        from rtlsdr import RtlSdr
        rf = self.cfg.rf
        self.sdr = RtlSdr(self.idx)
        self.sdr.sample_rate = rf.sample_rate
        self.fs = float(self.sdr.sample_rate)
        self.stats["fs"] = self.fs
        self.set_ppm(rf.ppm)
        self.set_freq(rf.vor_freq_mhz * 1e6)
        self.set_gain(rf.gain_db, rf.agc)
        try:
            self.sdr.set_agc_mode(False)  # AGC digital do RTL2832 sempre desligado
        except Exception:  # noqa: BLE001
            pass
        self.set_bias_tee(rf.bias_tee)
        self._t_start = None
        self._n = 0
        super().start(q)

    def _run(self):
        self.sdr.read_samples_async(self._cb, num_samples=self.block)

    def _cb(self, samples, ctx):
        if self._stop.is_set():
            try:
                self.sdr.cancel_read_async()
            except Exception:  # noqa: BLE001
                pass
            return
        now = time.time()
        if self._t_start is None:
            self._t_start = now - len(samples) / self.fs
        t_first = self._t_start + self._n / self.fs
        self._n += len(samples)
        el = now - self._t_start
        if el > 5.0:
            self.stats["loss_pct"] = max(0.0, 100.0 * (1.0 - self._n / (el * self.fs)))
        self._put((t_first, samples.astype(np.complex64), self.fs, "full"), block=False)

    def stop(self):
        self._stop.set()
        if self.sdr is not None:
            try:
                self.sdr.cancel_read_async()
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
        if self.sdr is not None:
            try:
                self.sdr.close()
            except Exception:  # noqa: BLE001
                pass
            self.sdr = None

    def set_freq(self, f_vor_hz):
        if self.sdr is not None:
            with self._lock:
                self.sdr.center_freq = f_vor_hz + self.fs / 4

    def set_gain(self, gain_db, agc):
        if self.sdr is not None:
            with self._lock:
                self.sdr.gain = "auto" if agc else gain_db

    def set_ppm(self, ppm):
        if self.sdr is not None and int(ppm) != 0:
            with self._lock:
                try:
                    self.sdr.freq_correction = int(ppm)
                except Exception:  # noqa: BLE001  (librtlsdr recusa valor igual ao atual)
                    pass

    def set_bias_tee(self, on):
        if self.sdr is not None:
            with self._lock:
                self.sdr.set_bias_tee(bool(on))

    @property
    def actual_gain(self):
        try:
            return self.sdr.gain
        except Exception:  # noqa: BLE001
            return None


class SimSource(Source):
    name = "Simulação"

    def __init__(self, cfg: AppConfig, fs: float = 1.024e6, realtime: bool = True):
        super().__init__()
        self.sim = VorSimulator(cfg, fs=fs, seed=cfg.sim.seed or None)
        self.kind = "full" if fs > 100e3 else "wide"
        self.realtime = realtime
        self.stats["fs"] = fs
        self.messages = self.sim.messages

    def _run(self):
        t_wall0 = time.time()
        while not self._stop.is_set():
            t_first, x = self.sim.next_block()
            self._put((t_first, x, self.sim.fs, self.kind), block=True)
            if self.realtime:
                ahead = self.sim.t - (time.time() - t_wall0)
                if ahead > 0:
                    time.sleep(ahead)


class ReplaySource(Source):
    """Reproduz SigMF (cf32_le ou cu8) ou .cu8 cru (rtl_sdr, 1,024 MS/s)."""
    name = "Replay"

    def __init__(self, path: str, cfg: AppConfig, speed: float = 1.0):
        super().__init__()
        p = Path(path)
        self.t0 = time.time()
        if p.suffix in (".sigmf-meta", ".sigmf-data"):
            meta_p = p.with_suffix(".sigmf-meta")
            self.data_p = p.with_suffix(".sigmf-data")
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            g = meta["global"]
            self.dtype = g["core:datatype"]
            self.fs = float(g["core:sample_rate"])
            cap = (meta.get("captures") or [{}])[0]
            if "core:datetime" in cap:
                try:
                    self.t0 = datetime.fromisoformat(cap["core:datetime"].replace("Z", "+00:00")).timestamp()
                except ValueError:
                    pass
            self.wide_offset_applied_hz = float(g.get("vorradar:fine_offset_hz", 0.0))
            self.f_center = cap.get("core:frequency")
        else:
            self.data_p = p
            self.dtype = "cu8"
            self.fs = cfg.rf.sample_rate
            self.f_center = None
        if abs(self.fs - 1.024e6) < 1:
            self.kind = "full"
        elif abs(self.fs - FS_WIDE) < 1:
            self.kind = "wide"
        else:
            raise ValueError(f"taxa {self.fs:.0f} S/s não suportada (use 1,024 MS/s ou 32 kS/s)")
        if self.dtype not in ("cf32_le", "cu8"):
            raise ValueError(f"tipo {self.dtype} não suportado (cf32_le ou cu8)")
        self.speed = speed
        self.stats["fs"] = self.fs
        bps = 8 if self.dtype == "cf32_le" else 2
        self.total = self.data_p.stat().st_size // bps
        self.progress = 0.0

    def _run(self):
        blk = int(self.fs * 0.256) // 4 * 4
        if self.dtype == "cf32_le":
            data = np.memmap(self.data_p, dtype=np.complex64, mode="r")
        else:
            data = np.memmap(self.data_p, dtype=np.uint8, mode="r")
        t_wall0 = time.time()
        i = 0
        while i < self.total and not self._stop.is_set():
            j = min(i + blk, self.total)
            if self.dtype == "cf32_le":
                x = np.array(data[i:j])
            else:
                u = (np.asarray(data[2 * i: 2 * j], dtype=np.float32) - 127.5) / 127.5
                x = (u[0::2] + 1j * u[1::2]).astype(np.complex64)
            self._put((self.t0 + i / self.fs, x, self.fs, self.kind), block=True)
            i = j
            self.progress = i / self.total
            ahead = (i / self.fs) / self.speed - (time.time() - t_wall0)
            if ahead > 0:
                time.sleep(ahead)
