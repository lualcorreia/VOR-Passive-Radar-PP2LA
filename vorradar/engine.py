"""Motor de processamento em thread própria.

fila IQ → DDC → FFT/integração → referência (0 Hz) → fundo → CFAR → picos
      → rastreador → estimadores físicos (v_b, df/dt, CPA, radial exp.)
      → logs CSV / gravação SigMF → FrameResult para a interface.
"""
from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import AppConfig, FS_WIDE, FS_NB, KT
from .dsp import DDC, SpectrumAnalyzer, Background, ReferenceEstimator, Cfar, detect_peaks
from .physics import fit_cpa, bistatic_rate
from .recorder import IQRecorder, CsvLogger, utc_iso, local_str
from .tracker import Tracker, Track, Detection
from .vor_azimuth import VorAzimuthEstimator


@dataclass
class FrameResult:
    t: float
    dt: float
    freqs: np.ndarray
    psd_db: np.ndarray       # PSD integrada (dBFS/Hz)
    det_db: np.ndarray       # espectro de detecção: SNR local por bin (dB)
    noise_db: np.ndarray     # piso de ruído local (dBFS/Hz)
    thr_db: np.ndarray       # limiar de detecção (dBFS/Hz)
    f_ref: float
    ref_state: str
    ref_prom_db: float
    detections: list
    tracks: list
    finished: list
    stats: dict


class RadarEngine:
    def __init__(self, cfg: AppConfig, on_frame, on_message, on_end=None):
        self.cfg = cfg
        self.on_frame = on_frame
        self.msg = on_message
        self.on_end = on_end
        self.q: queue.Queue = queue.Queue(maxsize=48)
        self.cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.source = None
        self.last_t = 0.0

    # ------------------------------------------------------------ controle
    def start(self, source) -> None:
        self.source = source
        self._build()
        self._stop.clear()
        source.start(self.q)
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.source is not None:
            self.source.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def post(self, fn) -> None:
        self.cmds.put(fn)

    def rebuild_analysis(self) -> None:
        self.post(self._build_analysis)

    def record_manual(self, seconds: float = 30.0) -> None:
        def fn():
            c = self.cfg
            if self.recorder.active is not None:
                self.msg("Já existe uma gravação IQ em andamento.")
                return
            path = self.recorder.start("manual", c.f_carrier_hz, c.rf.fine_offset_hz, self.f_ref)
            if path:
                self.recorder.active["stop_at"] = self.last_t + seconds
                self.msg(f"Gravando IQ manual ({seconds:.0f} s + pré-buffer): {path}")
        self.post(fn)

    # ------------------------------------------------------------ construção
    def _build(self) -> None:
        c = self.cfg
        self.ddc = DDC(c.rf.sample_rate)
        self.nb_count = 0
        self.t_stream0 = None
        self.tracker = Tracker(c.tracker, c.wavelength)
        rec = c.record
        self.recorder = IQRecorder(rec.log_dir, rec.prebuffer_s, rec.postbuffer_s, rec.max_record_s)
        self.logger = CsvLogger(rec.log_dir)
        self.az = VorAzimuthEstimator()
        self.az_ring: deque = deque()
        self.az_ring_n = 0
        self.az_last_t = 0.0
        self.az_t_end = 0.0
        self.rec_uid = None
        self.f_ref = 0.0
        self.n_frames = 0
        self.proc_ms: deque = deque(maxlen=40)
        self._build_analysis()

    def _build_analysis(self) -> None:
        d = self.cfg.dsp
        self.an = SpectrumAnalyzer(FS_NB, d.nfft, d.overlap, d.window, d.n_integrate)
        self.an_idx0 = self.nb_count
        self.bg = Background(d.bg_tau_s, self.an.dt)
        self.ref = ReferenceEstimator(self.an.freqs, self.an.dt, d.ref_tau_s, d.ref_search_hz,
                                      d.ref_min_prominence_db)
        self.cfar = Cfar(d.cfar_guard, d.cfar_ref, d.cfar_mode, d.n_integrate)

    def rebuild_cfar(self) -> None:
        d = self.cfg.dsp
        self.post(lambda: setattr(self, "cfar", Cfar(d.cfar_guard, d.cfar_ref, d.cfar_mode, d.n_integrate)))

    # ------------------------------------------------------------ laço
    def _run(self) -> None:
        while not self._stop.is_set():
            while True:
                try:
                    fn = self.cmds.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    self.msg("Erro em comando: " + traceback.format_exc(limit=2))
            try:
                item = self.q.get(timeout=0.25)
            except queue.Empty:
                src = self.source
                if src is not None and src.finished:
                    self.msg(f"Fonte encerrada com erro: {src.error}" if src.error
                             else "Fonte encerrada (fim do arquivo).")
                    break
                continue
            t0 = time.perf_counter()
            try:
                self._process_block(*item)
            except Exception:  # noqa: BLE001
                self.msg("Erro no processamento: " + traceback.format_exc(limit=3))
            self.proc_ms.append((time.perf_counter() - t0) * 1e3)
        self._finalize()
        if self.on_end:
            self.on_end()

    def _finalize(self) -> None:
        try:
            for tr in self.tracker.flush():
                self._finish_track(tr)
            p = self.recorder.close()
            if p:
                self.msg(f"IQ salvo: {p}")
        except Exception:  # noqa: BLE001
            self.msg("Erro ao finalizar: " + traceback.format_exc(limit=2))

    def _process_block(self, t_first: float, x: np.ndarray, fs: float, kind: str) -> None:
        if self.t_stream0 is None:
            self.t_stream0 = t_first
        if kind == "full":
            if abs(fs - self.ddc.fs_in) > 1:
                self.ddc = DDC(fs)
            w = self.ddc.full_to_wide(x)
        else:
            w = x.astype(np.complex64, copy=False)
        shift = self.cfg.rf.fine_offset_hz - getattr(self.source, "wide_offset_applied_hz", 0.0)
        w = self.ddc.nco(w, shift)
        done = self.recorder.push(t_first, w)
        if done:
            self.msg(f"IQ salvo: {done}")
        if self.cfg.record.azimuth_enabled:
            self.az_ring.append(w)
            self.az_ring_n += len(w)
            keep = int((self.cfg.record.azimuth_int_s + 1.0) * FS_WIDE)
            while self.az_ring and self.az_ring_n - len(self.az_ring[0]) >= keep:
                self.az_ring_n -= len(self.az_ring.popleft())
            self.az_t_end = t_first + len(w) / FS_WIDE
        nb = self.ddc.wide_to_nb(w)
        blk_idx0 = self.nb_count
        self.nb_count += len(nb)
        for tc, p in self.an.push(nb):
            idx = self.an_idx0 + tc * FS_NB
            self._frame(t_first + (idx - blk_idx0) / FS_NB, p)

    # ------------------------------------------------------------ quadro
    def _frame(self, t: float, p: np.ndarray) -> None:
        c = self.cfg
        d = c.dsp
        self.last_t = t
        self.n_frames += 1
        self.bg.tau_s = d.bg_tau_s
        self.ref.search_hz = d.ref_search_hz
        self.ref.min_prom_db = d.ref_min_prominence_db
        self.ref.update(p)
        if d.auto_ref and self.ref.f_ref is not None:
            f_ref = self.ref.f_ref
            state = "travada" if self.ref.locked else "retida"
        else:
            f_ref = 0.0
            state = "manual"
        self.f_ref = f_ref
        self.ref_state = state
        bgv = self.bg.value
        use_bg = d.bg_suppress and bgv is not None
        s = p / np.maximum(bgv, 1e-30) if use_bg else p
        noise_s = self.cfar.noise(s)
        snr_lin = s / np.maximum(noise_s, 1e-30)
        thr_lin = 10 ** (d.threshold_db / 10)
        warm = (not d.bg_suppress) or self.bg.age_s >= min(10.0, d.bg_tau_s)
        peaks = detect_peaks(s, noise_s, self.an.freqs, f_ref, d.threshold_db, d.max_doppler_hz,
                             d.exclude_hz, d.exclude_am30) if warm else []
        mask = np.convolve(snr_lin > thr_lin, np.ones(5), mode="same") > 0
        self.bg.update(p, mask)
        noise_raw = noise_s * bgv if use_bg else noise_s
        psd_db = 10 * np.log10(np.maximum(p, 1e-30))
        dets = [Detection(t, pk.f_abs - f_ref, pk.f_abs, pk.snr_db, float(psd_db[pk.idx])) for pk in peaks]

        def dead(f, d=d):
            return abs(f) <= d.exclude_hz + 1.0 or (d.exclude_am30 and abs(abs(f) - 30.0) <= 2.5)

        new_conf, finished = self.tracker.update(t, dets, self.an.dt, dead_zone=dead)
        for tr in new_conf:
            self.msg(f"Alvo #{tr.uid} confirmado: Δf = {tr.f[-1]:+.1f} Hz, SNR {tr.snr[-1]:.1f} dB")
            if c.record.iq_on_detect and self.recorder.active is None:
                path = self.recorder.start(f"trk{tr.uid}", c.f_carrier_hz, c.rf.fine_offset_hz, f_ref,
                                           {"track_id": tr.uid})
                if path:
                    tr.iq_file = path
                    self.rec_uid = tr.uid
        live = {tr.uid for tr in self.tracker.tracks if tr.confirmed}
        if self.recorder.active is not None and self.rec_uid is not None and self.rec_uid not in live:
            self.recorder.request_stop(t)
            self.rec_uid = None
        if c.record.azimuth_enabled:
            self._azimuth(t)
        for tr in self.tracker.tracks:
            if tr.confirmed and len(tr.t) >= 15 and t - tr.fit_t > 2.0:
                tr.fit = fit_cpa(tr.t, tr.f, c.wavelength, c.tracker.v_max_ms)
                tr.fit_t = t
        events = [self._finish_track(tr) for tr in finished]

        st = dict(getattr(self.source, "stats", {}))
        st.update(queue=self.q.qsize(), proc_ms=float(np.mean(self.proc_ms)) if self.proc_ms else 0.0,
                  frames=self.n_frames, recording=self.recorder.active is not None,
                  n_tentative=sum(1 for tr in self.tracker.tracks if not tr.confirmed),
                  bin_hz=self.an.bin_hz, enbw_hz=self.an.enbw_hz, window_s=self.an.nfft / FS_NB)
        res = FrameResult(t=t, dt=self.an.dt, freqs=self.an.freqs, psd_db=psd_db,
                          det_db=10 * np.log10(np.maximum(snr_lin, 1e-30)),
                          noise_db=10 * np.log10(np.maximum(noise_raw, 1e-30)),
                          thr_db=10 * np.log10(np.maximum(noise_raw, 1e-30)) + d.threshold_db,
                          f_ref=f_ref, ref_state=state, ref_prom_db=self.ref.prominence_db,
                          detections=dets, tracks=[self._view(tr) for tr in self.tracker.tracks if tr.confirmed],
                          finished=events, stats=st)
        self.on_frame(res)

    def _azimuth(self, t: float) -> None:
        rc = self.cfg.record
        if t - self.az_last_t < rc.azimuth_int_s:
            return
        cands = [tr for tr in self.tracker.tracks if tr.confirmed and t - tr.last_t < 1.0]
        n_need = int(rc.azimuth_int_s * FS_WIDE)
        if not cands or self.az_ring_n < n_need:
            return
        tr = max(cands, key=lambda z: z.snr[-1])
        x = np.concatenate(list(self.az_ring))[-n_need:]
        t_mid = self.az_t_end - n_need / FS_WIDE / 2
        s_abs, fa_last = tr.slope_abs(5.0)
        f_mid = fa_last + s_abs * (t_mid - tr.last_t)
        tr.radial = self.az.estimate(x, f_mid, s_abs, self.cfg.geo.radial_cal_deg)
        self.az_last_t = t

    # ------------------------------------------------------------ saídas
    def _view(self, tr: Track) -> dict:
        lam = self.cfg.wavelength
        s, f0 = tr.slope(5.0)
        vb = -f0 * lam
        return dict(uid=tr.uid, t=np.array(tr.t), f=np.array(tr.f), fa=np.array(tr.fa),
                    snr=np.array(tr.snr), t0=tr.t0, df=f0, dfdt=s, vb_ms=vb, vb_kt=vb / KT,
                    vb_kmh=vb * 3.6, snr_now=tr.snr[-1], snr_peak=max(tr.snr), status=tr.status(),
                    implausible=tr.implausible, fit=tr.fit, radial=tr.radial, iq=tr.iq_file)

    def _finish_track(self, tr: Track) -> dict:
        c = self.cfg
        lam = c.wavelength
        t = np.array(tr.t)
        f = np.array(tr.f)
        snr = np.array(tr.snr)
        fit = fit_cpa(t, f, lam, c.tracker.v_max_ms)
        tr.fit = fit
        vb = bistatic_rate(f, lam)
        k = min(9, max(3, len(f) // 5 * 2 + 1))
        fsm = np.convolve(f, np.ones(k) / k, mode="valid")
        tsm = np.convolve(t, np.ones(k) / k, mode="valid")
        dfdt = np.abs(np.gradient(fsm, tsm)) if len(fsm) > 2 else np.array([0.0])
        i_pk = int(np.argmax(snr))
        r = tr.radial or {}
        ev = {
            "id": tr.uid, "inicio_utc": utc_iso(t[0]), "fim_utc": utc_iso(t[-1]),
            "duracao_s": round(tr.duration, 1), "n_pontos": len(t), "f_vor_mhz": c.rf.vor_freq_mhz,
            "ref_hz": round(self.f_ref, 2), "ref_estado": getattr(self, "ref_state", ""), "df_max_hz": round(float(f.max()), 2),
            "df_min_hz": round(float(f.min()), 2),
            "vb_inicio_kt": round(float(vb[:3].mean() / KT), 1), "vb_fim_kt": round(float(vb[-3:].mean() / KT), 1),
            "dfdt_max_abs_hz_s": round(float(np.percentile(dfdt, 95)), 3),
            "snr_pico_db": round(float(snr[i_pk]), 1), "t_snr_pico_utc": utc_iso(t[i_pk]),
            "ajuste_msg": fit.message, "arquivo_iq": tr.iq_file or "",
        }
        if fit.ok:
            ev.update({
                "cpa_utc": utc_iso(fit.t0), "cpa_local": local_str(fit.t0),
                "cpa_sigma_s": round(fit.t0_sigma, 2), "cpa_observado": fit.observed,
                "doppler_nulo_utc": utc_iso(fit.t_zero) if math.isfinite(fit.t_zero) else "",
                "doppler_nulo_local": local_str(fit.t_zero) if math.isfinite(fit.t_zero) else "",
                "ajuste_A_hz": round(fit.A_hz, 2), "ajuste_tau_s": round(fit.tau_s, 2),
                "ajuste_c_hz": round(fit.c_hz, 2), "dfdt_cpa_hz_s": round(fit.slope_cpa, 3),
                "ajuste_r2": round(fit.r2, 4), "ajuste_rms_hz": round(fit.rms_hz, 3),
                "v_perna_unica_kt": round(fit.v_single / KT, 1), "v_monostatico_kt": round(fit.v_mono / KT, 1),
                "d_cpa_perna_unica_km": round(fit.d_single_km, 2), "d_cpa_monostatico_km": round(fit.d_mono_km, 2),
            })
        if r:
            ev.update({"radial_exp_deg": round(r.get("radial_deg", math.nan), 1) if r.get("valid") else "",
                       "radial_snr_am_db": round(r.get("snr_am_db", math.nan), 1),
                       "radial_snr_fm_db": round(r.get("snr_fm_db", math.nan), 1)})
        try:
            self.logger.log_event(ev)
            self.logger.log_points([
                {"id": tr.uid, "t_utc": utc_iso(ti), "epoch_s": round(ti, 3), "df_hz": round(fi, 3),
                 "f_eixo_hz": round(fa, 3), "vb_ms": round(-fi * lam, 2), "vb_kt": round(-fi * lam / KT, 2),
                 "snr_db": round(si, 2), "p_dbfs_hz": round(pi, 2)}
                for ti, fi, fa, si, pi in zip(tr.t, tr.f, tr.fa, tr.snr, tr.pdb)])
        except OSError as e:
            self.msg(f"Falha ao gravar CSV: {e}")
        self.msg(f"Passagem #{tr.uid} encerrada ({tr.duration:.0f} s, SNR pico {snr[i_pk]:.1f} dB) — "
                 f"CPA: {ev.get('cpa_local', '—')} local; {fit.message}")
        ev["_t"], ev["_f"], ev["_fa"], ev["_fit"] = t, f, np.array(tr.fa), fit
        return ev
