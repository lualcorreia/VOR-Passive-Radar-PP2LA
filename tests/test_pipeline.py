"""Testes de validação do pipeline (sem interface e sem hardware).

Compara as saídas do DSP, do rastreador e dos estimadores físicos com a
"verdade" do simulador físico. Rodar a partir da raiz do repositório:

    python -m tests.test_pipeline        (relatório detalhado)
    python -m pytest -s                  (mesmos testes via pytest)

Em CI (variável de ambiente CI definida) os limites de desempenho em tempo
real viram apenas informativos, pois dependem da máquina.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, ".")
from vorradar.config import AppConfig, FS_WIDE, FS_NB  # noqa: E402
from vorradar.dsp import DDC, SpectrumAnalyzer, parabolic_offset  # noqa: E402
from vorradar.engine import RadarEngine  # noqa: E402
from vorradar.physics import fit_cpa, s_curve  # noqa: E402
from vorradar.simulator import VorSimulator  # noqa: E402
from vorradar.vor_azimuth import VorAzimuthEstimator  # noqa: E402

OK = True
ON_CI = bool(os.environ.get("CI"))


def check(name, cond, info="", perf=False):
    """Registra e imprime uma verificação; sob pytest, falha o teste."""
    global OK
    if perf and ON_CI and not cond:
        print(f"[INFO ] {name} {info} (limite de desempenho ignorado em CI)")
        return
    print(f"[{'OK ' if cond else 'FALHA'}] {name} {info}")
    OK &= bool(cond)
    if "pytest" in sys.modules:
        assert cond, f"{name} {info}"


def test_ddc_tone():
    fs = 1.024e6
    f_tone = 123.4
    ddc = DDC(fs)
    an = SpectrumAnalyzer(FS_NB, 8192, 0.5, "hann", 1)
    rng = np.random.default_rng(1)
    n_tot, k = 0, 0
    nb_all = []
    t_proc = 0.0
    for b in range(40):
        n = 262144
        t = (n_tot + np.arange(n)) / fs
        x = 0.05 * np.exp(2j * np.pi * (-fs / 4 + f_tone) * t) + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        n_tot += n
        t0 = time.perf_counter()
        nb = ddc.wide_to_nb(ddc.nco(ddc.full_to_wide(x.astype(np.complex64)), 0.0))
        t_proc += time.perf_counter() - t0
        nb_all.append(nb)
    nb = np.concatenate(nb_all)
    check("DDC: nº de amostras na saída", abs(len(nb) - n_tot / 256) <= 2, f"({len(nb)} vs {n_tot / 256:.0f})")
    frames = an.push(nb)
    p = frames[-1][1]
    i = int(np.argmax(p))
    f_est = an.freqs[i] + parabolic_offset(10 * np.log10(p[i - 1:i + 2])) * an.bin_hz
    check("DDC+FFT: frequência do tom", abs(f_est - f_tone) < 0.1, f"(estimado {f_est:.3f} Hz, real {f_tone} Hz)")
    rt = (n_tot / fs) / t_proc
    check("DDC: velocidade ≥ 5× tempo real", rt > 5, f"({rt:.1f}× tempo real)", perf=True)


def run_sim(cfg, ac_kwargs, fs=FS_WIDE, extra_s=12.0, t0_epoch=1.8e9):
    cfg.record.log_dir = "logs/_testes"
    sim = VorSimulator(cfg, fs=fs, t0_epoch=t0_epoch, seed=7, spawn=False)
    ac = sim.add_aircraft(**ac_kwargs)
    frames, events, msgs = [], [], []
    eng = RadarEngine(cfg, frames.append, msgs.append)
    eng._build()
    kind = "full" if fs > 100e3 else "wide"
    t_sim = t_eng = 0.0
    while sim.t < ac.t_end + extra_s:
        a = time.perf_counter()
        tf, x = sim.next_block()
        b = time.perf_counter()
        eng._process_block(tf, x, fs, kind)
        t_eng += time.perf_counter() - b
        t_sim += b - a
    eng._finalize()   # fecha gravações IQ abertas
    for fr in frames:
        events.extend(fr.finished)
    return sim, ac, frames, events, msgs, t_sim, t_eng


def test_tracking_and_cpa():
    cfg = AppConfig()
    cfg.sim.lo_error_hz = 37.0
    cfg.sim.direct_cn0_dbhz = 28.0
    kw = dict(speed=200.0, d_cpa=6e3, heading_deg=100.0, alt=8e3, t_to_cpa=80.0, life_after=80.0, cn0_peak=28.0)
    sim, ac, frames, events, msgs, ts, te = run_sim(cfg, kw)
    dur = sim.t
    print(f"    (sim {dur:.0f} s: gerar {ts:.1f} s, processar {te:.1f} s → {dur / te:.0f}× tempo real)")
    for m in msgs + sim.messages:
        print("    ", m)
    ref = frames[-1].f_ref
    check("Referência automática ≈ erro de LO simulado", abs(ref - 37.0) < 0.5, f"(ref {ref:.2f} Hz, LO +37 Hz)")
    check("Exatamente 1 passagem detectada", len(events) == 1, f"({len(events)} eventos)")
    if not events:
        return
    ev = events[0]
    t = ev["_t"]
    f = ev["_f"]
    f_true = ac.doppler(t - sim.t0_epoch, sim.tx, sim.lam)
    err = f - f_true
    rms = float(np.sqrt(np.mean(err ** 2)))
    check("Δf medido vs verdade (RMS)", rms < 1.0, f"({rms:.2f} Hz RMS, {len(t)} pontos, {t[-1] - t[0]:.0f} s)")
    vb_err = rms * sim.lam
    print(f"    → erro de v_b equivalente: {vb_err:.2f} m/s ({vb_err / 0.514444:.1f} kt)")
    # cruzamento verdadeiro do Doppler bistático (zero)
    g = lambda ts_: float(ac.doppler(np.array([ts_]), sim.tx, sim.lam)[0])
    tz = brentq(g, ac.t_cpa - 70, ac.t_cpa + 70)
    fit = ev["_fit"]
    dt_cpa = (fit.t0 - sim.t0_epoch) - ac.t_cpa
    check("Ajuste S convergiu", fit.ok and fit.r2 > 0.99, f"(R²={fit.r2:.4f}; msg='{fit.message}')")
    print(f"    [INFO] t0 do ajuste (CPA perna dominante) − CPA real ao receptor = {dt_cpa:+.2f} s "
          f"(viés do modelo de perna única em geometria bistática; VOR a {sim.L / 1e3:.0f} km)")
    dt_z = (fit.t_zero - sim.t0_epoch) - tz
    check("Doppler nulo (mínimo de R_T+R_R) vs verdade", abs(dt_z) < 2.0, f"(erro {dt_z:+.2f} s)")
    print(f"    → v perna única {fit.v_single:.0f} m/s | quase-monostático {fit.v_mono:.0f} m/s "
          f"(real {ac.speed:.0f} m/s); d_CPA {fit.d_mono_km:.1f}–{fit.d_single_km:.1f} km (real {ac.d_cpa_m / 1e3:.1f} km)")
    check("Velocidade real dentro do intervalo das hipóteses", fit.v_mono * 0.8 <= ac.speed <= fit.v_single * 1.2)


def test_azimuth():
    cfg = AppConfig()
    cfg.sim.direct_cn0_dbhz = 0.0   # sem visada direta
    cfg.sim.lo_error_hz = 37.0
    sim = VorSimulator(cfg, fs=FS_WIDE, t0_epoch=1.8e9, seed=3, spawn=False)
    ac = sim.add_aircraft(speed=180.0, d_cpa=5e3, heading_deg=30.0, alt=6e3, t_to_cpa=40.0,
                          life_after=40.0, cn0_peak=50.0)
    est = VorAzimuthEstimator()
    for t_mid_target, cn0 in ((20.0, 50.0), (35.0, 50.0)):
        while sim.t < t_mid_target - 2.0:
            sim.next_block()
        blocks = []
        t_start = sim.t
        while sim.t < t_start + 4.0:
            blocks.append(sim.next_block()[1])
        x = np.concatenate(blocks)
        t_mid = t_start + len(x) / FS_WIDE / 2
        f_mid = float(ac.doppler(np.array([t_mid]), sim.tx, sim.lam)[0]) + 37.0
        f2 = float(ac.doppler(np.array([t_mid + 0.5]), sim.tx, sim.lam)[0]) + 37.0
        slope = (f2 - (f_mid)) / 0.5
        pm = ac.pos(np.array([t_mid]))[0] - sim.tx
        th_true = math.degrees(math.atan2(pm[0], pm[1])) % 360
        r = est.estimate(x, f_mid, slope)
        e = (r["radial_deg"] - th_true + 180) % 360 - 180
        check(f"Azimute exp. (t={t_mid:.0f} s, Δf={f_mid - 37:+.0f} Hz)", r["valid"] and abs(e) < 5,
              f"(estimado {r['radial_deg']:.1f}°, real {th_true:.1f}°, erro {e:+.1f}°, "
              f"SNR AM {r['snr_am_db']:.0f} dB / FM {r['snr_fm_db']:.0f} dB, desvio FM {r['fm_dev_hz']:.0f} Hz)")


def test_fullrate_realtime():
    cfg = AppConfig()
    kw = dict(speed=200.0, d_cpa=6e3, heading_deg=100.0, alt=8e3, t_to_cpa=10.0, life_after=10.0, cn0_peak=30.0)
    sim, ac, frames, events, msgs, ts, te = run_sim(cfg, kw, fs=1.024e6, extra_s=0.0, t0_epoch=1.9e9)
    dur = sim.t
    check("Taxa total 1,024 MS/s: processamento em tempo real", te < 0.5 * dur,
          f"(engine {te:.1f} s, simulador {ts:.1f} s para {dur:.0f} s de sinal)", perf=True)


def test_fit_synthetic():
    lam = 2.5867
    rng = np.random.default_rng(5)
    t = 1.8e9 + np.arange(0, 120, 0.256)
    f = s_curve(t - 1.8e9, 90.0, 61.0, 14.0, -5.0) + 0.4 * rng.standard_normal(len(t))
    fit = fit_cpa(t, f, lam)
    check("Ajuste S sintético", fit.ok and abs(fit.t0 - 1.8e9 - 61.0) < 0.3 and abs(fit.A_hz - 90) < 2,
          f"(t0 {fit.t0 - 1.8e9:.2f} s, A {fit.A_hz:.1f} Hz, τ {fit.tau_s:.1f} s, c {fit.c_hz:.2f} Hz)")


def test_replay():
    """Reproduz o SigMF gravado em test_tracking_and_cpa pela fonte de replay (com threads)."""
    import glob
    import threading
    from vorradar.sources import ReplaySource
    files = sorted(glob.glob("logs/_testes/iq/*_trk*.sigmf-data"), key=os.path.getsize)  # o mais longo
    if not files:   # rodando isolado (pytest -k replay): gera a gravação antes
        test_tracking_and_cpa()
        files = sorted(glob.glob("logs/_testes/iq/*_trk*.sigmf-data"), key=os.path.getsize)
    if not files:
        check("Replay: arquivo SigMF encontrado", False)
        return
    cfg = AppConfig()
    cfg.record.log_dir = "logs/_testes"
    cfg.record.iq_on_detect = False
    cfg.tracker.min_duration_s = 5.0
    frames, msgs = [], []
    done = threading.Event()
    eng = RadarEngine(cfg, frames.append, msgs.append, done.set)
    src = ReplaySource(files[-1], cfg, speed=60.0)
    eng.start(src)
    done.wait(timeout=120)
    eng.stop()
    n_conf = sum(1 for m in msgs if "confirmado" in m)
    dur = src.total / src.fs
    check("Replay SigMF: arquivo lido e alvo reencontrado", src.progress >= 0.999 and n_conf >= 1,
          f"({os.path.basename(files[-1])}, {dur:.0f} s, {len(frames)} quadros, {n_conf} alvo(s))")


if __name__ == "__main__":
    for fn in (test_fit_synthetic, test_ddc_tone, test_tracking_and_cpa, test_replay, test_azimuth,
               test_fullrate_realtime):
        print(f"--- {fn.__name__}")
        fn()
    print("\nRESULTADO GERAL:", "TODOS OK" if OK else "HÁ FALHAS")
    sys.exit(0 if OK else 1)
