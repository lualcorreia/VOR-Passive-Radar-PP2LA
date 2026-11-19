"""Interface gráfica em tempo real (PyQt6 + PyQtGraph).

Layout:
  esquerda  — controles (RF, DSP/detecção, rastreador, exibição, geometria, gravação, simulação)
  centro    — waterfall (com trilhas sobrepostas), PSD instantânea (piso, limiar, picos)
              e histórico Δf×tempo com o ajuste da curva em "S"
  embaixo   — tabelas de alvos ativos e de passagens concluídas, e log
"""
from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from .config import AppConfig, FS_NB, KT
from .engine import RadarEngine, FrameResult
from .physics import haversine_m, bearing_deg, geo_valid, max_plausible_doppler
from .recorder import CsvLogger, EVENT_FIELDS, utc_str, local_str, resolve_dir
from .sources import RtlSdrSource, SimSource, ReplaySource, R820T_GAINS

pg.setConfigOptions(imageAxisOrder="row-major", antialias=False, background="#0f1318", foreground="#cfd6dd")

COLORS = ["#ffd166", "#06d6a0", "#ef476f", "#4cc9f0", "#f78c6b", "#b5e48c", "#c77dff", "#ffadad"]
SOURCES = ["RTL-SDR (USB)", "Simulação (32 kS/s, leve)", "Simulação (1,024 MS/s, cadeia completa)",
           "Replay de arquivo IQ…"]
ACTIVE_COLS = ["ID", "Início UTC", "Início local", "Δf (Hz)", "v_b (kt)", "v_b (km/h)", "df/dt (Hz/s)",
               "SNR (dB)", "SNR pico (dB)", "Status", "CPA est. (local)", "Radial exp."]
EVENT_COLS = [("id", "ID"), ("cpa_local", "CPA local"), ("cpa_utc", "CPA UTC"),
              ("doppler_nulo_local", "Δf=0 local"), ("duracao_s", "Duração (s)"), ("df_max_hz", "Δf máx (Hz)"),
              ("df_min_hz", "Δf mín (Hz)"), ("dfdt_cpa_hz_s", "df/dt CPA (Hz/s)"), ("snr_pico_db", "SNR pico (dB)"),
              ("v_monostatico_kt", "v mín (kt)"), ("v_perna_unica_kt", "v máx (kt)"),
              ("d_cpa_monostatico_km", "d mín (km)"), ("d_cpa_perna_unica_km", "d máx (km)"),
              ("ajuste_r2", "R²"), ("radial_exp_deg", "Radial exp."), ("ajuste_msg", "Ajuste"),
              ("arquivo_iq", "Arquivo IQ")]
CMAP_NAMES = ["inferno", "viridis", "magma", "plasma", "turbo", "CET-L3", "CET-R4"]


def color_for(uid) -> str:
    return COLORS[(int(uid) - 1) % len(COLORS)] if uid else "#888888"


def fnum(v, fmt="{:+.1f}") -> str:
    try:
        return "—" if v is None or not math.isfinite(float(v)) else fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


class Bridge(QtCore.QObject):
    frame = QtCore.pyqtSignal(object)
    message = QtCore.pyqtSignal(str)
    ended = QtCore.pyqtSignal()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: AppConfig, autostart: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Radar passivo bistático VOR — RTL-SDR (Doppler)")
        self.resize(1680, 1000)
        self.bridge = Bridge()
        self.bridge.frame.connect(self.on_frame)
        self.bridge.message.connect(self.log)
        self.bridge.ended.connect(self.on_ended)
        self.engine: RadarEngine | None = None
        self.source = None
        self.wf = None
        self.wf_dt = None
        self.wf_filled = 0
        self.events: list[dict] = []
        self.finished_hist: list[dict] = []
        self.wf_items: dict = {}
        self.wf_labels: dict = {}
        self.hist_items: dict = {}
        self.fit_items: dict = {}
        self.frame_times: deque = deque(maxlen=20)
        self.last: FrameResult | None = None
        self._center = None
        self._msg_idx = 0
        self.cmaps = {}
        for n in CMAP_NAMES:
            try:
                self.cmaps[n] = pg.colormap.get(n)
            except Exception:  # noqa: BLE001
                pass
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)
        self._update_geo_label()
        if autostart is not None:
            self.src_combo.setCurrentIndex(autostart)
            QtCore.QTimer.singleShot(300, self.start)

    # ================================================================ UI
    def _build_ui(self):
        tb = self.addToolBar("Principal")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel(" Fonte: "))
        self.src_combo = QtWidgets.QComboBox()
        self.src_combo.addItems(SOURCES)
        tb.addWidget(self.src_combo)
        self.btn_start = QtWidgets.QPushButton("▶  Iniciar")
        self.btn_start.setMinimumWidth(110)
        self.btn_start.clicked.connect(lambda: self.stop() if self.engine else self.start())
        tb.addWidget(self.btn_start)
        tb.addSeparator()
        for text, fn in (("Auto níveis", self.auto_levels), ("Centralizar", self.recenter),
                         ("Gravar IQ 30 s", self.record_now), ("Exportar CSV…", self.export_csv),
                         ("Pasta de logs", self.open_logs), ("Limpar", self.clear_displays)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            tb.addWidget(b)
        self.lbl_rec = QtWidgets.QLabel("")
        tb.addWidget(self.lbl_rec)

        # ---- painel de controles
        dock = QtWidgets.QDockWidget("Controles", self)
        dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)
        lay.addWidget(self._grp_rf())
        lay.addWidget(self._grp_dsp())
        lay.addWidget(self._grp_tracker())
        lay.addWidget(self._grp_display())
        lay.addWidget(self._grp_geo())
        lay.addWidget(self._grp_record())
        lay.addWidget(self._grp_sim())
        lay.addStretch(1)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(370)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        dock.setWidget(scroll)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # ---- gráficos
        self._build_plots()
        # ---- tabelas / log
        bottom = QtWidgets.QDockWidget("Alvos e eventos", self)
        bottom.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable)
        tabs = QtWidgets.QTabWidget()
        self.tbl_active = self._table(ACTIVE_COLS)
        self.tbl_events = self._table([h for _, h in EVENT_COLS])
        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(2000)
        tabs.addTab(self.tbl_active, "Alvos ativos")
        tabs.addTab(self.tbl_events, "Passagens concluídas")
        tabs.addTab(self.txt_log, "Log")
        bottom.setWidget(tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, bottom)
        self.resizeDocks([bottom], [250], QtCore.Qt.Orientation.Vertical)

        # ---- barra de status
        sb = self.statusBar()
        self.st_src = QtWidgets.QLabel("Parado")
        self.st_ref = QtWidgets.QLabel("Ref: —")
        self.st_perf = QtWidgets.QLabel("")
        self.st_res = QtWidgets.QLabel("")
        for w in (self.st_src, self.st_ref, self.st_res, self.st_perf):
            sb.addPermanentWidget(w, 1)

    def _table(self, cols):
        t = QtWidgets.QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        t.setAlternatingRowColors(True)
        return t

    # ---- helpers de widgets
    def _dspin(self, lo, hi, val, step=1.0, dec=1, suffix="", cb=None, tip=None):
        w = QtWidgets.QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setSingleStep(step)
        w.setValue(val)
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        if cb:
            w.valueChanged.connect(cb)
        if tip:
            w.setToolTip(tip)
        return w

    def _ispin(self, lo, hi, val, cb=None, tip=None, suffix=""):
        w = QtWidgets.QSpinBox()
        w.setRange(lo, hi)
        w.setValue(int(val))
        w.setSuffix(suffix)
        w.setKeyboardTracking(False)
        if cb:
            w.valueChanged.connect(cb)
        if tip:
            w.setToolTip(tip)
        return w

    def _check(self, text, val, cb=None, tip=None):
        w = QtWidgets.QCheckBox(text)
        w.setChecked(bool(val))
        if cb:
            w.toggled.connect(cb)
        if tip:
            w.setToolTip(tip)
        return w

    def _combo(self, items, current, cb=None, tip=None):
        w = QtWidgets.QComboBox()
        w.addItems([str(i) for i in items])
        if str(current) in [str(i) for i in items]:
            w.setCurrentText(str(current))
        if cb:
            w.currentTextChanged.connect(cb)
        if tip:
            w.setToolTip(tip)
        return w

    def _set(self, obj, attr, conv=lambda v: v, after=None):
        def fn(v):
            setattr(obj, attr, conv(v))
            if after:
                after()
        return fn

    # ================================================================ grupos de controles
    def _grp_rf(self):
        rf = self.cfg.rf
        g = QtWidgets.QGroupBox("RF / RTL-SDR")
        f = QtWidgets.QFormLayout(g)
        self.sp_freq = self._dspin(108.0, 117.95, rf.vor_freq_mhz, 0.05, 3, " MHz", self._on_freq,
                                   "Frequência da portadora do VOR (108,00–117,95 MHz). O LO fica em f + 256 kHz "
                                   "para afastar o spike DC do RTL-SDR.")
        f.addRow("Portadora VOR", self.sp_freq)
        f.addRow("Correção do cristal", self._ispin(-100, 100, rf.ppm, self._on_ppm,
                 "PPM aplicado no tuner. TCXO de 1 ppm ≈ ±116 Hz em 116 MHz.", " ppm"))
        self.sp_fine = self._dspin(-3000, 3000, rf.fine_offset_hz, 1.0, 1, " Hz",
                                   self._set(rf, "fine_offset_hz"),
                                   "Offset fino digital: a portadora real está em nominal + offset. "
                                   "Com a referência automática ligada, o resíduo é medido sozinho.")
        f.addRow("Offset fino", self.sp_fine)
        gains = [f"{v:.1f}" for v in R820T_GAINS]
        self.cb_gain = self._combo(gains, f"{rf.gain_db:.1f}", self._on_gain,
                                   "Ganho manual do LNA/mixer (R820T). Nesta instalação, clipping começa em 33,8 dB.")
        f.addRow("Ganho", self.cb_gain)
        self.ck_agc = self._check("AGC do tuner", rf.agc, self._on_agc,
                                  "Desligado por padrão: o AGC pode reduzir o ganho por sinais fortes fora do canal "
                                  "e mascarar reflexões fracas.")
        self.ck_bias = self._check("Bias-T (5 V na antena)", rf.bias_tee, self._on_bias,
                                   "Só ligue com LNA/antena ativa. Antenas com curto DC (dipolo dobrado, loop) "
                                   "drenam corrente.")
        f.addRow(self.ck_agc)
        f.addRow(self.ck_bias)
        return g

    def _grp_dsp(self):
        d = self.cfg.dsp
        g = QtWidgets.QGroupBox("Análise espectral e detecção")
        f = QtWidgets.QFormLayout(g)
        self.cb_nfft = self._combo([1024, 2048, 4096, 8192, 16384], d.nfft, self._on_nfft)
        f.addRow("FFT (pontos @ 4 kS/s)", self.cb_nfft)
        self.lbl_res = QtWidgets.QLabel()
        f.addRow("", self.lbl_res)
        self.cb_ovl = self._combo(["50", "66.7", "75", "87.5"], f"{d.overlap * 100:g}", self._on_overlap)
        f.addRow("Overlap (%)", self.cb_ovl)
        f.addRow("Janela", self._combo(["hann", "blackmanharris"], d.window,
                                       self._set(d, "window", after=self._rebuild)))
        f.addRow("Integração (espectros)", self._ispin(1, 16, d.n_integrate,
                 self._set(d, "n_integrate", int, self._rebuild),
                 "Média incoerente de N espectros: ganho de detectabilidade ≈ 5·log10(N) dB, com borrão temporal."))
        f.addRow(self._check("Remover linhas estáticas (fundo)", d.bg_suppress, self._set(d, "bg_suppress"),
                             "Normaliza pelo espectro médio de longa duração: some a portadora residual, "
                             "clutter e espúrios; restam os traços que se movem."))
        f.addRow("τ do fundo", self._dspin(5, 900, d.bg_tau_s, 5, 0, " s", self._set(d, "bg_tau_s")))
        f.addRow("CFAR", self._combo(["OS", "CA"], d.cfar_mode, self._set(d, "cfar_mode", after=self._rebuild_cfar),
                                     "OS = mediana das células (robusto a alvos vizinhos); CA = média."))
        f.addRow("Células guarda / ref.", self._row(
            self._ispin(0, 20, d.cfar_guard, self._set(d, "cfar_guard", int, self._rebuild_cfar)),
            self._ispin(4, 64, d.cfar_ref, self._set(d, "cfar_ref", int, self._rebuild_cfar))))
        f.addRow("Limiar", self._dspin(3, 30, d.threshold_db, 0.5, 1, " dB", self._set(d, "threshold_db"),
                                       "Limiar acima do ruído local estimado pelo CFAR."))
        f.addRow("Zona morta em 0 Hz (±)", self._dspin(0, 30, d.exclude_hz, 0.5, 1, " Hz", self._set(d, "exclude_hz")))
        f.addRow(self._check("Ignorar bandas AM ±30 Hz da portadora", d.exclude_am30, self._set(d, "exclude_am30")))
        f.addRow("|Δf| máximo", self._dspin(50, 1500, d.max_doppler_hz, 10, 0, " Hz", self._set(d, "max_doppler_hz")))
        f.addRow(self._check("Referência 0 Hz automática (linha estática)", d.auto_ref, self._set(d, "auto_ref"),
                             "Mede o erro conjunto LO+transmissor pela linha de Doppler nulo (difração/clutter)."))
        f.addRow("Busca da referência (±)", self._dspin(20, 1500, d.ref_search_hz, 10, 0, " Hz",
                                                         self._set(d, "ref_search_hz")))
        self._update_res_label()
        return g

    def _grp_tracker(self):
        t = self.cfg.tracker
        g = QtWidgets.QGroupBox("Rastreador")
        f = QtWidgets.QFormLayout(g)
        f.addRow("Janela de associação", self._dspin(1, 50, t.gate_hz, 0.5, 1, " Hz", self._set(t, "gate_hz")))
        f.addRow("Confirmar após", self._dspin(0.5, 30, t.confirm_min_s, 0.5, 1, " s", self._set(t, "confirm_min_s")))
        f.addRow("Tolerância sem detecção", self._dspin(0.5, 60, t.max_miss_s, 0.5, 1, " s", self._set(t, "max_miss_s")))
        f.addRow("Duração mínima p/ log", self._dspin(1, 600, t.min_duration_s, 1, 0, " s",
                                                       self._set(t, "min_duration_s")))
        f.addRow("v máx. plausível", self._dspin(50, 700, t.v_max_ms, 10, 0, " m/s", self._set(t, "v_max_ms"),
                                                 "Δf acima de 2·v_máx/λ é marcado como não físico."))
        return g

    def _grp_display(self):
        s = self.cfg.display
        g = QtWidgets.QGroupBox("Exibição")
        f = QtWidgets.QFormLayout(g)
        f.addRow("Zoom (±)", self._dspin(20, 2000, s.zoom_hz, 50, 0, " Hz", self._on_zoom))
        f.addRow("Waterfall", self._combo(["normalizado", "bruto"], s.wf_mode, self._on_wf_mode,
                                          "normalizado = SNR local por bin (fundo removido); bruto = PSD em dBFS/Hz"))
        self.sp_min = self._dspin(-200, 100, s.wf_min_db, 1, 1, " dB", self._on_levels)
        self.sp_max = self._dspin(-200, 100, s.wf_max_db, 1, 1, " dB", self._on_levels)
        f.addRow("Mín / máx", self._row(self.sp_min, self.sp_max))
        f.addRow("Paleta", self._combo(list(self.cmaps), s.colormap, self._on_cmap))
        self.ck_overlay = self._check("Sobrepor trilhas no waterfall", True)
        f.addRow(self.ck_overlay)
        f.addRow("Histórico Δf×t", self._dspin(30, 3600, s.history_s, 30, 0, " s", self._set(s, "history_s")))
        return g

    def _grp_geo(self):
        geo = self.cfg.geo
        g = QtWidgets.QGroupBox("Geometria (opcional)")
        f = QtWidgets.QFormLayout(g)
        self.ed_rx = QtWidgets.QLineEdit("" if not geo_valid(geo.rx_lat, geo.rx_lon) else f"{geo.rx_lat}, {geo.rx_lon}")
        self.ed_rx.setPlaceholderText("lat, lon (graus decimais)")
        self.ed_vor = QtWidgets.QLineEdit(f"{geo.vor_lat}, {geo.vor_lon}")
        self.ed_rx.editingFinished.connect(self._on_geo)
        self.ed_vor.editingFinished.connect(self._on_geo)
        f.addRow("Receptor", self.ed_rx)
        f.addRow("VOR", self.ed_vor)
        self.lbl_geo = QtWidgets.QLabel()
        self.lbl_geo.setWordWrap(True)
        f.addRow(self.lbl_geo)
        return g

    def _grp_record(self):
        r = self.cfg.record
        g = QtWidgets.QGroupBox("Gravação e módulo experimental")
        f = QtWidgets.QFormLayout(g)
        f.addRow(self._check("Gravar IQ (SigMF) ao detectar alvo", r.iq_on_detect, self._set(r, "iq_on_detect"),
                             "Fluxo de 32 kS/s com pré-buffer; serve para replay e análise offline."))
        f.addRow("Pré-buffer", self._dspin(0, 60, r.prebuffer_s, 1, 0, " s", self._set(r, "prebuffer_s")))
        f.addRow(self._check("[Exp.] Radial pelo eco (30 Hz AM × FM 9960)", r.azimuth_enabled,
                             self._set(r, "azimuth_enabled"),
                             "Experimental: requer C/N0 alto e ausência do sinal direto (subportadoras se "
                             "sobrepõem). Mostra a radial VOR→aeronave quando AM e FM passam de 10 dB de SNR."))
        f.addRow("Integração do azimute", self._dspin(1, 20, r.azimuth_int_s, 1, 0, " s", self._set(r, "azimuth_int_s")))
        f.addRow("Calibração da radial", self._dspin(-180, 180, self.cfg.geo.radial_cal_deg, 0.5, 1, "°",
                                                      self._set(self.cfg.geo, "radial_cal_deg")))
        return g

    def _grp_sim(self):
        s = self.cfg.sim
        g = QtWidgets.QGroupBox("Simulação")
        f = QtWidgets.QFormLayout(g)
        f.addRow("Erro de LO simulado", self._dspin(-500, 500, s.lo_error_hz, 1, 1, " Hz", self._set(s, "lo_error_hz")))
        f.addRow("Resíduo direto (C/N0)", self._dspin(0, 70, s.direct_cn0_dbhz, 1, 1, " dB-Hz",
                                                       self._set(s, "direct_cn0_dbhz"), "0 = sem sinal direto"))
        f.addRow("Eco de pico (C/N0)", self._dspin(5, 70, s.echo_cn0_dbhz, 1, 1, " dB-Hz",
                                                    self._set(s, "echo_cn0_dbhz"),
                                                    "SNR por bin ≈ C/N0 − 10·log10(ENBW); Hann 4096 → ENBW 1,47 Hz"))
        return g

    def _row(self, *ws):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        for x in ws:
            h.addWidget(x)
        return w

    # ================================================================ gráficos
    def _build_plots(self):
        disp = self.cfg.display
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        dash = QtCore.Qt.PenStyle.DashLine
        # waterfall
        self.wf_plot = pg.PlotWidget()
        pi = self.wf_plot.getPlotItem()
        pi.setTitle("Waterfall — rastro Doppler (curva em “S”) em torno da portadora")
        pi.setLabel("bottom", "Frequência relativa à portadora nominal + offset (Hz)")
        pi.setLabel("left", "Tempo (s, 0 = agora)")
        pi.setMouseEnabled(x=True, y=False)
        self.wf_img = pg.ImageItem()
        pi.addItem(self.wf_img)
        cm = self.cmaps.get(disp.colormap) or next(iter(self.cmaps.values()))
        self.cbar = pg.ColorBarItem(values=(disp.wf_min_db, disp.wf_max_db), colorMap=cm, interactive=True,
                                    width=14, label="dB")
        self.cbar.setImageItem(self.wf_img, insert_in=pi)
        self.cbar.sigLevelsChangeFinished.connect(self._on_cbar)
        self.wf_ref = pg.InfiniteLine(angle=90, pen=pg.mkPen("#00e5ff", width=1, style=dash))
        pi.addItem(self.wf_ref)
        # PSD
        self.psd_plot = pg.PlotWidget()
        p2 = self.psd_plot.getPlotItem()
        p2.setTitle("Espectro instantâneo (PSD integrada)")
        p2.setLabel("left", "dBFS/Hz")
        p2.addLegend(offset=(-10, 5), colCount=3)
        self.psd_curve = p2.plot(pen=pg.mkPen("#e6e6e6", width=1), name="PSD")
        self.noise_curve = p2.plot(pen=pg.mkPen("#8a9ba8", width=1, style=dash), name="piso (CFAR)")
        self.thr_curve = p2.plot(pen=pg.mkPen("#ff5d5d", width=1), name="limiar")
        self.det_scatter = pg.ScatterPlotItem(symbol="t", size=12, brush=pg.mkBrush("#ffd166"), pen=None)
        p2.addItem(self.det_scatter)
        self.psd_ref = pg.InfiniteLine(angle=90, pen=pg.mkPen("#00e5ff", width=1, style=dash))
        p2.addItem(self.psd_ref)
        p2.setXLink(pi)
        # histórico
        self.hist_plot = pg.PlotWidget()
        p3 = self.hist_plot.getPlotItem()
        p3.setTitle("Histórico Δf × tempo (pontos) e ajuste da curva em “S” (tracejado)")
        p3.setLabel("left", "Δf (Hz)   [v_b = −λ·Δf]")
        p3.setLabel("bottom", "Tempo (s, 0 = agora)")
        p3.addLine(y=0, pen=pg.mkPen("#556", width=1))
        p3.showGrid(x=True, y=True, alpha=0.2)
        split.addWidget(self.wf_plot)
        split.addWidget(self.psd_plot)
        split.addWidget(self.hist_plot)
        split.setSizes([520, 220, 230])
        self.setCentralWidget(split)
        self._apply_zoom(0.0)

    # ================================================================ callbacks RF/DSP
    def _on_freq(self, v):
        self.cfg.rf.vor_freq_mhz = float(v)
        if self.source:
            self.source.set_freq(v * 1e6)
        if self.engine:
            self.engine.post(lambda: setattr(self.engine.tracker, "lam", self.cfg.wavelength))
        self._update_geo_label()

    def _on_ppm(self, v):
        self.cfg.rf.ppm = int(v)
        if self.source:
            self.source.set_ppm(int(v))

    def _on_gain(self, txt):
        self.cfg.rf.gain_db = float(txt)
        if self.source:
            self.source.set_gain(self.cfg.rf.gain_db, self.cfg.rf.agc)

    def _on_agc(self, on):
        self.cfg.rf.agc = bool(on)
        self.cb_gain.setEnabled(not on)
        if self.source:
            self.source.set_gain(self.cfg.rf.gain_db, self.cfg.rf.agc)

    def _on_bias(self, on):
        if on:
            r = QtWidgets.QMessageBox.question(
                self, "Bias-T", "Ligar 4,5–5 V na entrada da antena?\n\nUse apenas com LNA ou antena ativa. "
                "Antenas com curto DC (dipolo dobrado, loop) podem drenar corrente do dongle.")
            if r != QtWidgets.QMessageBox.StandardButton.Yes:
                self.ck_bias.blockSignals(True)
                self.ck_bias.setChecked(False)
                self.ck_bias.blockSignals(False)
                return
        self.cfg.rf.bias_tee = bool(on)
        if self.source:
            try:
                self.source.set_bias_tee(bool(on))
                self.log(f"Bias-T {'ligado' if on else 'desligado'}.")
            except Exception as e:  # noqa: BLE001
                self.log(f"Falha no Bias-T: {e}")

    def _on_nfft(self, txt):
        self.cfg.dsp.nfft = int(txt)
        self._rebuild()

    def _on_overlap(self, txt):
        self.cfg.dsp.overlap = float(txt) / 100.0
        self._rebuild()

    def _rebuild(self):
        self._update_res_label()
        if self.engine:
            self.engine.rebuild_analysis()

    def _rebuild_cfar(self):
        if self.engine:
            self.engine.rebuild_cfar()

    def _update_res_label(self):
        d = self.cfg.dsp
        hop = d.nfft * (1 - d.overlap) / FS_NB
        self.lbl_res.setText(f"{FS_NB / d.nfft:.3f} Hz/bin · janela {d.nfft / FS_NB:.2f} s · "
                             f"{1 / hop:.1f} linhas/s")

    def _on_zoom(self, v):
        self.cfg.display.zoom_hz = float(v)
        self._apply_zoom(self._center or 0.0)

    def _apply_zoom(self, center):
        z = self.cfg.display.zoom_hz
        self._center = center
        self.wf_plot.setXRange(center - z, center + z, padding=0)

    def recenter(self):
        self._apply_zoom(self.last.f_ref if self.last else 0.0)

    def _on_wf_mode(self, txt):
        self.cfg.display.wf_mode = txt
        self.wf = None
        if txt == "bruto":
            QtCore.QTimer.singleShot(3000, self.auto_levels)
        else:
            self._set_levels(-5.0, 12.0)

    def _on_levels(self, *_):
        self._set_levels(self.sp_min.value(), self.sp_max.value(), from_spin=True)

    def _set_levels(self, lo, hi, from_spin=False):
        self.cfg.display.wf_min_db, self.cfg.display.wf_max_db = float(lo), float(hi)
        self.cbar.setLevels((lo, hi))
        if not from_spin:
            for sp, v in ((self.sp_min, lo), (self.sp_max, hi)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)

    def _on_cbar(self, *_):
        lo, hi = self.cbar.levels()
        self._set_levels(lo, hi)

    def _on_cmap(self, name):
        self.cfg.display.colormap = name
        if name in self.cmaps:
            self.cbar.setColorMap(self.cmaps[name])

    def auto_levels(self):
        if self.wf is None or self.wf_filled < 3:
            return
        data = self.wf[-min(self.wf_filled, 150):]
        (x0, x1), _ = self.wf_plot.getPlotItem().viewRange()
        f = self.wf_freqs
        cols = (f >= x0) & (f <= x1)
        if cols.sum() > 10:
            data = data[:, cols]
        lo, hi = np.nanpercentile(data, 20), np.nanpercentile(data, 99.8)
        self._set_levels(round(float(lo), 1), round(float(max(hi, lo + 3)), 1))

    def _on_geo(self):
        geo = self.cfg.geo
        for ed, a, b in ((self.ed_rx, "rx_lat", "rx_lon"), (self.ed_vor, "vor_lat", "vor_lon")):
            txt = ed.text().strip()
            if not txt:
                setattr(geo, a, math.nan)
                setattr(geo, b, math.nan)
                continue
            try:
                la, lo = (float(v) for v in txt.replace(";", ",").split(",")[:2])
                setattr(geo, a, la)
                setattr(geo, b, lo)
            except ValueError:
                self.log(f"Coordenada inválida: '{txt}' (use: lat, lon)")
        self._update_geo_label()

    def _update_geo_label(self):
        g = self.cfg.geo
        lam = self.cfg.wavelength
        fmax = max_plausible_doppler(self.cfg.tracker.v_max_ms, lam)
        txt = f"λ = {lam:.3f} m · |Δf| físico ≤ {fmax:.0f} Hz (v ≤ {self.cfg.tracker.v_max_ms:.0f} m/s)"
        if geo_valid(g.rx_lat, g.rx_lon, g.vor_lat, g.vor_lon):
            L = haversine_m(g.rx_lat, g.rx_lon, g.vor_lat, g.vor_lon)
            brg = bearing_deg(g.rx_lat, g.rx_lon, g.vor_lat, g.vor_lon)
            txt = (f"Linha de base RX→VOR: {L / 1e3:.1f} km, rumo {brg:.0f}° · "
                   f"elipse R_T+R_R ≥ {L / 1e3:.1f} km\n") + txt
        self.lbl_geo.setText(txt)

    # ================================================================ execução
    def start(self):
        if self.engine:
            return
        sel = self.src_combo.currentIndex()
        try:
            if sel == 0:
                src = RtlSdrSource(self.cfg)
            elif sel == 1:
                src = SimSource(self.cfg, fs=32_000.0)
            elif sel == 2:
                src = SimSource(self.cfg, fs=1.024e6)
            else:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Abrir gravação IQ", str(resolve_dir(self.cfg.record.log_dir) / "iq"),
                    "IQ (*.sigmf-meta *.sigmf-data *.cu8)")
                if not path:
                    return
                src = ReplaySource(path, self.cfg)
            self.engine = RadarEngine(self.cfg, self.bridge.frame.emit, self.bridge.message.emit,
                                      self.bridge.ended.emit)
            self.engine.start(src)
        except Exception as e:  # noqa: BLE001
            self.engine = None
            QtWidgets.QMessageBox.critical(self, "Erro ao iniciar", f"{e.__class__.__name__}: {e}")
            return
        self.source = src
        self._msg_idx = 0
        self.clear_displays()
        self.btn_start.setText("■  Parar")
        self.src_combo.setEnabled(False)
        self.log(f"Iniciado: {SOURCES[sel]} — VOR {self.cfg.rf.vor_freq_mhz:.3f} MHz, "
                 f"ganho {self.cfg.rf.gain_db:.1f} dB{' (AGC)' if self.cfg.rf.agc else ''}")

    def stop(self):
        eng = self.engine
        if eng is None:
            return
        self.engine = None
        eng.stop()
        self.source = None
        self.btn_start.setText("▶  Iniciar")
        self.src_combo.setEnabled(True)
        self.st_src.setText("Parado")
        self.log("Parado.")

    def on_ended(self):
        if self.engine is not None:   # a fonte terminou sozinha (fim do replay ou erro)
            self.stop()

    def record_now(self):
        if self.engine:
            self.engine.record_manual(30.0)

    def clear_displays(self):
        self.wf = None
        for d in (self.wf_items, self.wf_labels, self.hist_items, self.fit_items):
            for it in d.values():
                it.getViewBox() and it.getViewBox().removeItem(it)
            d.clear()
        self.finished_hist.clear()
        self.tbl_active.setRowCount(0)

    def open_logs(self):
        p = resolve_dir(self.cfg.record.log_dir)
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # noqa: S606
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))

    def export_csv(self):
        if not self.events:
            QtWidgets.QMessageBox.information(self, "Exportar CSV", "Nenhuma passagem concluída ainda.\n"
                                              "(Os eventos também são gravados automaticamente em logs/eventos_*.csv)")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar passagens", str(resolve_dir(self.cfg.record.log_dir) /
                                            f"passagens_{datetime.now():%Y%m%d_%H%M%S}.csv"), "CSV (*.csv)")
        if path:
            CsvLogger.export(path, self.events, EVENT_FIELDS)
            self.log(f"Exportado: {path}")

    def log(self, msg: str):
        self.txt_log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {msg}")

    # ================================================================ quadros
    def on_frame(self, fr: FrameResult):
        if self.engine is None:
            return
        self.last = fr
        self.frame_times.append(time.time())
        disp = self.cfg.display
        n = len(fr.freqs)
        if self.wf is None or self.wf.shape[1] != n or abs(self.wf_dt - fr.dt) > 1e-9:
            rows = disp.wf_rows
            fill = disp.wf_min_db if disp.wf_mode == "normalizado" else float(np.median(fr.psd_db))
            self.wf = np.full((rows, n), fill, np.float32)
            self.wf_freqs = fr.freqs
            self.wf_dt = fr.dt
            self.wf_filled = 0
            new_geometry = True
        else:
            new_geometry = False
        self.wf[:-1] = self.wf[1:]
        self.wf[-1] = fr.det_db if disp.wf_mode == "normalizado" else fr.psd_db
        self.wf_filled += 1
        self.wf_img.setImage(self.wf, autoLevels=False)
        if new_geometry:   # setRect só depois de setImage (a escala depende do tamanho da imagem)
            rows = self.wf.shape[0]
            df = fr.freqs[1] - fr.freqs[0]
            self.wf_img.setRect(QtCore.QRectF(fr.freqs[0] - df / 2, -rows * fr.dt, n * df, rows * fr.dt))
            self.wf_plot.setYRange(-rows * fr.dt, 0, padding=0)
        if self.wf_filled == 25:
            self.auto_levels()
        if self._center is None or abs(fr.f_ref - self._center) > max(3.0, 0.1 * disp.zoom_hz):
            self._apply_zoom(fr.f_ref)   # recentra só quando a referência 0 Hz se desloca
        self.wf_ref.setPos(fr.f_ref)
        self.psd_ref.setPos(fr.f_ref)
        self.psd_curve.setData(fr.freqs, fr.psd_db)
        self.noise_curve.setData(fr.freqs, fr.noise_db)
        self.thr_curve.setData(fr.freqs, fr.thr_db)
        self.det_scatter.setData([d.f_abs for d in fr.detections], [d.p_db + 1.5 for d in fr.detections])
        for ev in fr.finished:
            self._add_event(ev)
        self._draw_tracks(fr)
        self._fill_active(fr)

    def _scatter(self, store, plot, uid, size):
        it = store.get(uid)
        if it is None:
            c = QtGui.QColor(color_for(uid))
            c.setAlpha(200)
            it = pg.ScatterPlotItem(size=size, pen=None, brush=pg.mkBrush(c))
            plot.addItem(it)
            store[uid] = it
        return it

    def _draw_tracks(self, fr: FrameResult):
        now = fr.t
        t_wf = self.cfg.display.wf_rows * fr.dt
        t_h = self.cfg.display.history_s
        tracks = list(fr.tracks) + [
            {"uid": ev["id"], "t": ev["_t"], "f": ev["_f"], "fa": ev["_fa"], "fit": ev["_fit"]}
            for ev in self.finished_hist if now - ev["_t"][-1] < max(t_h, t_wf)]
        seen = set()
        for tv in tracks:
            uid = tv["uid"]
            seen.add(uid)
            t = tv["t"]
            sel = t >= now - t_wf
            ln = self.wf_items.get(uid)
            if ln is None:   # linha paralela deslocada +12 Hz, para não cobrir o rastro real
                c = QtGui.QColor(color_for(uid))
                c.setAlpha(170)
                ln = self.wf_plot.plot(pen=pg.mkPen(c, width=1))
                self.wf_items[uid] = ln
                lab = pg.TextItem(f"#{uid}", color=color_for(uid), anchor=(0, 1))
                self.wf_plot.addItem(lab)
                self.wf_labels[uid] = lab
            lab = self.wf_labels[uid]
            if self.ck_overlay.isChecked() and sel.any():
                ln.setData(tv["fa"][sel] + 12.0, t[sel] - now)
                lab.setPos(tv["fa"][sel][-1] + 14.0, t[sel][-1] - now)
                lab.setVisible(True)
            else:
                ln.setData([], [])
                lab.setVisible(False)
            sel = t >= now - t_h
            self._scatter(self.hist_items, self.hist_plot, uid, 4).setData(t[sel] - now, tv["f"][sel])
            fit = tv.get("fit")
            if fit is not None and fit.ok:
                tt = np.linspace(max(t[0], now - t_h), t[-1], 200)
                it = self.fit_items.get(uid)
                if it is None:
                    it = self.hist_plot.plot(pen=pg.mkPen(color_for(uid), width=1.5,
                                                          style=QtCore.Qt.PenStyle.DashLine))
                    self.fit_items[uid] = it
                it.setData(tt - now, fit.curve(tt))
        for store in (self.wf_items, self.wf_labels, self.hist_items, self.fit_items):
            for uid in [u for u in store if u not in seen]:
                it = store.pop(uid)
                if it.getViewBox():
                    it.getViewBox().removeItem(it)
        self.hist_plot.setXRange(-t_h, 0, padding=0.01)

    def _fill_active(self, fr: FrameResult):
        tb = self.tbl_active
        tb.setRowCount(len(fr.tracks))
        for r, tv in enumerate(sorted(fr.tracks, key=lambda z: z["uid"])):
            fit = tv["fit"]
            cpa = "—"
            if fit is not None and fit.ok:
                dt = fit.t0 - fr.t
                cpa = f"{local_str(fit.t0)} ({'em ' if dt > 0 else 'há '}{abs(dt):.0f} s)"
                if not fit.observed:
                    cpa += " extrap."
            rad = tv["radial"]
            rad_s = "—" if not rad else (f"{rad['radial_deg']:.0f}°" if rad.get("valid") else
                                         f"SNR {rad.get('snr_am_db', 0):.0f}/{rad.get('snr_fm_db', 0):.0f} dB")
            status = tv["status"] + (" ⚠ não físico" if tv["implausible"] else "")
            vals = [str(tv["uid"]), utc_str(tv["t0"]), local_str(tv["t0"]), fnum(tv["df"]),
                    fnum(tv["vb_kt"], "{:+.0f}"), fnum(tv["vb_kmh"], "{:+.0f}"), fnum(tv["dfdt"], "{:+.2f}"),
                    fnum(tv["snr_now"], "{:.1f}"), fnum(tv["snr_peak"], "{:.1f}"), status, cpa, rad_s]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(QtGui.QColor(color_for(tv["uid"])))
                tb.setItem(r, c, it)

    def _add_event(self, ev: dict):
        self.events.append(ev)
        self.finished_hist.append(ev)
        tb = self.tbl_events
        r = tb.rowCount()
        tb.insertRow(r)
        for c, (k, _) in enumerate(EVENT_COLS):
            v = ev.get(k, "")
            if k == "arquivo_iq" and v:
                v = os.path.basename(v)
            tb.setItem(r, c, QtWidgets.QTableWidgetItem("—" if v in ("", None) else str(v)))
        tb.scrollToBottom()

    def _tick(self):
        src = self.source
        if src is not None:
            msgs = getattr(src, "messages", [])
            while self._msg_idx < len(msgs):
                self.log(msgs[self._msg_idx])
                self._msg_idx += 1
        fr = self.last
        if self.engine is None or fr is None:
            return
        st = fr.stats
        fps = (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0]) \
            if len(self.frame_times) > 2 else 0.0
        gain = getattr(src, "actual_gain", None)
        prog = f" · {100 * src.progress:.0f}%" if hasattr(src, "progress") else ""
        self.st_src.setText(
            f"{getattr(src, 'name', '?')}{prog} · {st.get('fs', 0) / 1e3:.0f} kS/s · perdas {st.get('loss_pct', 0):.2f}% · "
            f"descartes fila {st.get('queue_drops', 0)}" + (f" · ganho {gain} dB" if gain is not None else ""))
        ppm = fr.f_ref / self.cfg.f_carrier_hz * 1e6
        txt = (f"Ref 0 Hz: {fr.f_ref:+.2f} Hz ({ppm:+.3f} ppm) · {fr.ref_state} · "
               f"proeminência {fr.ref_prom_db:.1f} dB")
        if fr.ref_state == "manual":
            unc = self.cfg.f_carrier_hz * 1e-6
            txt += (f" · ⚠ sem linha de Doppler nulo: Δf e v_b absolutos incluem o erro do LO "
                    f"(±{unc:.0f} Hz/ppm); df/dt e CPA não são afetados")
            self.st_ref.setStyleSheet("color:#ffb74d")
        else:
            self.st_ref.setStyleSheet("")
        self.st_ref.setText(txt)
        self.st_ref.setToolTip(txt)
        self.st_res.setText(f"{st.get('bin_hz', 0):.3f} Hz/bin · ENBW {st.get('enbw_hz', 0):.2f} Hz · "
                            f"janela {st.get('window_s', 0):.2f} s · tentativas {st.get('n_tentative', 0)}")
        lat = time.time() - fr.t
        self.st_perf.setText(f"{fps:.1f} quadros/s · DSP {st.get('proc_ms', 0):.1f} ms/bloco · "
                             f"fila {st.get('queue', 0)} · latência {lat:.1f} s")
        self.lbl_rec.setText("  ● GRAVANDO IQ" if st.get("recording") else "")
        self.lbl_rec.setStyleSheet("color:#ff5d5d;font-weight:bold")

    def closeEvent(self, ev):
        self.stop()
        try:
            self.cfg.record.log_dir = getattr(self, "saved_logdir", self.cfg.record.log_dir)
            self.cfg.save()
        except OSError:
            pass
        super().closeEvent(ev)


def _dark(app: QtWidgets.QApplication):
    app.setStyle("Fusion")
    p = QtGui.QPalette()
    C = QtGui.QPalette.ColorRole
    for role, col in ((C.Window, "#1b2027"), (C.WindowText, "#dfe5ea"), (C.Base, "#12161b"),
                      (C.AlternateBase, "#1e242c"), (C.Text, "#dfe5ea"), (C.Button, "#262d36"),
                      (C.ButtonText, "#dfe5ea"), (C.Highlight, "#2d6cdf"), (C.HighlightedText, "#ffffff"),
                      (C.ToolTipBase, "#262d36"), (C.ToolTipText, "#dfe5ea")):
        p.setColor(role, QtGui.QColor(col))
    app.setPalette(p)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="vorradar", description="Radar passivo bistático (Doppler) com VOR + RTL-SDR")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sdr", action="store_true", help="inicia já capturando do RTL-SDR")
    g.add_argument("--sim", action="store_true", help="inicia já em simulação leve (32 kS/s)")
    g.add_argument("--sim-full", action="store_true", help="inicia em simulação na taxa total (1,024 MS/s)")
    ap.add_argument("--logdir", help="pasta de logs/IQ desta execução (não é salva no config.json)")
    ap.add_argument("--screenshot", metavar="PNG", help="salva uma captura da janela e fecha (testes/documentação)")
    ap.add_argument("--after", type=float, default=20.0, help="segundos até a captura (padrão 20)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    _dark(app)
    cfg = AppConfig.load()
    saved_logdir = cfg.record.log_dir
    if args.logdir:
        cfg.record.log_dir = args.logdir
    auto = 0 if args.sdr else 1 if args.sim else 2 if args.sim_full else None
    w = MainWindow(cfg, auto)
    w.saved_logdir = saved_logdir
    w.show()
    if args.screenshot:
        def snap():
            w.grab().save(args.screenshot)
            w.close()
            app.quit()
        QtCore.QTimer.singleShot(int(args.after * 1000), snap)
    sys.exit(app.exec())
