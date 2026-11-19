"""Parâmetros configuráveis e constantes físicas do radar passivo VOR.

Todos os blocos de configuração são dataclasses serializáveis em JSON
(config.json na raiz do projeto), para que a UI lembre o último estado.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

C = 299_792_458.0          # velocidade da luz (m/s)
KT = 0.514444              # 1 nó em m/s
FS_WIDE = 32_000.0         # taxa do fluxo "largo" (azimute VOR / gravação IQ)
FS_NB = 4_000.0            # taxa do fluxo de banda estreita (análise Doppler)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config.json"


@dataclass
class RFConfig:
    vor_freq_mhz: float = 115.900   # VOR-DME BRS (Brasília) — ajuste para o seu VOR
    sample_rate: float = 1.024e6    # taxa do RTL-SDR (o LO fica em f_VOR + fs/4)
    ppm: int = 0                    # correção do cristal (TCXO 1 ppm ≈ ±116 Hz em 116 MHz)
    fine_offset_hz: float = 0.0     # deslocamento fino digital: portadora real = nominal + offset
    gain_db: float = 29.7           # ganho manual (sem clipping medido nesta instalação)
    agc: bool = False               # AGC do tuner (desligado: não mascarar ecos fracos)
    bias_tee: bool = False


@dataclass
class DSPConfig:
    nfft: int = 4096                # 4096 @ 4 kHz → 0,98 Hz/bin, janela de 1,02 s
    overlap: float = 0.75
    window: str = "hann"            # "hann" ou "blackmanharris"
    n_integrate: int = 2            # média incoerente de N espectros
    bg_suppress: bool = True        # remove linhas estáticas (fundo de média longa)
    bg_tau_s: float = 45.0
    cfar_mode: str = "OS"           # "OS" (mediana) ou "CA" (média)
    cfar_guard: int = 3
    cfar_ref: int = 16
    threshold_db: float = 9.0       # limiar acima do ruído local
    exclude_hz: float = 3.0         # zona morta em torno da referência (0 Hz)
    exclude_am30: bool = True       # ignora as bandas laterais AM 30 Hz da portadora direta
    max_doppler_hz: float = 400.0   # |Δf| máximo pesquisado
    auto_ref: bool = True           # estima 0 Hz pela linha estática (clutter/direto)
    ref_search_hz: float = 300.0
    ref_tau_s: float = 120.0
    ref_min_prominence_db: float = 5.0


@dataclass
class TrackerConfig:
    gate_hz: float = 6.0
    confirm_hits: int = 6
    confirm_min_s: float = 3.0      # duração mínima para confirmar (quadros são correlacionados)
    max_miss_s: float = 4.0
    min_duration_s: float = 8.0
    v_max_ms: float = 340.0         # velocidade máxima plausível de aeronave
    suppress_am30_ghosts: bool = True


@dataclass
class GeoConfig:
    rx_lat: float = math.nan
    rx_lon: float = math.nan
    vor_lat: float = -15.873700     # BRS (OurAirports)
    vor_lon: float = -48.021400
    radial_cal_deg: float = 0.0     # calibração do módulo experimental de azimute


@dataclass
class RecordConfig:
    log_dir: str = "logs"
    iq_on_detect: bool = True
    prebuffer_s: float = 10.0
    postbuffer_s: float = 5.0
    max_record_s: float = 120.0
    azimuth_enabled: bool = False   # módulo experimental (radial pela fase 30 Hz AM×FM)
    azimuth_int_s: float = 4.0


@dataclass
class DisplayConfig:
    zoom_hz: float = 500.0
    wf_mode: str = "normalizado"    # "normalizado" (fundo removido) ou "bruto"
    wf_min_db: float = -5.0
    wf_max_db: float = 12.0
    colormap: str = "inferno"
    wf_rows: int = 600
    history_s: float = 300.0


@dataclass
class SimConfig:
    lo_error_hz: float = 37.0       # erro de LO simulado (≈0,3 ppm)
    direct_cn0_dbhz: float = 28.0   # resíduo do sinal direto/clutter (dB-Hz)
    echo_cn0_dbhz: float = 30.0     # C/N0 de pico do eco na CPA (dB-Hz)
    spawn_min_s: float = 60.0
    spawn_max_s: float = 120.0
    seed: int = 0                   # 0 = aleatório


@dataclass
class AppConfig:
    rf: RFConfig = field(default_factory=RFConfig)
    dsp: DSPConfig = field(default_factory=DSPConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    geo: GeoConfig = field(default_factory=GeoConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    @property
    def f_carrier_hz(self) -> float:
        return self.rf.vor_freq_mhz * 1e6

    @property
    def wavelength(self) -> float:
        return C / self.f_carrier_hz

    def save(self, path: Path = CONFIG_PATH) -> None:
        d = asdict(self)
        path.write_text(json.dumps(d, indent=2, allow_nan=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cfg
        for name in ("rf", "dsp", "tracker", "geo", "record", "display", "sim"):
            sub = getattr(cfg, name)
            for k, v in d.get(name, {}).items():
                if hasattr(sub, k):
                    setattr(sub, k, type(getattr(sub, k))(v) if v is not None else v)
        return cfg
