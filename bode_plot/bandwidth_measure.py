#!/usr/bin/env python3
"""
bandwidth_measure.py
════════════════════════════════════════════════════════════════
BLDC Current Controller Bandwidth Measurement
  - UDP receiver  : hand controller → desktop (time, i_ref, i_meas)
  - Chirp excitation : 5 ~ 400 Hz logarithmic, 30 s
  - FRF estimation   : Welch cross/auto PSD
  - Plot             : Magnitude Bode | Phase Bode | Step response (IFFT)

Usage
─────
  python bandwidth_measure.py          # live measurement
  python bandwidth_measure.py --demo   # synthetic demo (no hardware)
"""

import argparse
import csv
import json
import os
import socket
import re
import threading
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation
from scipy import signal
from collections import deque
from dataclasses import dataclass, fields
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════
@dataclass
class MeasurementConfig:
    # UDP
    udp_host: str        = "0.0.0.0"
    udp_port: int        = 55150
    udp_buffer_size: int = 1024

    # Signal
    fs: float              = 500.0    # [Hz]  2 ms interval
    f_start: float         = 5.0      # [Hz]
    f_end: float           = 400.0    # [Hz]
    chirp_duration: float  = 30.0     # [s]
    amplitude: float       = 0.3      # [A]   15 % of nominal
    dc_bias: float         = 0.0      # [A]

    # Step test
    step_settle: float     = 0.5      # [s]  pre-step settle
    step_hold: float       = 0.2      # [s]  step hold duration
    step_repeats: int      = 5        # number of step cycles

    # FRF
    nperseg: int   = 2048
    noverlap: int  = 1024
    window: str    = "hann"

    # Safety
    max_current: float     = 1.0      # [A]
    nominal_current: float = 2.0      # [A]

    # Plot
    coh_threshold: float   = 0.80
    ref_f_low: float       = 10.0     # [Hz]  reference-level band
    ref_f_high: float      = 30.0     # [Hz]


CFG = MeasurementConfig()


# ════════════════════════════════════════════════════════════
# Output directory manager
# ════════════════════════════════════════════════════════════
class OutputManager:
    """Create and manage a timestamped output directory.

    Structure
    ─────────
      YYMMDD_HHMM/
      ├── plots/    — .png visualization files
      ├── data/     — .npz raw data files
      └── export/   — .csv, .json exported files
    """

    _SUBDIRS = {
        ".png": "plots",
        ".npz": "data",
        ".csv": "export",
        ".json": "export",
    }

    def __init__(self, base_dir: str = "."):
        stamp = datetime.now().strftime("%y%m%d_%H%M")
        self.root = os.path.join(base_dir, stamp)
        self._created: set[str] = set()

    def _ensure_dir(self, subdir: str) -> None:
        path = os.path.join(self.root, subdir)
        if path not in self._created:
            os.makedirs(path, exist_ok=True)
            self._created.add(path)

    def path(self, filename: str) -> str:
        """Return full path for *filename*, placed in the correct subdirectory."""
        ext = os.path.splitext(filename)[1].lower()
        subdir = self._SUBDIRS.get(ext, "")
        self._ensure_dir(subdir)
        return os.path.join(self.root, subdir, filename)

    def log_structure(self) -> None:
        """Log the output directory tree."""
        logger.info(f"Output directory → {self.root}/")
        for subdir in sorted(set(self._SUBDIRS.values())):
            d = os.path.join(self.root, subdir)
            if os.path.isdir(d):
                files = os.listdir(d)
                if files:
                    logger.info(f"  {subdir}/ ({len(files)} files)")


# ════════════════════════════════════════════════════════════
# Plot style  (dark, monospace — instrument-grade aesthetic)
# ════════════════════════════════════════════════════════════
_BG     = "#0b0f14"
_PANEL  = "#111820"
_GRID   = "#1e2a35"
_SPINE  = "#243040"
_TXT    = "#cdd9e5"
_DIMTXT = "#637a90"
_BLUE   = "#4fa3e0"
_RED    = "#e05c5c"
_GREEN  = "#4ecb82"
_YELLOW = "#e0b94f"
_PURPLE = "#a585e0"
_CYAN   = "#4ec8d4"

def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _BG,
        "axes.facecolor":    _PANEL,
        "axes.edgecolor":    _SPINE,
        "axes.labelcolor":   _TXT,
        "axes.titlecolor":   _TXT,
        "xtick.color":       _DIMTXT,
        "ytick.color":       _DIMTXT,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "text.color":        _TXT,
        "grid.color":        _GRID,
        "grid.linewidth":    0.5,
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    9.5,
        "axes.labelsize":    8.5,
        "axes.titlepad":     7,
        "legend.fontsize":   7.5,
        "legend.framealpha": 0.4,
        "legend.facecolor":  "#0d151f",
        "legend.edgecolor":  _SPINE,
        "lines.antialiased": True,
    })

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_PANEL)
    ax.spines[:].set_color(_SPINE)
    ax.grid(True, which="both", alpha=1.0)
    ax.grid(True, which="minor", alpha=0.4, linewidth=0.3)


# ════════════════════════════════════════════════════════════
# UDP layer
# ════════════════════════════════════════════════════════════
# STM32 sends text-based UDP messages in a sequential protocol:
#   Start      : "bandwidth measure start"
#   Chirp data : "chirp: t=0.002, ref=0.300, cur=0.159"
#   Transition : "chirp done, transition to multisine"
#   MSine data : "MSine: t=0.002, ref=0.300, cur=0.159"
#   Transition : "Multisine done, transition to step"
#   Step data  : "Step: t=0.002, ref=0.300, cur=0.159"
#   End        : "step done, bandwidth measure completed"
_DATA_RE = re.compile(
    r"(?:chirp|MSine|Step):\s*t=([\d.]+),\s*ref=([\-\d.]+),\s*cur=([\-\d.]+)",
    re.IGNORECASE,
)
_PHASE_RE = re.compile(
    r"(chirp|MSine|Step):", re.IGNORECASE,
)

# Protocol message constants
_MSG_START           = "bandwidth measure start"
_MSG_CHIRP_DONE      = "chirp done, transition to multisine"
_MSG_MULTISINE_DONE  = "Multisine done, transition to step"
_MSG_ALL_DONE        = "step done, bandwidth measure completed"

# Legacy protocol (kept for backward compat)
_MSG_LEGACY_START = "Bandwidth Measurement Started"
_MSG_LEGACY_DONE  = "Bandwidth Measurement Done"

@dataclass
class DataPoint:
    t:      float
    i_ref:  float
    i_meas: float

class UDPReceiver:
    """Receives sequential chirp → multisine → step data over UDP.

    Data is stored in per-phase buffers accessible via ``get_phase_data()``.
    The legacy single-signal protocol is also supported for backward compat.
    """

    PHASES = ("chirp", "multisine", "step")

    def __init__(self, cfg: MeasurementConfig):
        self.cfg     = cfg
        # Per-phase data buffers
        max_samples = int(cfg.fs * cfg.chirp_duration * 1.5)
        self._phase_buffers: dict[str, deque] = {
            p: deque(maxlen=max_samples) for p in self.PHASES
        }
        # Legacy single buffer (for single-signal runs)
        self.buffer  = deque(maxlen=max_samples)

        self._stop   = threading.Event()
        self._started = threading.Event()
        self._done    = threading.Event()
        # Per-phase done events
        self._phase_done: dict[str, threading.Event] = {
            p: threading.Event() for p in self.PHASES
        }
        self._current_phase: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()
        self._rx     = 0
        self._drop   = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"UDP receiver  {self.cfg.udp_host}:{self.cfg.udp_port}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_for_start(self, timeout: float = 60.0) -> bool:
        return self._started.wait(timeout=timeout)

    def wait_for_done(self, timeout: float = 120.0) -> bool:
        return self._done.wait(timeout=timeout)

    def wait_for_phase(self, phase: str, timeout: float = 120.0) -> bool:
        """Block until a specific phase completes."""
        ev = self._phase_done.get(phase)
        if ev is None:
            return False
        return ev.wait(timeout=timeout)

    def get_data(self) -> list[DataPoint]:
        """Return all collected data (legacy compat)."""
        with self._lock:
            return list(self.buffer)

    def get_phase_data(self, phase: str) -> list[DataPoint]:
        """Return collected data for a specific phase."""
        with self._lock:
            buf = self._phase_buffers.get(phase, [])
            return list(buf)

    def stats(self) -> dict:
        phase_counts = {}
        with self._lock:
            for p in self.PHASES:
                phase_counts[p] = len(self._phase_buffers[p])
        return {"received": self._rx, "dropped": self._drop,
                "per_phase": phase_counts}

    # ── internal helpers ──────────────────────────────────
    @staticmethod
    def _classify_phase(msg: str) -> Optional[str]:
        """Return signal phase from a data line prefix."""
        m = _PHASE_RE.search(msg)
        if not m:
            return None
        tag = m.group(1).lower()
        if tag in ("chirp",):
            return "chirp"
        if tag in ("msine",):
            return "multisine"
        if tag in ("step",):
            return "step"
        return None

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind((self.cfg.udp_host, self.cfg.udp_port))
            sock.settimeout(0.5)
            collecting = False

            while not self._stop.is_set():
                try:
                    raw, _ = sock.recvfrom(self.cfg.udp_buffer_size)
                    msg = raw.decode("utf-8", errors="replace").strip()
                    msg_lower = msg.lower()

                    # ── start messages ────────────────────────
                    if _MSG_START.lower() in msg_lower or _MSG_LEGACY_START.lower() in msg_lower:
                        collecting = True
                        self._current_phase = "chirp"
                        self._started.set()
                        logger.info(f"Received: {msg}")
                        continue

                    # ── transition: chirp → multisine ─────────
                    if _MSG_CHIRP_DONE.lower() in msg_lower:
                        self._phase_done["chirp"].set()
                        self._current_phase = "multisine"
                        logger.info(f"Received: {msg}")
                        continue

                    # ── transition: multisine → step ──────────
                    if _MSG_MULTISINE_DONE.lower() in msg_lower:
                        self._phase_done["multisine"].set()
                        self._current_phase = "step"
                        logger.info(f"Received: {msg}")
                        continue

                    # ── end messages ──────────────────────────
                    if (_MSG_ALL_DONE.lower() in msg_lower
                            or _MSG_LEGACY_DONE.lower() in msg_lower):
                        self._phase_done["step"].set()
                        collecting = False
                        self._current_phase = None
                        self._done.set()
                        logger.info(f"Received: {msg}")
                        continue

                    if not collecting:
                        continue

                    # ── data line ─────────────────────────────
                    m = _DATA_RE.search(msg)
                    if not m:
                        self._drop += 1
                        continue

                    t_val  = float(m.group(1))
                    i_ref  = float(m.group(2))
                    i_meas = float(m.group(3))

                    if abs(i_meas) > self.cfg.max_current * 3:
                        self._drop += 1
                        continue

                    phase = self._classify_phase(msg) or self._current_phase
                    with self._lock:
                        self.buffer.append(DataPoint(t_val, i_ref, i_meas))
                        if phase and phase in self._phase_buffers:
                            self._phase_buffers[phase].append(
                                DataPoint(t_val, i_ref, i_meas))
                    self._rx += 1
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"UDP error: {e}")
        finally:
            sock.close()


# ════════════════════════════════════════════════════════════
# Chirp generator
# ════════════════════════════════════════════════════════════
class ChirpGenerator:
    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg
        self.t_arr = np.arange(0, cfg.chirp_duration, 1.0 / cfg.fs)
        self.i_ref_arr = self._generate()

    def _generate(self) -> np.ndarray:
        c = self.cfg
        chirp = c.amplitude * signal.chirp(
            self.t_arr, f0=c.f_start, f1=c.f_end,
            t1=c.chirp_duration, method="logarithmic", phi=-90,
        ) + c.dc_bias
        return np.clip(chirp, -c.max_current, c.max_current)

    def get_full_reference(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t_arr.copy(), self.i_ref_arr.copy()


# ════════════════════════════════════════════════════════════
# Multisine generator  (Schroeder-phase, log-spaced freqs)
# ════════════════════════════════════════════════════════════
class MultisineGenerator:
    """
    Sum-of-sines with Schroeder phase optimization.

    x(t) = (A / √N) · Σ cos(2π·f_k·t + φ_k) + dc_bias

    - Frequencies f_k are snapped to DFT bins (k·fs/nperseg) to eliminate
      spectral leakage, then log-approximately spaced in [f_start, f_end]
    - Schroeder phases φ_k = −k(k−1)π/N minimize crest factor
    - Duration matches chirp_duration for fair comparison
    """

    def __init__(self, cfg: MeasurementConfig, n_freqs: int = 60):
        self.cfg     = cfg
        self.n_freqs = n_freqs
        self.freqs   = self._snap_to_dft_bins(cfg, n_freqs)
        self.t_arr   = np.arange(0, cfg.chirp_duration, 1.0 / cfg.fs)
        self.i_ref_arr = self._generate()

    @staticmethod
    def _snap_to_dft_bins(cfg: MeasurementConfig, n_freqs: int) -> np.ndarray:
        """Snap log-spaced target frequencies to nearest DFT bin centres."""
        df          = cfg.fs / cfg.nperseg          # bin resolution
        f_targets   = np.geomspace(cfg.f_start, cfg.f_end, n_freqs)
        bin_indices = np.round(f_targets / df).astype(int)
        bin_indices = np.unique(bin_indices)         # remove duplicates
        freqs       = bin_indices * df
        # Filter to valid range
        freqs = freqs[(freqs >= cfg.f_start) & (freqs <= cfg.f_end)]
        return freqs

    def _generate(self) -> np.ndarray:
        c = self.cfg
        N = len(self.freqs)
        self.n_freqs = N  # update after dedup
        # Schroeder phases for low crest factor
        k      = np.arange(N)
        phases = -k * (k - 1) * np.pi / N

        # Build multisine  (amplitude scaled by 1/√N to match chirp RMS)
        t_col = self.t_arr[:, np.newaxis]       # (samples, 1)
        f_row = self.freqs[np.newaxis, :]       # (1, N)
        p_row = phases[np.newaxis, :]           # (1, N)

        x = (c.amplitude / np.sqrt(N)) * np.sum(
            np.cos(2 * np.pi * f_row * t_col + p_row), axis=1
        ) + c.dc_bias

        return np.clip(x, -c.max_current, c.max_current)

    def get_full_reference(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t_arr.copy(), self.i_ref_arr.copy()


# ════════════════════════════════════════════════════════════
# Step generator  (0 → amplitude step, repeated N times)
# ════════════════════════════════════════════════════════════
class StepGenerator:
    """
    Generates repeated step signals for direct time-domain analysis.

    Each cycle:
      [0, settle_time)   → dc_bias    (baseline)
      [settle_time, end) → amplitude  (step)

    Repeated step_repeats times → ensemble average for noise reduction.
    """

    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg
        cycle_len = cfg.step_settle + cfg.step_hold
        total_dur = cycle_len * cfg.step_repeats
        self.t_arr = np.arange(0, total_dur, 1.0 / cfg.fs)
        self.i_ref_arr = self._generate()

    def _generate(self) -> np.ndarray:
        c = self.cfg
        cycle_samples  = int((c.step_settle + c.step_hold) * c.fs)
        settle_samples = int(c.step_settle * c.fs)

        x = np.full(len(self.t_arr), c.dc_bias)
        for i in range(c.step_repeats):
            start = i * cycle_samples + settle_samples
            end   = (i + 1) * cycle_samples
            if end <= len(x):
                x[start:end] = c.amplitude
            else:
                x[start:len(x)] = c.amplitude

        return np.clip(x, -c.max_current, c.max_current)

    def get_full_reference(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t_arr.copy(), self.i_ref_arr.copy()


# ════════════════════════════════════════════════════════════
# FRF estimator
# ════════════════════════════════════════════════════════════
class FRFEstimator:
    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg

    def estimate(
        self,
        t: np.ndarray,
        i_ref: np.ndarray,
        i_meas: np.ndarray,
        noise_report: Optional["NoiseReport"] = None,
    ) -> dict:
        fs      = self.cfg.fs

        # Use adaptive nperseg if NoiseReport recommends it
        base_nperseg = (noise_report.recommended_nperseg
                        if noise_report else self.cfg.nperseg)
        nperseg = min(base_nperseg, len(i_ref) // 4)
        noverlap= min(self.cfg.noverlap, nperseg // 2)
        kw      = dict(fs=fs, window=self.cfg.window,
                       nperseg=nperseg, noverlap=noverlap)
        eps = 1e-12

        f,  Sxx = signal.welch(i_ref,         **kw)
        _,  Sxy = signal.csd(i_ref, i_meas,   **kw)
        _,  Syy = signal.welch(i_meas,        **kw)
        _,  Syx = signal.csd(i_meas, i_ref,   **kw)

        # H1 estimator (robust to output noise)
        H1         = Sxy / (Sxx + eps)
        # H2 estimator (robust to input noise)
        H2         = Syy / (Syx + eps)

        # Select estimator based on noise analysis
        use_hv = (noise_report is not None
                  and noise_report.recommended_estimator == "Hv")
        if use_hv:
            # Hv = geometric mean of H1 and H2 (robust to noise on both)
            H = np.sqrt(H1 * H2)
            logger.info(f"  FRF: using Hv estimator  (nperseg={nperseg})")
        else:
            H = H1
            if noise_report:
                logger.info(f"  FRF: using H1 estimator  (nperseg={nperseg})")

        mag_db     = 20 * np.log10(np.abs(H) + eps)
        phase_deg  = np.degrees(np.unwrap(np.angle(H)))
        coherence  = np.clip(np.abs(Sxy)**2 / (Sxx * Syy + eps), 0.0, 1.0)
        valid      = coherence > self.cfg.coh_threshold

        bw_hz      = self._bandwidth(f, mag_db, valid)

        # Gain margin: gain at phase = -180° crossover
        gain_margin, f_gm = self._gain_margin(f, mag_db, phase_deg, valid)

        return dict(
            f=f, H=H, H1=H1, H2=H2, mag_db=mag_db,
            phase_deg=phase_deg, coherence=coherence,
            valid=valid, bandwidth_hz=bw_hz,
            gain_margin_db=gain_margin, f_gain_margin=f_gm,
            estimator="Hv" if use_hv else "H1",
            nperseg_used=nperseg,
        )

    def _bandwidth(
        self,
        f: np.ndarray,
        mag_db: np.ndarray,
        valid: np.ndarray,
    ) -> float:
        cfg  = self.cfg
        rmask = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
        ref_db = float(np.mean(mag_db[rmask])) if np.any(rmask) else 0.0

        drop = valid & (mag_db < ref_db - 3.0) & (f > cfg.ref_f_high)
        if not np.any(drop):
            return float(f[valid][-1]) if np.any(valid) else float(f[-1])

        idx = int(np.argmax(drop))
        if idx > 0:
            f0, f1 = float(f[idx-1]), float(f[idx])
            m0, m1 = float(mag_db[idx-1]), float(mag_db[idx])
            dm = m1 - m0
            if abs(dm) < 1e-6:
                return f0
            return f0 + (f1-f0) * (ref_db-3.0-m0) / dm
        return float(f[idx])

    def _gain_margin(
        self,
        f: np.ndarray,
        mag_db: np.ndarray,
        phase_deg: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[Optional[float], Optional[float]]:
        """Find gain margin at -180° phase crossover."""
        cfg = self.cfg
        fmsk = valid & (f >= cfg.f_start) & (f <= cfg.f_end)
        for i in range(len(phase_deg) - 1):
            if not fmsk[i]:
                continue
            if phase_deg[i] >= -180.0 >= phase_deg[i+1]:
                # Interpolate frequency at -180°
                dp = phase_deg[i+1] - phase_deg[i]
                if abs(dp) < 1e-6:
                    f_cross = float(f[i])
                else:
                    frac = (-180.0 - phase_deg[i]) / dp
                    f_cross = float(f[i] + (f[i+1] - f[i]) * frac)
                mag_at_cross = float(
                    mag_db[i] + (mag_db[i+1] - mag_db[i]) * frac
                    if abs(dp) >= 1e-6 else mag_db[i]
                )
                gain_margin = -mag_at_cross  # GM = -|H| at -180°
                return gain_margin, f_cross
        return None, None


# ════════════════════════════════════════════════════════════
# Noise analysis
# ════════════════════════════════════════════════════════════
@dataclass
class NoiseReport:
    """Results of automatic noise characterisation."""
    snr_db: float                       # broadband SNR estimate
    noise_rms: float                    # RMS of residual  (i_meas − trend)
    signal_rms: float                   # RMS of signal content
    spike_count: int                    # samples exceeding 4σ
    spike_ratio: float                  # spike_count / total
    tonal_peaks: list[float]            # frequencies of detected tonal peaks [Hz]
    tonal_powers_db: list[float]        # power of each tonal peak [dB]
    mean_coherence: float               # mean γ² in [f_start, f_end]
    recommended_filters: list[str]      # list of filter names to apply
    recommended_nperseg: int            # Welch segment length
    recommended_estimator: str          # "H1" or "Hv"


class NoiseAnalyzer:
    """
    Analyse raw i_meas to characterise noise and recommend preprocessing.

    Pipeline
    ────────
    1. Compute error signal  e(t) = i_meas − i_ref  (noise + tracking error)
    2. PSD of e(t) → detect tonal peaks (PWM harmonics, rotor ripple)
    3. Statistics  → spike count, RMS, SNR
    4. Quick coherence scan  → mean γ² in measurement band
    5. Decision rules → recommend notch, median, nperseg, estimator
    """

    # Tonal peak must be this many dB above local median to be flagged
    _PEAK_PROMINENCE_DB = 10.0
    # Spike threshold in multiples of σ
    _SPIKE_SIGMA = 4.0
    # If mean coherence drops below this, switch to Hv estimator
    _COH_HV_THRESHOLD = 0.85
    # If spike ratio exceeds this, enable median filter
    _SPIKE_RATIO_THRESHOLD = 0.005
    # SNR below this triggers nperseg reduction (more averaging)
    _LOW_SNR_DB = 15.0

    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg

    def analyze(
        self,
        t: np.ndarray,
        i_ref: np.ndarray,
        i_meas: np.ndarray,
    ) -> NoiseReport:
        cfg = self.cfg
        fs  = cfg.fs

        # ── 1. Error signal ──────────────────────────────────
        error = i_meas - i_ref

        # ── 2. Signal / noise RMS ────────────────────────────
        signal_rms = float(np.sqrt(np.mean(i_ref ** 2)))
        noise_rms  = float(np.sqrt(np.mean(error ** 2)))
        eps = 1e-12
        snr_db = float(20.0 * np.log10(signal_rms / (noise_rms + eps)))

        # ── 3. Spike detection ───────────────────────────────
        sigma  = float(np.std(error))
        spikes = np.abs(error) > self._SPIKE_SIGMA * sigma
        spike_count = int(np.sum(spikes))
        spike_ratio = spike_count / max(len(error), 1)

        # ── 4. PSD of error → tonal peak detection ──────────
        nperseg_psd = min(2048, len(error) // 4)
        f_psd, Pee = signal.welch(error, fs=fs, nperseg=nperseg_psd)

        Pee_db = 10.0 * np.log10(Pee + eps)

        # Local median in sliding window for adaptive threshold
        win = max(15, nperseg_psd // 32)
        if win % 2 == 0:
            win += 1
        from scipy.ndimage import median_filter as _medfilt1d
        Pee_median = _medfilt1d(Pee_db, size=win)

        prominence = Pee_db - Pee_median
        peak_idx, _ = signal.find_peaks(
            prominence,
            height=self._PEAK_PROMINENCE_DB,
            distance=max(3, int(5.0 / (fs / nperseg_psd))),  # ≥ 5 Hz apart
        )

        # Filter: only peaks outside reference band & within Nyquist
        # Avoid notching frequencies in the reference band (ref_f_low–ref_f_high)
        # as that would distort the FRF reference level
        valid_peaks = [
            i for i in peak_idx
            if f_psd[i] > cfg.ref_f_high and f_psd[i] < fs / 2.0
        ]
        tonal_peaks     = [float(f_psd[i]) for i in valid_peaks]
        tonal_powers_db = [float(Pee_db[i]) for i in valid_peaks]

        # ── 5. Quick coherence estimate ──────────────────────
        kw = dict(fs=fs, nperseg=nperseg_psd, noverlap=nperseg_psd // 2,
                  window="hann")
        f_c, Sxx = signal.welch(i_ref, **kw)
        _,   Sxy = signal.csd(i_ref, i_meas, **kw)
        _,   Syy = signal.welch(i_meas, **kw)
        coh = np.clip(np.abs(Sxy)**2 / (Sxx * Syy + eps), 0.0, 1.0)

        band = (f_c >= cfg.f_start) & (f_c <= cfg.f_end)
        mean_coh = float(np.mean(coh[band])) if np.any(band) else 0.0

        # ── 6. Decision rules ────────────────────────────────
        filters = []

        # Spikes → median filter (always first — outliers bias all estimates)
        if spike_ratio > self._SPIKE_RATIO_THRESHOLD:
            filters.append("median")

        # Tonal peaks → notch filter ONLY when coherence is already high
        # (meaning the tones leak into the CSD and bias H1).
        # For broadband excitation (chirp/multisine), Welch's CSD
        # naturally rejects uncorrelated tonal noise.
        if tonal_peaks and mean_coh > 0.90:
            filters.append("notch")

        # Low SNR → LPF at f_end to remove out-of-band noise
        if snr_db < self._LOW_SNR_DB:
            filters.append("lowpass")

        # nperseg: reduce for more averaging only when very noisy
        if snr_db < 10.0:
            rec_nperseg = max(512, cfg.nperseg // 2)
        else:
            rec_nperseg = cfg.nperseg

        # Estimator: Hv only when coherence is very poor
        rec_estimator = "Hv" if mean_coh < 0.70 else "H1"

        report = NoiseReport(
            snr_db=snr_db,
            noise_rms=noise_rms,
            signal_rms=signal_rms,
            spike_count=spike_count,
            spike_ratio=spike_ratio,
            tonal_peaks=tonal_peaks,
            tonal_powers_db=tonal_powers_db,
            mean_coherence=mean_coh,
            recommended_filters=filters,
            recommended_nperseg=rec_nperseg,
            recommended_estimator=rec_estimator,
        )

        self._log_report(report)
        return report

    @staticmethod
    def _log_report(r: NoiseReport) -> None:
        logger.info("── Noise analysis ─────────────────────────────")
        logger.info(f"  SNR          : {r.snr_db:.1f} dB")
        logger.info(f"  Signal RMS   : {r.signal_rms*1e3:.2f} mA")
        logger.info(f"  Noise  RMS   : {r.noise_rms*1e3:.2f} mA")
        logger.info(f"  Spikes       : {r.spike_count}  "
                     f"({r.spike_ratio*100:.2f} %)")
        if r.tonal_peaks:
            peaks_str = ", ".join(f"{f:.1f} Hz" for f in r.tonal_peaks[:5])
            if len(r.tonal_peaks) > 5:
                peaks_str += f" (+{len(r.tonal_peaks)-5} more)"
            logger.info(f"  Tonal peaks  : {peaks_str}")
        else:
            logger.info("  Tonal peaks  : none")
        logger.info(f"  Mean γ²      : {r.mean_coherence:.3f}")
        logger.info(f"  Filters      : {r.recommended_filters or 'none'}")
        logger.info(f"  nperseg      : {r.recommended_nperseg}")
        logger.info(f"  Estimator    : {r.recommended_estimator}")
        logger.info("────────────────────────────────────────────────")


# ════════════════════════════════════════════════════════════
# Adaptive preprocessor
# ════════════════════════════════════════════════════════════
class AdaptivePreprocessor:
    """
    Apply filter chain selected by NoiseAnalyzer.

    Available filters (applied in order):
      1. median   — kernel=5 removes impulse spikes
      2. notch    — 2nd-order IIR notch at each detected tonal peak
      3. lowpass  — Butterworth LPF at f_end × 1.2  (remove OOB noise)
    """

    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg

    def apply(
        self,
        i_meas: np.ndarray,
        report: NoiseReport,
        i_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return filtered copy of i_meas.  i_ref is never filtered."""
        x = i_meas.copy()
        applied = []

        # ── 1. Spike removal (interpolate outliers) ──────────
        if "median" in report.recommended_filters:
            # Use error signal for spike detection when i_ref available
            if i_ref is not None:
                residual = x - i_ref
            else:
                residual = x - np.median(x)
            sigma = float(np.std(residual))
            spike_mask = np.abs(residual) > 4.0 * sigma
            n_fixed = int(np.sum(spike_mask))
            if n_fixed > 0:
                for idx in np.where(spike_mask)[0]:
                    lo = max(0, idx - 2)
                    hi = min(len(x), idx + 3)
                    neighbors = [
                        i for i in range(lo, hi)
                        if i != idx and not spike_mask[i]
                    ]
                    if neighbors:
                        x[idx] = np.mean(x[neighbors])
            applied.append(f"despike(interpolated {n_fixed} outliers)")

        # ── 2. Notch filters (tonal peaks) ───────────────────
        if "notch" in report.recommended_filters and report.tonal_peaks:
            fs = self.cfg.fs
            for f_notch in report.tonal_peaks:
                if f_notch <= 0 or f_notch >= fs / 2.0:
                    continue
                Q = max(30.0, f_notch / 3.0)  # narrow notch, Q scales with freq
                b, a = signal.iirnotch(f_notch, Q, fs)
                x = signal.filtfilt(b, a, x)
                applied.append(f"notch({f_notch:.1f} Hz, Q={Q:.0f})")

        # ── 3. Lowpass filter (OOB noise removal) ────────────
        if "lowpass" in report.recommended_filters:
            fs   = self.cfg.fs
            f_lp = min(self.cfg.f_end * 1.2, fs / 2.0 - 1.0)
            sos  = signal.butter(4, f_lp, btype="low", fs=fs, output="sos")
            x    = signal.sosfiltfilt(sos, x)
            applied.append(f"lowpass({f_lp:.0f} Hz, 4th-order Butterworth)")

        if applied:
            logger.info("  Applied filters: " + " → ".join(applied))
        else:
            logger.info("  No filtering needed (clean signal)")

        return x


# ════════════════════════════════════════════════════════════
# Noise analysis plot
# ════════════════════════════════════════════════════════════
def plot_noise_analysis(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    i_meas_filtered: np.ndarray,
    report: NoiseReport,
    cfg: MeasurementConfig,
    save_path: str = "noise_analysis.png",
) -> None:
    """
    Three-panel noise analysis figure
    ──────────────────────────────────
    Top    : Error PSD (raw vs filtered) with tonal peak markers
    Mid    : Time-domain overlay (raw / filtered / reference)
    Bottom : Coherence comparison (raw vs filtered)
    """
    _apply_style()
    fs = cfg.fs

    fig = plt.figure(figsize=(14, 10), dpi=140)
    fig.patch.set_facecolor(_BG)

    fig.suptitle(
        f"Noise Analysis   ·   SNR = {report.snr_db:.1f} dB   |   "
        f"Filters: {', '.join(report.recommended_filters) or 'none'}   |   "
        f"Estimator: {report.recommended_estimator}",
        color=_TXT, fontsize=10.5, fontweight="bold", y=0.985,
    )

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 0.8, 0.8],
        hspace=0.45,
        left=0.07, right=0.97, top=0.94, bottom=0.06,
    )

    # ── [0] Error PSD ─────────────────────────────────────────
    ax_psd = fig.add_subplot(gs[0])
    _style_ax(ax_psd)

    error_raw = i_meas - i_ref
    error_filt = i_meas_filtered - i_ref
    npsd = min(2048, len(error_raw) // 4)

    f_p, P_raw  = signal.welch(error_raw,  fs=fs, nperseg=npsd)
    _,   P_filt = signal.welch(error_filt, fs=fs, nperseg=npsd)
    eps = 1e-12

    ax_psd.semilogy(f_p, P_raw,  color=_RED,  lw=1.2, alpha=0.7, label="raw error PSD")
    ax_psd.semilogy(f_p, P_filt, color=_GREEN, lw=1.5, zorder=3, label="filtered error PSD")

    # Mark tonal peaks
    for fp in report.tonal_peaks:
        ax_psd.axvline(fp, color=_YELLOW, lw=0.8, ls=":", alpha=0.7)
    if report.tonal_peaks:
        ax_psd.scatter(
            report.tonal_peaks,
            [P_raw[np.argmin(np.abs(f_p - fp))] for fp in report.tonal_peaks],
            color=_YELLOW, s=25, zorder=5, label="tonal peaks",
        )

    ax_psd.set_xlim([0, fs / 2])
    ax_psd.set_xlabel("Frequency [Hz]")
    ax_psd.set_ylabel("PSD [A²/Hz]")
    ax_psd.set_title("Error Power Spectral Density   (i_meas − i_ref)")
    ax_psd.legend(loc="upper right", handlelength=1.6)

    # ── [1] Time-domain ────────────────────────────────────────
    ax_time = fig.add_subplot(gs[1])
    _style_ax(ax_time)

    # Show only first 500ms for clarity
    n_show = min(int(0.5 * fs), len(t))
    t_ms = t[:n_show] * 1e3

    ax_time.plot(t_ms, i_ref[:n_show], color=_BLUE, lw=1.0, alpha=0.6,
                 label="i_ref")
    ax_time.plot(t_ms, i_meas[:n_show], color=_RED, lw=0.8, alpha=0.5,
                 label="i_meas (raw)")
    ax_time.plot(t_ms, i_meas_filtered[:n_show], color=_GREEN, lw=1.2,
                 zorder=3, label="i_meas (filtered)")

    ax_time.set_xlabel("Time [ms]")
    ax_time.set_ylabel("Current [A]")
    ax_time.set_title("Time Domain   (first 500 ms)")
    ax_time.legend(loc="upper right", handlelength=1.6, ncol=3)

    # ── [2] Coherence comparison ───────────────────────────────
    ax_coh = fig.add_subplot(gs[2])
    _style_ax(ax_coh)

    kw_coh = dict(fs=fs, nperseg=min(cfg.nperseg, len(i_ref) // 4),
                  window=cfg.window)
    kw_coh["noverlap"] = kw_coh["nperseg"] // 2

    f_c, Sxx   = signal.welch(i_ref, **kw_coh)
    _,   Sxy_r = signal.csd(i_ref, i_meas, **kw_coh)
    _,   Syy_r = signal.welch(i_meas, **kw_coh)
    _,   Sxy_f = signal.csd(i_ref, i_meas_filtered, **kw_coh)
    _,   Syy_f = signal.welch(i_meas_filtered, **kw_coh)

    coh_raw  = np.clip(np.abs(Sxy_r)**2 / (Sxx * Syy_r + eps), 0.0, 1.0)
    coh_filt = np.clip(np.abs(Sxy_f)**2 / (Sxx * Syy_f + eps), 0.0, 1.0)

    fmsk = (f_c >= cfg.f_start) & (f_c <= cfg.f_end)
    ax_coh.semilogx(f_c[fmsk], coh_raw[fmsk],  color=_RED,   lw=1.0,
                    alpha=0.6, label="raw")
    ax_coh.semilogx(f_c[fmsk], coh_filt[fmsk], color=_GREEN, lw=1.5,
                    zorder=3, label="filtered")
    ax_coh.axhline(cfg.coh_threshold, color=_YELLOW, lw=1.0, ls="--",
                   label=f"threshold γ² = {cfg.coh_threshold}")

    # Mean coherence annotations
    mean_raw  = float(np.mean(coh_raw[fmsk]))
    mean_filt = float(np.mean(coh_filt[fmsk]))
    ax_coh.text(
        0.98, 0.05,
        f"mean γ²:  raw={mean_raw:.3f}   filtered={mean_filt:.3f}   "
        f"Δ={mean_filt-mean_raw:+.3f}",
        transform=ax_coh.transAxes, color=_TXT, fontsize=7.5,
        ha="right", va="bottom", fontfamily="monospace",
    )

    ax_coh.set_xlim([cfg.f_start, cfg.f_end])
    ax_coh.set_ylim([-0.05, 1.08])
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylabel("Coherence γ²")
    ax_coh.set_title("Coherence Improvement")
    ax_coh.legend(loc="lower left", handlelength=1.6)
    ax_coh.xaxis.set_minor_formatter(ticker.NullFormatter())

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    logger.info(f"Noise analysis plot saved → {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
# Preprocessing
# ════════════════════════════════════════════════════════════
def preprocess(
    data: list[DataPoint],
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Returns (t_uniform, i_ref, i_meas, fs_detected)."""
    if len(data) < 100:
        raise ValueError(f"Too few data points: {len(data)}")

    t_r     = np.array([d.t     for d in data])
    ref_r   = np.array([d.i_ref  for d in data])
    meas_r  = np.array([d.i_meas for d in data])

    idx     = np.argsort(t_r)
    t_r, ref_r, meas_r = t_r[idx], ref_r[idx], meas_r[idx]

    _, uniq = np.unique(t_r, return_index=True)
    t_r, ref_r, meas_r = t_r[uniq], ref_r[uniq], meas_r[uniq]

    # ── Auto-detect sampling frequency from timestamps ──
    dt_arr = np.diff(t_r)
    dt_arr = dt_arr[dt_arr > 0]  # discard zero-diff
    if len(dt_arr) > 0:
        dt_median = float(np.median(dt_arr))
        fs_detected = round(1.0 / dt_median, 1)
        if abs(fs_detected - fs) / max(fs, 1.0) > 0.05:
            logger.info(
                f"Auto-detected fs = {fs_detected:.1f} Hz "
                f"(dt_median = {dt_median*1e3:.3f} ms), "
                f"overriding configured fs = {fs:.1f} Hz"
            )
        else:
            logger.info(
                f"Detected fs = {fs_detected:.1f} Hz "
                f"(matches configured {fs:.1f} Hz)"
            )
        fs = fs_detected
    else:
        logger.warning("Cannot detect fs from timestamps, using configured value")

    t_u   = np.arange(t_r[0], t_r[-1], 1.0/fs)
    i_ref  = np.interp(t_u, t_r, ref_r)
    i_meas = np.interp(t_u, t_r, meas_r)

    expected = int((t_r[-1]-t_r[0]) * fs)
    loss_pct = 100.0 * (1 - len(t_r)/max(expected,1))
    logger.info(f"Packet loss {loss_pct:.1f} % ({expected-len(t_r)}/{expected})")

    return t_u, i_ref, i_meas, fs


# ════════════════════════════════════════════════════════════
# Step response  (IFFT-based estimation from FRF)
# ════════════════════════════════════════════════════════════
def _estimate_step_response(
    f_valid: np.ndarray,
    H_valid: np.ndarray,
    fs: float,
    n_pts: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    H(jω) on valid band  →  IFFT  →  impulse h(t)  →  ∫h dτ = step s(t)

    Steps
    ─────
    1. Interpolate H onto uniform grid [0, fs/2]
    2. Hermitian extension → irfft → h(t)
    3. cumsum * dt → s(t),  normalize to unit steady-state
    """
    f_uni  = np.linspace(0, fs / 2.0, n_pts // 2 + 1)
    H_uni  = np.zeros(len(f_uni), dtype=complex)

    if len(f_valid) == 0:
        t_step = np.arange(n_pts) * (1.0 / fs)
        crop   = min(int(0.025 * fs) + 1, n_pts)
        return t_step[:crop], np.zeros(crop)

    f0, f1 = float(f_valid[0]), float(f_valid[-1])
    band   = (f_uni >= f0) & (f_uni <= f1)

    H_uni[band] = (
        np.interp(f_uni[band], f_valid, H_valid.real)
        + 1j * np.interp(f_uni[band], f_valid, H_valid.imag)
    )
    # DC: unity gain (current controller steady-state: i_meas = i_ref)
    H_uni[0] = 1.0 + 0j

    # Smooth edges to reduce Gibbs ringing
    ramp_len = max(1, int(0.05 * np.sum(band)))
    band_idx = np.where(band)[0]
    if len(band_idx) > ramp_len * 2:
        ramp   = np.linspace(0, 1, ramp_len)
        H_uni[band_idx[:ramp_len]]  *= ramp
        H_uni[band_idx[-ramp_len:]] *= ramp[::-1]

    h      = np.fft.irfft(H_uni, n=n_pts)
    dt     = 1.0 / fs
    s      = np.cumsum(h) * dt

    ss     = float(np.mean(s[int(0.80*len(s)):]))
    s_norm = s / ss if abs(ss) > 1e-9 else s

    t_step = np.arange(n_pts) * dt
    # Crop to first 25 ms  (well beyond any BLDC current-loop settling)
    crop   = min(int(0.025 * fs) + 1, n_pts)
    return t_step[:crop], s_norm[:crop]


def _step_metrics(t: np.ndarray, s: np.ndarray) -> dict:
    m = dict(t_rise=None, overshoot_pct=None, t_settle=None)
    if len(s) < 10:
        return m

    ss  = float(np.mean(s[int(0.80*len(s)):]))
    pk  = float(s.max())

    lo_val, hi_val = 0.10 * ss, 0.90 * ss
    i_lo = int(np.argmax(s >= lo_val)) if np.any(s >= lo_val) else None
    i_hi = int(np.argmax(s >= hi_val)) if np.any(s >= hi_val) else None
    if i_lo is not None and i_hi is not None and i_hi > i_lo:
        m["t_rise"] = float(t[i_hi] - t[i_lo])

    if ss > 1e-9:
        m["overshoot_pct"] = max(0.0, (pk/ss - 1.0) * 100.0)

    tol     = 0.02 * abs(ss)
    outside = np.where(np.abs(s - ss) > tol)[0]
    if len(outside):
        m["t_settle"] = float(t[outside[-1]])

    return m


# ════════════════════════════════════════════════════════════
# Step response analysis  (direct time-domain measurement)
# ════════════════════════════════════════════════════════════
def _analyze_step_response(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    cfg: MeasurementConfig,
) -> dict:
    """
    Extract individual step responses from repeated step signal,
    ensemble-average them, and compute metrics.

    Returns
    ───────
    dict with keys:
        t_step     — time array for one step cycle [s]
        responses  — list of individual normalized responses
        avg        — ensemble-averaged normalized response
        metrics    — dict with t_rise, overshoot_pct, t_settle, ss_error
    """
    cycle_samples  = int((cfg.step_settle + cfg.step_hold) * cfg.fs)
    settle_samples = int(cfg.step_settle * cfg.fs)
    hold_samples   = int(cfg.step_hold * cfg.fs)

    # Extract individual step responses (from step edge onward)
    responses = []
    for i in range(cfg.step_repeats):
        start = i * cycle_samples + settle_samples
        end   = start + hold_samples
        if end > len(i_meas):
            break

        # Baseline: mean of last 20% of settle phase
        base_start = i * cycle_samples + int(settle_samples * 0.8)
        base_end   = i * cycle_samples + settle_samples
        baseline   = float(np.mean(i_meas[base_start:base_end]))

        resp = i_meas[start:end]
        target = cfg.amplitude - baseline

        # Normalize: 0 = baseline, 1 = target
        if abs(target) > 1e-9:
            resp_norm = (resp - baseline) / target
        else:
            resp_norm = resp - baseline

        responses.append(resp_norm)

    if not responses:
        raise ValueError("No valid step responses found")

    # Ensemble average
    min_len = min(len(r) for r in responses)
    responses = [r[:min_len] for r in responses]
    avg = np.mean(responses, axis=0)

    t_step = np.arange(min_len) / cfg.fs

    # Metrics from averaged response
    metrics = _step_metrics(t_step, avg)

    # Steady-state error (last 20%)
    ss_val = float(np.mean(avg[int(0.8 * len(avg)):]))
    metrics["ss_error_pct"] = abs(1.0 - ss_val) * 100.0

    # ── FRF estimation from step data ──────────────────────
    # The repeated step signal contains broadband energy; use
    # Welch cross-spectral estimation to extract H(jω) and BW.
    est = FRFEstimator(cfg)
    frf = est.estimate(t, i_ref, i_meas)

    return dict(
        t_step=t_step,
        responses=responses,
        avg=avg,
        metrics=metrics,
        frf=frf,
    )


def plot_step_results(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    step_data: dict,
    cfg: MeasurementConfig,
    save_path: str = "step_response_result.png",
) -> None:
    """
    Four-panel step response figure
    ────────────────────────────────
    [0,0] Bode Magnitude     [0,1] Bode Phase
    [1,:] Raw time-domain data (i_ref + i_meas)
    [2,:] Ensemble-averaged normalized step response with metrics
    """
    _apply_style()

    t_step   = step_data["t_step"]
    avg      = step_data["avg"]
    resps    = step_data["responses"]
    metrics  = step_data["metrics"]
    frf      = step_data.get("frf")
    t_ms     = t_step * 1e3

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=140)
    fig.patch.set_facecolor(_BG)

    # Suptitle
    parts = []
    if frf is not None:
        parts.append(f"BW = {frf['bandwidth_hz']:.1f} Hz")
    if metrics["t_rise"] is not None:
        parts.append(f"t_rise = {metrics['t_rise']*1e3:.2f} ms")
    if metrics["overshoot_pct"] is not None:
        parts.append(f"OS = {metrics['overshoot_pct']:.1f} %")
    if metrics["t_settle"] is not None:
        parts.append(f"t_settle = {metrics['t_settle']*1e3:.2f} ms")
    if metrics.get("ss_error_pct") is not None:
        parts.append(f"SS err = {metrics['ss_error_pct']:.2f} %")

    fig.suptitle(
        "BLDC Current Controller   ·   Step Response + Bode   ·   "
        + "   |   ".join(parts),
        color=_TXT, fontsize=10.5, fontweight="bold",
        y=0.985, x=0.5,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 0.6, 1.2],
        hspace=0.45, wspace=0.28,
        left=0.07, right=0.97, top=0.96, bottom=0.04,
    )

    # ════════════════════════════════════════════════════════
    # [0,0]  Bode — Magnitude
    # ════════════════════════════════════════════════════════
    ax_mag = fig.add_subplot(gs[0, 0])
    _style_ax(ax_mag)

    if frf is not None:
        f     = frf["f"]
        mag   = frf["mag_db"]
        coh   = frf["coherence"]
        valid = frf["valid"]
        bw    = frf["bandwidth_hz"]

        ax_mag.semilogx(f, mag, color=_DIMTXT, lw=0.7, alpha=0.45,
                        label="all")
        ax_mag.semilogx(f[valid], mag[valid], color=_GREEN, lw=1.5,
                        label=f"γ² > {cfg.coh_threshold:.2f}")

        # -3 dB line
        rmask  = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
        ref_db = float(np.mean(mag[rmask])) if np.any(rmask) else 0.0
        ax_mag.axhline(ref_db - 3.0, color=_YELLOW, lw=1.0, ls="--",
                       alpha=0.7, label=f"−3 dB ({ref_db-3:.1f} dB)")

        # BW marker
        ax_mag.axvline(bw, color=_CYAN, lw=1.0, ls=":", alpha=0.8)
        ax_mag.scatter([bw], [ref_db - 3.0], color=_CYAN, s=50, zorder=5)
        ax_mag.text(
            bw, ref_db - 3.0 - 2.0,
            f"BW = {bw:.1f} Hz", color=_CYAN,
            fontsize=9, fontweight="bold", ha="center", va="top",
            zorder=5,
        )

    ax_mag.set_xlim([cfg.f_start, cfg.f_end])
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Bode  —  Magnitude  (from step)")
    ax_mag.legend(loc="lower left", handlelength=1.6)
    ax_mag.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [0,1]  Bode — Phase
    # ════════════════════════════════════════════════════════
    ax_ph = fig.add_subplot(gs[0, 1])
    _style_ax(ax_ph)

    if frf is not None:
        phase = frf["phase_deg"]

        ax_ph.semilogx(f, phase, color=_DIMTXT, lw=0.7, alpha=0.45,
                       label="all")
        ax_ph.semilogx(f[valid], phase[valid], color=_PURPLE, lw=1.5,
                       label=f"γ² > {cfg.coh_threshold:.2f}")

        # Phase margin at BW
        if bw > 0:
            ph_at_bw = float(np.interp(bw, f[valid], phase[valid]))
            pm = 180.0 + ph_at_bw
            ax_ph.axvline(bw, color=_CYAN, lw=1.0, ls=":", alpha=0.8)
            ax_ph.scatter([bw], [ph_at_bw], color=_CYAN, s=50, zorder=5)
            ax_ph.text(
                bw, ph_at_bw - 8,
                f"PM = {pm:.1f}°", color=_CYAN,
                fontsize=9, fontweight="bold", ha="center", va="top",
                zorder=5,
            )

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase  (from step)")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [1,:]  Raw time-domain data
    # ════════════════════════════════════════════════════════
    ax_raw = fig.add_subplot(gs[1, :])
    _style_ax(ax_raw)

    t_plot = t * 1e3  # ms
    ax_raw.plot(t_plot, i_ref,  color=_BLUE, lw=1.2, alpha=0.8,
                label="i_ref (target)")
    ax_raw.plot(t_plot, i_meas, color=_GREEN, lw=1.0, alpha=0.7,
                label="i_meas (measured)")

    ax_raw.set_xlabel("Time [ms]")
    ax_raw.set_ylabel("Current [A]")
    ax_raw.set_title(
        f"Raw Data   ({cfg.step_repeats} cycles × "
        f"{cfg.step_settle}s settle + {cfg.step_hold}s hold)"
    )
    ax_raw.legend(loc="upper right", handlelength=1.6)

    # ════════════════════════════════════════════════════════
    # [2,:]  Ensemble-averaged step response
    # ════════════════════════════════════════════════════════
    ax_step = fig.add_subplot(gs[2, :])
    _style_ax(ax_step)

    # Individual responses (dim)
    for i, r in enumerate(resps):
        t_r = np.arange(len(r)) / cfg.fs * 1e3
        ax_step.plot(t_r, r, color=_DIMTXT, lw=0.6, alpha=0.4,
                     label="individual" if i == 0 else None)

    # Averaged response
    ax_step.plot(t_ms, avg, color=_GREEN, lw=2.2, zorder=4,
                 label=f"ensemble avg (N={len(resps)})")

    # Guides
    ax_step.axhline(1.00, color=_DIMTXT, lw=0.8, ls=":", zorder=1)
    ax_step.fill_between(t_ms, 0.98, 1.02, color=_DIMTXT,
                         alpha=0.08, zorder=1)
    ax_step.plot(t_ms, np.full_like(t_ms, 0.98),
                 color=_DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5)
    ax_step.plot(t_ms, np.full_like(t_ms, 1.02),
                 color=_DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5,
                 label="±2 % band")
    ax_step.axhline(0.0, color=_SPINE, lw=0.6, zorder=1)

    # ── Rise time bracket ─────────────────────────────────────
    if metrics["t_rise"] is not None:
        ss_val = float(np.mean(avg[int(0.80 * len(avg)):]))
        i_lo   = int(np.argmax(avg >= 0.10 * ss_val))
        i_hi   = int(np.argmax(avg >= 0.90 * ss_val))
        tr_ms  = metrics["t_rise"] * 1e3

        ax_step.axvline(t_ms[i_lo], color=_CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.axvline(t_ms[i_hi], color=_CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.scatter([t_ms[i_lo], t_ms[i_hi]],
                        [avg[i_lo], avg[i_hi]],
                        color=_CYAN, s=22, zorder=5)

        mid_t = (t_ms[i_lo] + t_ms[i_hi]) / 2.0
        y_arr = (avg[i_lo] + avg[i_hi]) / 2.0
        ax_step.annotate(
            "", xy=(t_ms[i_hi], y_arr),
            xytext=(t_ms[i_lo], y_arr),
            arrowprops=dict(arrowstyle="<->", color=_CYAN, lw=1.2),
            zorder=5,
        )
        ax_step.text(
            mid_t, y_arr + 0.055,
            f"t_rise = {tr_ms:.2f} ms",
            color=_CYAN, fontsize=8.5, ha="center", va="bottom",
            fontweight="bold", zorder=5,
        )

    # ── Settling time ──────────────────────────────────────────
    if metrics["t_settle"] is not None:
        ts_ms = metrics["t_settle"] * 1e3
        ax_step.axvline(ts_ms, color=_PURPLE, lw=1.3, ls="--", zorder=2,
                        label=f"t_settle = {ts_ms:.2f} ms")
        ax_step.text(
            ts_ms + (t_ms[-1] - t_ms[0]) * 0.007, 0.08,
            f"t_settle\n{ts_ms:.2f} ms",
            color=_PURPLE, fontsize=7.5, va="bottom",
        )

    # ── Overshoot ──────────────────────────────────────────────
    if metrics["overshoot_pct"] is not None and metrics["overshoot_pct"] > 0.3:
        pk_idx = int(np.argmax(avg))
        os_pct = metrics["overshoot_pct"]
        ax_step.scatter([t_ms[pk_idx]], [avg[pk_idx]],
                        color=_YELLOW, s=30, zorder=6)
        ax_step.annotate(
            f"OS = {os_pct:.1f} %",
            xy=(t_ms[pk_idx], avg[pk_idx]),
            xytext=(t_ms[pk_idx] + (t_ms[-1] - t_ms[0]) * 0.03,
                    avg[pk_idx] + 0.04),
            color=_YELLOW, fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=_YELLOW, lw=0.9),
            zorder=5,
        )

    # Axis config
    y_lo_s = min(float(avg.min()) - 0.08, -0.12)
    y_hi_s = max(float(avg.max()) + 0.12, 1.25)
    ax_step.set_xlim([0.0, t_ms[-1]])
    ax_step.set_ylim([y_lo_s, y_hi_s])
    ax_step.set_xlabel("Time [ms]")
    ax_step.set_ylabel("Normalized amplitude")
    ax_step.set_title(
        "Step Response   (direct measurement, ensemble averaged)"
    )
    ax_step.legend(loc="lower right", handlelength=1.6, ncol=2)

    # ── Save ──────────────────────────────────────────────────
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    logger.info(f"Step response plot saved → {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
# plot_results()  — 3-panel Bode + Step response
# ════════════════════════════════════════════════════════════
def plot_results(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    frf: dict,
    cfg: MeasurementConfig = None,
    save_path: str = "bandwidth_result.png",
) -> None:
    """
    Four-panel figure
    ─────────────────
    [0,0] Bode Magnitude  (log-x, dB)
    [0,1] Bode Phase      (log-x, deg, unwrapped)
    [1,:] Time-domain ref vs measured
    [2,:] Step response   (IFFT-estimated, ms)
    """
    _apply_style()

    if cfg is None:
        cfg = CFG
    f    = frf["f"]
    bw   = float(frf["bandwidth_hz"])
    H    = frf.get("H")

    # Rebuild H from mag/phase if not stored
    if H is None:
        mag_lin = 10 ** (frf["mag_db"] / 20.0)
        H = mag_lin * np.exp(1j * np.radians(frf["phase_deg"]))

    valid = frf.get("valid", frf["coherence"] > cfg.coh_threshold)
    fmsk  = (f >= cfg.f_start) & (f <= cfg.f_end)

    # Reference level
    rmask  = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
    ref_db = float(np.mean(frf["mag_db"][rmask])) if np.any(rmask) else 0.0

    # Phase (unwrapped already in estimator; guard against re-wrap)
    phase_uw = frf["phase_deg"]

    # Phase margin  (ref_db crossover — closed-loop reference level)
    phase_margin = None
    f_cross      = None
    cross_mask   = valid & fmsk & (f > cfg.ref_f_high)
    for i in range(len(frf["mag_db"]) - 1):
        if cross_mask[i] and frf["mag_db"][i] >= ref_db >= frf["mag_db"][i+1]:
            f_cross      = float(f[i])
            phase_margin = 180.0 + float(phase_uw[i])
            break

    # Gain margin  (from FRF estimator, or recompute)
    gain_margin = frf.get("gain_margin_db")
    f_gm        = frf.get("f_gain_margin")

    # Step response
    fv            = fmsk & valid
    if not np.any(fv):
        logger.warning("No valid coherence data — using all frequency data for step response")
        fv = fmsk
    t_step, s_step = _estimate_step_response(f[fv], H[fv], cfg.fs)
    t_ms           = t_step * 1e3
    metrics        = _step_metrics(t_step, s_step)

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=140)
    fig.patch.set_facecolor(_BG)

    # Suptitle
    parts = [f"BW = {bw:.1f} Hz"]
    if phase_margin is not None:
        parts.append(f"PM = {phase_margin:.1f}°")
    if gain_margin is not None:
        parts.append(f"GM = {gain_margin:.1f} dB")
    if metrics["t_rise"] is not None:
        parts.append(f"t_rise = {metrics['t_rise']*1e3:.2f} ms")
    if metrics["overshoot_pct"] is not None:
        parts.append(f"OS = {metrics['overshoot_pct']:.1f} %")
    if metrics["t_settle"] is not None:
        parts.append(f"t_settle = {metrics['t_settle']*1e3:.2f} ms")

    fig.suptitle(
        "BLDC Current Controller   ·   " + "   |   ".join(parts),
        color=_TXT, fontsize=10.5, fontweight="bold",
        y=0.99, x=0.5,
    )

    gs = gridspec.GridSpec(
        4, 2,
        figure=fig,
        height_ratios=[1.0, 0.6, 0.6, 1.0],
        hspace=0.50, wspace=0.28,
        left=0.07, right=0.97,
        top=0.96, bottom=0.04,
    )

    # ════════════════════════════════════════════════════════
    # [0, 0]  Magnitude
    # ════════════════════════════════════════════════════════
    ax_mag = fig.add_subplot(gs[0, 0])
    _style_ax(ax_mag)

    # Low-coherence shading
    low_coh = ~valid & fmsk
    if np.any(low_coh):
        ax_mag.fill_between(
            f[fmsk], -65, ref_db + 15,
            where=low_coh[fmsk],
            color=_YELLOW, alpha=0.06,
        )
        # small label
        ax_mag.text(
            cfg.f_end * 0.92, ref_db + 10,
            "low γ²", color=_YELLOW, fontsize=6.5,
            ha="right", va="top", alpha=0.7,
        )

    ax_mag.semilogx(
        f[fmsk], frf["mag_db"][fmsk],
        color=_BLUE, lw=1.7, zorder=3, label="|H₁(jω)|",
    )

    # H2 overlay (if available)
    H2 = frf.get("H2")
    if H2 is not None:
        eps = 1e-12
        mag_db_h2 = 20 * np.log10(np.abs(H2) + eps)
        ax_mag.semilogx(
            f[fmsk], mag_db_h2[fmsk],
            color=_CYAN, lw=1.0, ls=":", alpha=0.6, zorder=2,
            label="|H₂(jω)|",
        )

    # Horizontal guides
    ax_mag.axhline(ref_db,       color=_DIMTXT, lw=0.7, ls=":",  zorder=1)
    ax_mag.axhline(ref_db - 3.0, color=_RED,    lw=0.9, ls="--", zorder=2,
                   label=f"−3 dB  ({ref_db-3:.1f} dB)")
    ax_mag.axhline(0.0,          color=_SPINE,  lw=0.6, zorder=1)

    # BW vertical
    ax_mag.axvline(bw, color=_YELLOW, lw=1.4, ls="--", zorder=2,
                   label=f"BW = {bw:.1f} Hz")

    # BW annotation with arrow
    y_ann = ref_db - 3.0
    ax_mag.annotate(
        f"{bw:.1f} Hz",
        xy=(bw, y_ann),
        xytext=(bw * 1.55, y_ann + 6.0),
        color=_YELLOW, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=_YELLOW, lw=0.9,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=5,
    )

    # BW intersection dot
    ax_mag.scatter([bw], [y_ann], color=_YELLOW, s=25, zorder=6)

    # Gain margin annotation
    if f_gm is not None and gain_margin is not None:
        mag_at_gm = float(frf["mag_db"][np.argmin(np.abs(f - f_gm))])
        ax_mag.axvline(f_gm, color=_GREEN, lw=1.0, ls=":", alpha=0.7, zorder=2)
        ax_mag.scatter([f_gm], [mag_at_gm], color=_GREEN, s=25, zorder=6)
        ax_mag.annotate(
            f"GM = {gain_margin:.1f} dB\n@ {f_gm:.1f} Hz",
            xy=(f_gm, mag_at_gm),
            xytext=(f_gm * 0.45, mag_at_gm + 5.0),
            color=_GREEN, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=_GREEN, lw=0.9,
                            connectionstyle="arc3,rad=0.2"),
            zorder=5,
        )

    y_lo = max(float(frf["mag_db"][fmsk].min()) - 6.0, -55.0)
    ax_mag.set_xlim([cfg.f_start, cfg.f_end])
    ax_mag.set_ylim([y_lo, ref_db + 14.0])
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Bode  —  Magnitude")
    ax_mag.legend(loc="lower left", handlelength=1.6)
    ax_mag.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [0, 1]  Phase
    # ════════════════════════════════════════════════════════
    ax_ph = fig.add_subplot(gs[0, 1])
    _style_ax(ax_ph)

    ax_ph.semilogx(
        f[fmsk], phase_uw[fmsk],
        color=_RED, lw=1.7, zorder=3, label="∠H(jω)",
    )

    ax_ph.axhline( -45,  color=_SPINE,  lw=0.6, ls=":",  zorder=1)
    ax_ph.axhline( -90,  color=_YELLOW, lw=0.9, ls="--", zorder=2, label="−90°")
    ax_ph.axhline(-135,  color=_SPINE,  lw=0.6, ls=":",  zorder=1)
    ax_ph.axhline(-180,  color=_RED,    lw=0.7, ls=":",  zorder=1,
                  alpha=0.6, label="−180°")

    # Phase margin annotation
    if f_cross is not None and phase_margin is not None:
        ph_at_cross = float(phase_uw[np.argmin(np.abs(f - f_cross))])
        ax_ph.axvline(f_cross, color=_PURPLE, lw=1.2, ls=":", zorder=2,
                      label=f"f₀dB = {f_cross:.1f} Hz")
        ax_ph.scatter([f_cross], [ph_at_cross],
                      color=_PURPLE, s=25, zorder=6)
        ax_ph.annotate(
            f"PM = {phase_margin:.1f}°",
            xy=(f_cross, ph_at_cross),
            xytext=(f_cross * 0.55, ph_at_cross + 22),
            color=_PURPLE, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=_PURPLE, lw=0.9,
                            connectionstyle="arc3,rad=0.2"),
            zorder=5,
        )

    # Gain margin crossover annotation (phase = -180°)
    if f_gm is not None:
        ax_ph.axvline(f_gm, color=_GREEN, lw=1.0, ls=":", alpha=0.7, zorder=2,
                      label=f"f_{{−180°}} = {f_gm:.1f} Hz")
        ax_ph.scatter([f_gm], [-180.0], color=_GREEN, s=25, zorder=6)

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [1, :]  Time-domain  ref vs measured — full overview
    # ════════════════════════════════════════════════════════
    ax_td = fig.add_subplot(gs[1, :])
    _style_ax(ax_td)

    t_plot = t * 1e3  # convert to ms

    ax_td.plot(t_plot, i_ref,  color=_BLUE,  lw=0.8, alpha=0.7,
               label="i_ref (reference)")
    ax_td.plot(t_plot, i_meas, color=_GREEN, lw=0.6, alpha=0.6,
               label="i_meas (measured)")

    # Highlight zoom region
    t_total = t_plot[-1] - t_plot[0]
    zoom_center = t_plot[0] + t_total * 0.5
    zoom_half   = t_total * 0.03   # 6 % of total → visible waveform cycles
    zoom_lo     = zoom_center - zoom_half
    zoom_hi     = zoom_center + zoom_half
    ax_td.axvspan(zoom_lo, zoom_hi, color=_CYAN, alpha=0.12, zorder=1)
    ax_td.text(zoom_center, ax_td.get_ylim()[0] if ax_td.get_ylim()[0] != 0 else 0,
               "▼ zoom", color=_CYAN, fontsize=7, ha="center", va="bottom",
               alpha=0.8)

    ax_td.set_xlabel("Time [ms]")
    ax_td.set_ylabel("Current [A]")
    ax_td.set_title("Time Domain  —  Reference vs Measured  (overview)")
    ax_td.legend(loc="upper right", handlelength=1.6)

    # ════════════════════════════════════════════════════════
    # [2, :]  Time-domain  ref vs measured — zoomed-in
    # ════════════════════════════════════════════════════════
    ax_zoom = fig.add_subplot(gs[2, :])
    _style_ax(ax_zoom)

    # Select data within zoom range
    zmask = (t_plot >= zoom_lo) & (t_plot <= zoom_hi)
    ax_zoom.plot(t_plot[zmask], i_ref[zmask],  color=_BLUE,  lw=1.5, alpha=0.9,
                 label="i_ref")
    ax_zoom.plot(t_plot[zmask], i_meas[zmask], color=_GREEN, lw=1.2, alpha=0.8,
                 label="i_meas")

    ax_zoom.set_xlim([zoom_lo, zoom_hi])
    ax_zoom.set_xlabel("Time [ms]")
    ax_zoom.set_ylabel("Current [A]")
    ax_zoom.set_title(
        f"Time Domain  —  Zoomed  "
        f"[{zoom_lo:.1f} – {zoom_hi:.1f} ms]"
    )
    ax_zoom.legend(loc="upper right", handlelength=1.6)

    # ════════════════════════════════════════════════════════
    # [3, :]  Step response  (full width)
    # ════════════════════════════════════════════════════════
    ax_step = fig.add_subplot(gs[3, :])
    _style_ax(ax_step)

    # Main step curve
    ax_step.plot(t_ms, s_step, color=_GREEN, lw=2.0, zorder=3,
                 label="step response (IFFT estimated)")

    # Steady-state band  ±2 %
    ax_step.axhline(1.00, color=_DIMTXT, lw=0.8, ls=":",  zorder=1)
    ax_step.fill_between(t_ms, 0.98, 1.02, color=_DIMTXT,
                         alpha=0.08, zorder=1)
    ax_step.plot(t_ms, np.full_like(t_ms, 0.98),
                 color=_DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5)
    ax_step.plot(t_ms, np.full_like(t_ms, 1.02),
                 color=_DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5,
                 label="±2 % band")

    # Zero line
    ax_step.axhline(0.0, color=_SPINE, lw=0.6, zorder=1)

    # ── Rise time bracket ────────────────────────────────────
    if metrics["t_rise"] is not None:
        ss_val = float(np.mean(s_step[int(0.80*len(s_step)):]))
        i_lo   = int(np.argmax(s_step >= 0.10 * ss_val))
        i_hi   = int(np.argmax(s_step >= 0.90 * ss_val))
        tr_ms  = metrics["t_rise"] * 1e3

        # Vertical dashed markers at 10 % / 90 %
        ax_step.axvline(t_ms[i_lo], color=_CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.axvline(t_ms[i_hi], color=_CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.scatter([t_ms[i_lo], t_ms[i_hi]],
                        [s_step[i_lo], s_step[i_hi]],
                        color=_CYAN, s=22, zorder=5)

        # Double-headed arrow
        mid_t  = (t_ms[i_lo] + t_ms[i_hi]) / 2.0
        y_arr  = (s_step[i_lo] + s_step[i_hi]) / 2.0
        ax_step.annotate(
            "", xy=(t_ms[i_hi], y_arr),
            xytext=(t_ms[i_lo], y_arr),
            arrowprops=dict(arrowstyle="<->", color=_CYAN, lw=1.2),
            zorder=5,
        )
        ax_step.text(
            mid_t, y_arr + 0.055,
            f"t_rise = {tr_ms:.2f} ms",
            color=_CYAN, fontsize=8.5, ha="center", va="bottom",
            fontweight="bold", zorder=5,
        )

    # ── Settling time ─────────────────────────────────────────
    if metrics["t_settle"] is not None:
        ts_ms = metrics["t_settle"] * 1e3
        ax_step.axvline(ts_ms, color=_PURPLE, lw=1.3, ls="--", zorder=2,
                        label=f"t_settle = {ts_ms:.2f} ms")
        ax_step.text(
            ts_ms + (t_ms[-1] - t_ms[0]) * 0.007, 0.08,
            f"t_settle\n{ts_ms:.2f} ms",
            color=_PURPLE, fontsize=7.5, va="bottom",
        )

    # ── Overshoot annotation ──────────────────────────────────
    if metrics["overshoot_pct"] is not None and metrics["overshoot_pct"] > 0.3:
        pk_idx  = int(np.argmax(s_step))
        os_pct  = metrics["overshoot_pct"]
        ax_step.scatter([t_ms[pk_idx]], [s_step[pk_idx]],
                        color=_YELLOW, s=30, zorder=6)
        ax_step.annotate(
            f"OS = {os_pct:.1f} %",
            xy=(t_ms[pk_idx], s_step[pk_idx]),
            xytext=(t_ms[pk_idx] + (t_ms[-1]-t_ms[0])*0.03,
                    s_step[pk_idx] + 0.04),
            color=_YELLOW, fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=_YELLOW, lw=0.9),
            zorder=5,
        )

    # Axis config
    y_lo_s = min(float(s_step.min()) - 0.08, -0.12)
    y_hi_s = max(float(s_step.max()) + 0.12,  1.25)
    ax_step.set_xlim([0.0, t_ms[-1]])
    ax_step.set_ylim([y_lo_s, y_hi_s])
    ax_step.set_xlabel("Time [ms]")
    ax_step.set_ylabel("Normalized amplitude")
    ax_step.set_title(
        "Step Response   (numerically estimated:  H(jω) → IFFT → impulse h(t) → ∫h dτ)"
    )
    ax_step.legend(loc="lower right", handlelength=1.6, ncol=2)

    # Estimation method footnote
    ax_step.text(
        0.995, 0.018,
        "⚠  Estimated under small-signal linearity assumption  —  "
        "validate with hardware step test (TEST_SIG_STEP)",
        transform=ax_step.transAxes,
        color=_DIMTXT, fontsize=6.5, ha="right", va="bottom", style="italic",
    )

    # ── Save ─────────────────────────────────────────────────
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    logger.info(f"Plot saved → {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
# Comparison plot  (Chirp vs Multisine overlay)
# ════════════════════════════════════════════════════════════
def _load_frf_from_npz(path: str) -> dict:
    """Load an .npz file and reconstruct an FRF dict for plotting."""
    d = np.load(path)
    f        = d["f"]
    mag_db   = d["mag_db"]
    phase_deg = d["phase_deg"]
    coherence = d["coherence"]
    bw_hz    = float(d["bandwidth_hz"][0])

    mag_lin = 10 ** (mag_db / 20.0)
    H = mag_lin * np.exp(1j * np.radians(phase_deg))

    cfg = CFG
    valid = coherence > cfg.coh_threshold

    signal_type = str(d["signal_type"]) if "signal_type" in d else "unknown"

    return dict(
        f=f, H=H, mag_db=mag_db,
        phase_deg=phase_deg, coherence=coherence,
        valid=valid, bandwidth_hz=bw_hz,
        signal_type=signal_type,
    )


def plot_comparison(
    frf_a: dict,
    frf_b: dict,
    cfg: MeasurementConfig,
    save_path: str = "bandwidth_comparison.png",
) -> None:
    """
    Four-panel comparison figure
    ────────────────────────────
    Top-left   : Magnitude overlay
    Top-right  : Phase overlay
    Bot-left   : Coherence overlay
    Bot-right  : Step response overlay
    """
    _apply_style()

    fig = plt.figure(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor(_BG)

    label_a = frf_a.get("signal_type", "A")
    label_b = frf_b.get("signal_type", "B")
    bw_a = float(frf_a["bandwidth_hz"])
    bw_b = float(frf_b["bandwidth_hz"])

    fig.suptitle(
        f"{label_a} vs {label_b}   ·   "
        f"BW_{label_a} = {bw_a:.1f} Hz   |   BW_{label_b} = {bw_b:.1f} Hz",
        color=_TXT, fontsize=10.5, fontweight="bold", y=0.985,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        hspace=0.42, wspace=0.28,
        left=0.07, right=0.97, top=0.94, bottom=0.08,
    )

    fmsk_a = (frf_a["f"] >= cfg.f_start) & (frf_a["f"] <= cfg.f_end)
    fmsk_b = (frf_b["f"] >= cfg.f_start) & (frf_b["f"] <= cfg.f_end)

    valid_a = frf_a["valid"]
    valid_b = frf_b["valid"]

    # Reference level (from first dataset)
    rmask  = valid_a & (frf_a["f"] >= cfg.ref_f_low) & (frf_a["f"] <= cfg.ref_f_high)
    ref_db = float(np.mean(frf_a["mag_db"][rmask])) if np.any(rmask) else 0.0

    # ════════════════════════════════════════════════════════
    # [0, 0]  Magnitude overlay
    # ════════════════════════════════════════════════════════
    ax_mag = fig.add_subplot(gs[0, 0])
    _style_ax(ax_mag)

    ax_mag.semilogx(
        frf_a["f"][fmsk_a], frf_a["mag_db"][fmsk_a],
        color=_BLUE, lw=1.7, zorder=3, label=f"{label_a}  (BW={bw_a:.1f} Hz)",
    )
    ax_mag.semilogx(
        frf_b["f"][fmsk_b], frf_b["mag_db"][fmsk_b],
        color=_CYAN, lw=1.5, ls="--", zorder=3, label=f"{label_b}  (BW={bw_b:.1f} Hz)",
    )

    ax_mag.axhline(ref_db - 3.0, color=_RED, lw=0.9, ls="--", zorder=2,
                   label=f"−3 dB  ({ref_db-3:.1f} dB)")
    ax_mag.axhline(0.0, color=_SPINE, lw=0.6, zorder=1)

    ax_mag.axvline(bw_a, color=_BLUE,  lw=1.0, ls=":", alpha=0.7, zorder=2)
    ax_mag.axvline(bw_b, color=_CYAN,  lw=1.0, ls=":", alpha=0.7, zorder=2)

    y_lo = max(float(min(frf_a["mag_db"][fmsk_a].min(),
                         frf_b["mag_db"][fmsk_b].min())) - 6.0, -55.0)
    ax_mag.set_xlim([cfg.f_start, cfg.f_end])
    ax_mag.set_ylim([y_lo, ref_db + 14.0])
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Bode  —  Magnitude")
    ax_mag.legend(loc="lower left", handlelength=1.6)
    ax_mag.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [0, 1]  Phase overlay
    # ════════════════════════════════════════════════════════
    ax_ph = fig.add_subplot(gs[0, 1])
    _style_ax(ax_ph)

    ax_ph.semilogx(
        frf_a["f"][fmsk_a], frf_a["phase_deg"][fmsk_a],
        color=_BLUE, lw=1.7, zorder=3, label=label_a,
    )
    ax_ph.semilogx(
        frf_b["f"][fmsk_b], frf_b["phase_deg"][fmsk_b],
        color=_CYAN, lw=1.5, ls="--", zorder=3, label=label_b,
    )

    ax_ph.axhline(-90,  color=_YELLOW, lw=0.9, ls="--", zorder=2, label="−90°")
    ax_ph.axhline(-180, color=_RED,    lw=0.7, ls=":",  zorder=1, alpha=0.6, label="−180°")

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [1, 0]  Coherence overlay
    # ════════════════════════════════════════════════════════
    ax_coh = fig.add_subplot(gs[1, 0])
    _style_ax(ax_coh)

    ax_coh.semilogx(
        frf_a["f"][fmsk_a], frf_a["coherence"][fmsk_a],
        color=_BLUE, lw=1.5, zorder=3, label=label_a,
    )
    ax_coh.semilogx(
        frf_b["f"][fmsk_b], frf_b["coherence"][fmsk_b],
        color=_CYAN, lw=1.5, ls="--", zorder=3, label=label_b,
    )

    ax_coh.axhline(cfg.coh_threshold, color=_YELLOW, lw=1.0, ls="--", zorder=2,
                   label=f"threshold γ² = {cfg.coh_threshold}")
    ax_coh.set_xlim([cfg.f_start, cfg.f_end])
    ax_coh.set_ylim([-0.05, 1.08])
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylabel("Coherence γ²")
    ax_coh.set_title("Coherence")
    ax_coh.legend(loc="lower left", handlelength=1.6)
    ax_coh.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [1, 1]  Step response overlay
    # ════════════════════════════════════════════════════════
    ax_step = fig.add_subplot(gs[1, 1])
    _style_ax(ax_step)

    H_a = frf_a["H"]
    H_b = frf_b["H"]

    fv_a = fmsk_a & valid_a
    fv_b = fmsk_b & valid_b

    t_sa, s_sa = _estimate_step_response(frf_a["f"][fv_a], H_a[fv_a], cfg.fs)
    t_sb, s_sb = _estimate_step_response(frf_b["f"][fv_b], H_b[fv_b], cfg.fs)

    ax_step.plot(t_sa * 1e3, s_sa, color=_BLUE, lw=1.7, zorder=3, label=label_a)
    ax_step.plot(t_sb * 1e3, s_sb, color=_CYAN, lw=1.5, ls="--", zorder=3, label=label_b)

    ax_step.axhline(1.00, color=_DIMTXT, lw=0.8, ls=":", zorder=1)
    ax_step.fill_between(t_sa * 1e3, 0.98, 1.02, color=_DIMTXT, alpha=0.08, zorder=1)
    ax_step.axhline(0.0, color=_SPINE, lw=0.6, zorder=1)

    y_lo_s = min(float(min(s_sa.min(), s_sb.min())) - 0.08, -0.12)
    y_hi_s = max(float(max(s_sa.max(), s_sb.max())) + 0.12, 1.25)
    ax_step.set_xlim([0.0, max(t_sa[-1], t_sb[-1]) * 1e3])
    ax_step.set_ylim([y_lo_s, y_hi_s])
    ax_step.set_xlabel("Time [ms]")
    ax_step.set_ylabel("Normalized amplitude")
    ax_step.set_title("Step Response  (IFFT estimated)")
    ax_step.legend(loc="lower right", handlelength=1.6)

    # ── Save ──────────────────────────────────────────────
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    logger.info(f"Comparison plot saved → {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
# Nyquist plot
# ════════════════════════════════════════════════════════════
def plot_nyquist(
    frf: dict,
    cfg: MeasurementConfig = None,
    save_path: str = "nyquist_result.png",
) -> None:
    """
    Nyquist diagram of H(jω) with unit circle and stability annotations.
    """
    _apply_style()
    if cfg is None:
        cfg = CFG

    f     = frf["f"]
    H     = frf.get("H")
    if H is None:
        mag_lin = 10 ** (frf["mag_db"] / 20.0)
        H = mag_lin * np.exp(1j * np.radians(frf["phase_deg"]))

    valid = frf.get("valid", frf["coherence"] > cfg.coh_threshold)
    fmsk  = (f >= cfg.f_start) & (f <= cfg.f_end) & valid

    H_plot = H[fmsk]
    f_plot = f[fmsk]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)
    fig.patch.set_facecolor(_BG)
    _style_ax(ax)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta),
            color=_DIMTXT, lw=0.8, ls="--", alpha=0.5, label="unit circle")

    # Nyquist curve
    ax.plot(H_plot.real, H_plot.imag,
            color=_BLUE, lw=1.8, zorder=3, label="H(jω)")
    ax.plot(H_plot.real, -H_plot.imag,
            color=_BLUE, lw=1.0, ls=":", alpha=0.4, zorder=2,
            label="H(−jω)")

    # Critical point (-1, 0)
    ax.scatter([-1], [0], color=_RED, s=60, zorder=6, marker="x", linewidths=2)
    ax.annotate("(−1, 0)", xy=(-1, 0), xytext=(-1.15, 0.15),
                color=_RED, fontsize=8, fontweight="bold", zorder=5)

    # Frequency markers
    n_markers = 8
    marker_idx = np.linspace(0, len(H_plot) - 1, n_markers, dtype=int)
    for mi in marker_idx:
        ax.scatter([H_plot[mi].real], [H_plot[mi].imag],
                   color=_YELLOW, s=18, zorder=5)
        ax.annotate(
            f"{f_plot[mi]:.0f}",
            xy=(H_plot[mi].real, H_plot[mi].imag),
            xytext=(5, 5), textcoords="offset points",
            color=_YELLOW, fontsize=6.5, zorder=5,
        )

    # Start/end markers
    ax.scatter([H_plot[0].real], [H_plot[0].imag],
               color=_GREEN, s=40, zorder=6, marker="o",
               label=f"f_start = {f_plot[0]:.0f} Hz")
    ax.scatter([H_plot[-1].real], [H_plot[-1].imag],
               color=_RED, s=40, zorder=6, marker="s",
               label=f"f_end = {f_plot[-1]:.0f} Hz")

    # Axes
    ax.axhline(0, color=_SPINE, lw=0.6)
    ax.axvline(0, color=_SPINE, lw=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Nyquist Diagram  —  H(jω)")
    ax.legend(loc="lower left", handlelength=1.6, fontsize=7.5)

    bw = float(frf["bandwidth_hz"])
    gm = frf.get("gain_margin_db")
    pm_parts = [f"BW = {bw:.1f} Hz"]
    if gm is not None:
        pm_parts.append(f"GM = {gm:.1f} dB")
    ax.text(
        0.98, 0.98, "   |   ".join(pm_parts),
        transform=ax.transAxes, color=_TXT, fontsize=8.5,
        ha="right", va="top", fontweight="bold",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    logger.info(f"Nyquist plot saved → {save_path}")
    plt.show()


# ════════════════════════════════════════════════════════════
# CSV / JSON export
# ════════════════════════════════════════════════════════════
def export_csv(frf: dict, path: str = "bandwidth_result.csv") -> None:
    """Export FRF data to CSV for MATLAB / Excel interoperability."""
    f         = frf["f"]
    mag_db    = frf["mag_db"]
    phase_deg = frf["phase_deg"]
    coherence = frf["coherence"]

    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frequency_hz", "magnitude_db", "phase_deg", "coherence"])
        for i in range(len(f)):
            writer.writerow([
                f"{f[i]:.6f}",
                f"{mag_db[i]:.6f}",
                f"{phase_deg[i]:.6f}",
                f"{coherence[i]:.6f}",
            ])
    logger.info(f"CSV exported → {path}")


def export_json(frf: dict, path: str = "bandwidth_result.json") -> None:
    """Export FRF summary + data to JSON."""
    out = {
        "bandwidth_hz": float(frf["bandwidth_hz"]),
        "gain_margin_db": frf.get("gain_margin_db"),
        "f_gain_margin_hz": frf.get("f_gain_margin"),
        "data": {
            "frequency_hz": frf["f"].tolist(),
            "magnitude_db": frf["mag_db"].tolist(),
            "phase_deg": frf["phase_deg"].tolist(),
            "coherence": frf["coherence"].tolist(),
        },
    }
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2, default=float)
    logger.info(f"JSON exported → {path}")


# ════════════════════════════════════════════════════════════
# Real-time UDP monitor
# ════════════════════════════════════════════════════════════
def run_monitor(cfg: MeasurementConfig = None, window_sec: float = 5.0) -> None:
    """
    Live scrolling plot of UDP i_ref / i_meas for connection debugging.
    Press Ctrl+C to stop.
    """
    if cfg is None:
        cfg = CFG

    _apply_style()
    fig, (ax_sig, ax_err) = plt.subplots(2, 1, figsize=(12, 6), dpi=100,
                                          gridspec_kw={"height_ratios": [2, 1],
                                                       "hspace": 0.35})
    fig.patch.set_facecolor(_BG)
    _style_ax(ax_sig)
    _style_ax(ax_err)

    max_pts   = int(window_sec * cfg.fs)
    t_buf     = deque(maxlen=max_pts)
    ref_buf   = deque(maxlen=max_pts)
    meas_buf  = deque(maxlen=max_pts)

    receiver = UDPReceiver(cfg)
    receiver.start()
    logger.info(f"Monitor started  ({window_sec}s window)  —  Ctrl+C to stop")

    line_ref,  = ax_sig.plot([], [], color=_BLUE,  lw=1.2, label="i_ref")
    line_meas, = ax_sig.plot([], [], color=_GREEN, lw=1.0, label="i_meas")
    line_err,  = ax_err.plot([], [], color=_RED,   lw=1.0, label="error")

    ax_sig.set_ylabel("Current [A]")
    ax_sig.set_title("Real-Time UDP Monitor")
    ax_sig.legend(loc="upper right", handlelength=1.2)

    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [A]")
    ax_err.legend(loc="upper right", handlelength=1.2)

    pkt_text = ax_sig.text(
        0.01, 0.97, "", transform=ax_sig.transAxes,
        color=_DIMTXT, fontsize=7.5, va="top", fontfamily="monospace",
    )

    def _update(frame):
        data = receiver.get_data()
        if not data:
            return line_ref, line_meas, line_err, pkt_text

        for dp in data:
            t_buf.append(dp.t)
            ref_buf.append(dp.i_ref)
            meas_buf.append(dp.i_meas)

        # Clear receiver buffer to avoid re-processing
        with receiver._lock:
            receiver.buffer.clear()

        t_arr   = np.array(t_buf)
        ref_arr = np.array(ref_buf)
        meas_arr= np.array(meas_buf)

        line_ref.set_data(t_arr, ref_arr)
        line_meas.set_data(t_arr, meas_arr)
        line_err.set_data(t_arr, ref_arr - meas_arr)

        if len(t_arr) > 1:
            t_lo, t_hi = t_arr[0], t_arr[-1]
            ax_sig.set_xlim(t_lo, t_hi)
            ax_err.set_xlim(t_lo, t_hi)

            y_max = max(abs(ref_arr).max(), abs(meas_arr).max(), 0.01) * 1.1
            ax_sig.set_ylim(-y_max, y_max)

            e_max = max(abs(ref_arr - meas_arr).max(), 0.001) * 1.3
            ax_err.set_ylim(-e_max, e_max)

        stats = receiver.stats()
        pkt_text.set_text(
            f"rx={stats['received']}  drop={stats['dropped']}  "
            f"buf={len(t_buf)}"
        )

        return line_ref, line_meas, line_err, pkt_text

    ani = FuncAnimation(fig, _update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        receiver.stop()


# ════════════════════════════════════════════════════════════
# Demo mode  (synthetic second-order system)
# ════════════════════════════════════════════════════════════
def _simulate_plant(t, i_ref, cfg, rng_seed=42, noisy=False):
    """Pass signal through synthetic 2nd-order plant + noise.

    Parameters
    ----------
    noisy : bool
        If True, adds realistic BLDC noise:
        - PWM switching ripple (tonal peaks at 20 kHz harmonics,
          aliased into measurement band)
        - Rotor electrical ripple (~6× mechanical speed)
        - Impulse spikes (simulating EMI / commutation glitches)
        - Higher broadband noise floor
    """
    wn   = 2 * np.pi * 180.0
    zeta = 0.65
    num  = [wn**2]
    den  = [1, 2*zeta*wn, wn**2]
    sys  = signal.TransferFunction(num, den)
    _, i_meas, _ = signal.lsim(sys, i_ref, t)

    rng = np.random.default_rng(rng_seed)

    if noisy:
        fs = cfg.fs
        # ── 1. Broadband noise (higher level) ─────────────────
        i_meas += rng.normal(0, cfg.amplitude * 0.05, size=len(i_meas))

        # ── 2. PWM switching ripple (aliased tonal peaks) ──────
        #   Real PWM at 20 kHz aliases to 20kHz mod fs; add a few
        #   strong tonal peaks in the measurement band
        for f_tone in [47.0, 153.0, 347.0]:
            amp_tone = cfg.amplitude * 0.15
            i_meas += amp_tone * np.sin(2 * np.pi * f_tone * t
                                        + rng.uniform(0, 2*np.pi))

        # ── 3. Rotor electrical ripple ─────────────────────────
        f_rotor = 23.5  # ~6× mechanical speed for 4-pole-pair motor
        i_meas += cfg.amplitude * 0.05 * np.sin(2 * np.pi * f_rotor * t)

        # ── 4. Impulse spikes (EMI / commutation) ─────────────
        n_spikes = max(10, int(len(t) * 0.008))  # ~0.8% of samples
        spike_idx = rng.choice(len(t), size=n_spikes, replace=False)
        spike_amp = rng.choice([-1, 1], size=n_spikes) * cfg.amplitude * 2.5
        i_meas[spike_idx] += spike_amp
    else:
        # Clean: minimal broadband noise
        i_meas += rng.normal(0, cfg.amplitude * 0.02, size=len(i_meas))

    return i_meas


def _run_demo_single(signal_type: str, cfg: MeasurementConfig,
                     noisy: bool = False,
                     out: Optional[OutputManager] = None) -> Optional[dict]:
    """Run demo for a single signal type. Returns frf dict or step metrics."""
    est = FRFEstimator(cfg)
    noise_tag = " (noisy + adaptive)" if noisy else ""
    _p = out.path if out else lambda f: f  # path resolver

    if signal_type == "step":
        logger.info(f"── Demo ── Step excitation{noise_tag}")
        gen = StepGenerator(cfg)
        t, i_ref = gen.get_full_reference()
        i_meas   = _simulate_plant(t, i_ref, cfg, noisy=noisy)

        # Adaptive preprocessing for step test
        if noisy:
            analyzer = NoiseAnalyzer(cfg)
            report   = analyzer.analyze(t, i_ref, i_meas)
            preprocessor = AdaptivePreprocessor(cfg)
            i_meas_clean = preprocessor.apply(i_meas, report, i_ref=i_ref)
            plot_noise_analysis(t, i_ref, i_meas, i_meas_clean, report, cfg,
                                save_path=_p("noise_analysis_step.png"))
            i_meas = i_meas_clean

        step_data = _analyze_step_response(t, i_ref, i_meas, cfg)
        m   = step_data["metrics"]
        frf = step_data.get("frf")

        if frf is not None:
            logger.info(f"  BW estimate : {frf['bandwidth_hz']:.1f} Hz  (true: ~180 Hz)")
        logger.info(
            f"  Step metrics: t_rise={m['t_rise']*1e3:.2f}ms  "
            f"OS={m['overshoot_pct']:.1f}%  "
            f"t_settle={m['t_settle']*1e3:.2f}ms"
        )

        save_npz = _p("step_response_raw.npz")
        save_dict = dict(
            t=t, i_ref=i_ref, i_meas=i_meas,
            t_step=step_data["t_step"],
            avg=step_data["avg"],
            signal_type=np.array("step"),
        )
        save_dict.update(
            {f"resp_{i}": r for i, r in enumerate(step_data["responses"])}
        )
        save_dict.update(
            {f"metric_{k}": np.array([v]) for k, v in m.items()
             if v is not None}
        )
        if frf is not None:
            save_dict.update(
                f=frf["f"], mag_db=frf["mag_db"],
                phase_deg=frf["phase_deg"],
                coherence=frf["coherence"],
                bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            )
        np.savez(save_npz, **save_dict)
        logger.info(f"  Raw data saved → {save_npz}")
        plot_step_results(t, i_ref, i_meas, step_data, cfg,
                          save_path=_p("step_response_result.png"))
        return step_data["metrics"]

    if signal_type == "multisine":
        logger.info(f"── Demo ── Multisine excitation{noise_tag}")
        gen = MultisineGenerator(cfg)
    else:
        logger.info(f"── Demo ── Chirp excitation{noise_tag}")
        gen = ChirpGenerator(cfg)

    t, i_ref = gen.get_full_reference()
    i_meas   = _simulate_plant(t, i_ref, cfg, noisy=noisy)

    # ── Adaptive noise pipeline ────────────────────────────
    noise_report = None
    if noisy:
        analyzer = NoiseAnalyzer(cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            cfg, save_path=_p(f"noise_analysis_{signal_type}.png"))

        # Use filtered data for FRF estimation
        i_meas = i_meas_filtered

    frf = est.estimate(t, i_ref, i_meas, noise_report=noise_report)

    logger.info(f"  BW estimate : {frf['bandwidth_hz']:.1f} Hz  (true: ~180 Hz)")

    save_npz = _p(f"bandwidth_raw_{signal_type}.npz")
    np.savez(
        save_npz,
        t=t, i_ref=i_ref, i_meas=i_meas,
        f=frf["f"], mag_db=frf["mag_db"],
        phase_deg=frf["phase_deg"],
        coherence=frf["coherence"],
        bandwidth_hz=np.array([frf["bandwidth_hz"]]),
        signal_type=np.array(signal_type),
    )
    logger.info(f"  Raw data saved → {save_npz}")
    plot_results(t, i_ref, i_meas, frf, cfg=cfg,
                 save_path=_p(f"bandwidth_result_{signal_type}.png"))
    return frf


def _run_demo(signal_type: str = "all", noisy: bool = False) -> None:
    """
    Simulate a 2nd-order current controller:
        H(s) = ωn² / (s² + 2ζωn·s + ωn²)
    with ωn = 2π·180 rad/s, ζ = 0.65

    signal_type: "all", "chirp", "multisine", or "step"
    """
    cfg = CFG
    out = OutputManager()

    if signal_type == "all":
        logger.info("── Demo mode ── sequential: chirp → multisine → step")
        for sig in ("chirp", "multisine", "step"):
            frf = _run_demo_single(sig, cfg, noisy=noisy, out=out)
            # Export CSV/JSON for FRF results (not step metrics)
            if isinstance(frf, dict) and "f" in frf:
                export_csv(frf, path=out.path(f"bandwidth_result_{sig}.csv"))
                export_json(frf, path=out.path(f"bandwidth_result_{sig}.json"))
    else:
        logger.info(f"── Demo mode ── single signal: {signal_type}")
        frf = _run_demo_single(signal_type, cfg, noisy=noisy, out=out)
        if isinstance(frf, dict) and "f" in frf:
            export_csv(frf, path=out.path(f"bandwidth_result_{signal_type}.csv"))
            export_json(frf, path=out.path(f"bandwidth_result_{signal_type}.json"))

    out.log_structure()


# ════════════════════════════════════════════════════════════
# Live measurement
# ════════════════════════════════════════════════════════════
class BandwidthMeasurement:
    """Full bandwidth measurement: chirp → multisine → step (sequential).

    The STM32 sends all three signal types in a single session.
    Each phase is analysed independently and saved to separate files.
    A single signal type can still be run via ``signal_type`` param.
    """

    def __init__(self, cfg: MeasurementConfig = CFG, signal_type: str = "all"):
        self.cfg         = cfg
        self.signal_type = signal_type
        self.receiver    = UDPReceiver(cfg)
        self.estimator   = FRFEstimator(cfg)
        self.out         = OutputManager()

    # ── helpers ───────────────────────────────────────────
    def _analyze_frf_phase(self, phase: str, data: list) -> Optional[dict]:
        """Preprocess + FRF estimation for a chirp/multisine phase."""
        if not data:
            logger.warning(f"[{phase}] no data collected — skipping")
            return None

        logger.info(f"[{phase}] collected {len(data)} samples")
        t, i_ref, i_meas, fs_det = preprocess(data, self.cfg.fs)
        self.cfg.fs = fs_det

        # ── Adaptive noise analysis & preprocessing ────────────
        analyzer = NoiseAnalyzer(self.cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(self.cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            self.cfg,
                            save_path=self.out.path(f"noise_analysis_{phase}.png"))

        i_meas = i_meas_filtered

        frf = self.estimator.estimate(t, i_ref, i_meas,
                                      noise_report=noise_report)
        logger.info(f"  ★  [{phase}] Bandwidth : {frf['bandwidth_hz']:.1f} Hz  "
                    f"(estimator={frf.get('estimator', 'H1')}, "
                    f"nperseg={frf.get('nperseg_used', self.cfg.nperseg)})")

        save_npz = self.out.path(f"bandwidth_raw_{phase}.npz")
        np.savez(
            save_npz,
            t=t, i_ref=i_ref, i_meas=i_meas,
            f=frf["f"], mag_db=frf["mag_db"],
            phase_deg=frf["phase_deg"],
            coherence=frf["coherence"],
            bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            signal_type=np.array(phase),
        )
        logger.info(f"  Raw data saved → {save_npz}")

        plot_results(t, i_ref, i_meas, frf, cfg=self.cfg,
                     save_path=self.out.path(f"bandwidth_result_{phase}.png"))

        export_csv(frf, path=self.out.path(f"bandwidth_result_{phase}.csv"))
        export_json(frf, path=self.out.path(f"bandwidth_result_{phase}.json"))
        return frf

    def _analyze_step_phase(self, data: list) -> Optional[dict]:
        """Preprocess + step response analysis."""
        if not data:
            logger.warning("[step] no data collected — skipping")
            return None

        logger.info(f"[step] collected {len(data)} samples")
        t, i_ref, i_meas, fs_det = preprocess(data, self.cfg.fs)
        self.cfg.fs = fs_det

        # ── Adaptive noise analysis & preprocessing ────────────
        analyzer = NoiseAnalyzer(self.cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(self.cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            self.cfg,
                            save_path=self.out.path("noise_analysis_step.png"))

        i_meas = i_meas_filtered

        step_data = _analyze_step_response(t, i_ref, i_meas, self.cfg)
        m   = step_data["metrics"]
        frf = step_data.get("frf")

        if frf is not None:
            logger.info(f"  ★  [step] Bandwidth   : {frf['bandwidth_hz']:.1f} Hz")
        if m["t_rise"] is not None:
            logger.info(f"  ★  [step] Rise time   : {m['t_rise']*1e3:.2f} ms")
        if m["overshoot_pct"] is not None:
            logger.info(f"  ★  [step] Overshoot   : {m['overshoot_pct']:.1f} %")
        if m["t_settle"] is not None:
            logger.info(f"  ★  [step] Settle time : {m['t_settle']*1e3:.2f} ms")

        save_npz = self.out.path("step_response_raw.npz")
        save_dict = dict(
            t=t, i_ref=i_ref, i_meas=i_meas,
            t_step=step_data["t_step"],
            avg=step_data["avg"],
            signal_type=np.array("step"),
        )
        save_dict.update(
            {f"resp_{i}": r for i, r in enumerate(step_data["responses"])}
        )
        save_dict.update(
            {f"metric_{k}": np.array([v]) for k, v in m.items()
             if v is not None}
        )
        if frf is not None:
            save_dict.update(
                f=frf["f"], mag_db=frf["mag_db"],
                phase_deg=frf["phase_deg"],
                coherence=frf["coherence"],
                bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            )
        np.savez(save_npz, **save_dict)
        logger.info(f"  Raw data saved → {save_npz}")

        plot_step_results(t, i_ref, i_meas, step_data, self.cfg,
                          save_path=self.out.path("step_response_result.png"))
        return step_data["metrics"]

    # ── main entry ────────────────────────────────────────
    def run(self) -> dict:
        cfg = self.cfg

        # Estimate total duration for all phases
        chirp_dur = cfg.chirp_duration
        msine_dur = cfg.chirp_duration
        step_dur  = (cfg.step_settle + cfg.step_hold) * cfg.step_repeats
        total_dur = chirp_dur + msine_dur + step_dur

        logger.info("=" * 58)
        logger.info("BLDC Current Controller  Bandwidth Measurement")
        if self.signal_type == "all":
            logger.info(f"  Mode   : sequential (chirp → multisine → step)")
        else:
            logger.info(f"  Mode   : single ({self.signal_type})")
        logger.info(f"  Freq   : {cfg.f_start}–{cfg.f_end} Hz  A={cfg.amplitude}A")
        logger.info(f"  UDP    : {cfg.udp_host}:{cfg.udp_port}")
        logger.info("=" * 58)

        self.receiver.start()

        logger.info("Waiting for 'bandwidth measure start' from STM32 …")
        if not self.receiver.wait_for_start(timeout=60.0):
            self.receiver.stop()
            raise TimeoutError("Timed out waiting for start message from STM32")

        logger.info(f"Recording … (timeout {total_dur + 60.0:.0f} s)")
        if not self.receiver.wait_for_done(timeout=total_dur + 60.0):
            logger.warning("Timed out waiting for done message — using collected data")

        self.receiver.stop()
        logger.info(f"UDP stats: {self.receiver.stats()}")

        results: dict = {}

        # ── analyse each phase ────────────────────────────
        if self.signal_type in ("all", "chirp"):
            data = self.receiver.get_phase_data("chirp")
            frf_chirp = self._analyze_frf_phase("chirp", data)
            if frf_chirp:
                results["chirp"] = frf_chirp

        if self.signal_type in ("all", "multisine"):
            data = self.receiver.get_phase_data("multisine")
            frf_msine = self._analyze_frf_phase("multisine", data)
            if frf_msine:
                results["multisine"] = frf_msine

        if self.signal_type in ("all", "step"):
            data = self.receiver.get_phase_data("step")
            step_metrics = self._analyze_step_phase(data)
            if step_metrics:
                results["step"] = step_metrics

        logger.info("=" * 58)
        logger.info("Measurement complete — all phases processed")
        self.out.log_structure()
        logger.info("=" * 58)
        return results


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BLDC current controller bandwidth measurement"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run with synthetic 2nd-order plant (no hardware needed)",
    )
    parser.add_argument(
        "--signal", choices=["all", "chirp", "multisine", "step"],
        default="all",
        help="Signal type: 'all' runs sequential chirp→multisine→step "
             "(default). Single types also supported.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar="NPZ",
        help="Compare two saved .npz result files. "
             "Example: --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz",
    )
    parser.add_argument(
        "--nyquist", metavar="NPZ",
        help="Generate Nyquist plot from a saved .npz result file.",
    )
    parser.add_argument(
        "--monitor", action="store_true",
        help="Real-time UDP monitor (scrolling i_ref/i_meas plot). "
             "Press Ctrl+C to stop.",
    )
    parser.add_argument(
        "--export-csv", metavar="NPZ",
        help="Export .npz result to CSV. "
             "Example: --export-csv bandwidth_raw_chirp.npz",
    )
    parser.add_argument(
        "--export-json", metavar="NPZ",
        help="Export .npz result to JSON. "
             "Example: --export-json bandwidth_raw_chirp.npz",
    )
    parser.add_argument(
        "--noisy", action="store_true",
        help="(demo only) Add realistic BLDC noise (PWM ripple, spikes, "
             "rotor harmonics) and run adaptive preprocessing pipeline.",
    )
    args = parser.parse_args()

    cfg = CFG

    if args.monitor:
        run_monitor(cfg)
    elif args.nyquist:
        om  = OutputManager()
        frf = _load_frf_from_npz(args.nyquist)
        plot_nyquist(frf, cfg, save_path=om.path("nyquist_result.png"))
        om.log_structure()
    elif args.export_csv:
        om  = OutputManager()
        frf = _load_frf_from_npz(args.export_csv)
        base = os.path.splitext(os.path.basename(args.export_csv))[0]
        export_csv(frf, path=om.path(f"{base}.csv"))
        om.log_structure()
    elif args.export_json:
        om  = OutputManager()
        frf = _load_frf_from_npz(args.export_json)
        base = os.path.splitext(os.path.basename(args.export_json))[0]
        export_json(frf, path=om.path(f"{base}.json"))
        om.log_structure()
    elif args.compare:
        om  = OutputManager()
        frf_a = _load_frf_from_npz(args.compare[0])
        frf_b = _load_frf_from_npz(args.compare[1])
        logger.info(f"Comparing: {args.compare[0]} vs {args.compare[1]}")
        plot_comparison(frf_a, frf_b, cfg,
                        save_path=om.path("bandwidth_comparison.png"))
        om.log_structure()
    elif args.demo:
        _run_demo(signal_type=args.signal, noisy=args.noisy)
    else:
        meas = BandwidthMeasurement(cfg, signal_type=args.signal)
        meas.run()
