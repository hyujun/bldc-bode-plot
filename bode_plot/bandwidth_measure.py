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
import socket
import re
import threading
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy import signal
from collections import deque
from dataclasses import dataclass
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
    udp_host: str        = "192.168.1.2"
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
# STM32 sends text-based UDP messages:
#   Start : "Bandwidth Measurement Started"
#   Data  : "Chirp: t=0.002, ref=0.300, cur=0.159"  (every 2 ms)
#   End   : "Bandwidth Measurement Done"
_DATA_RE = re.compile(
    r"(?:Chirp|Multisine|Step):\s*t=([\d.]+),\s*ref=([\-\d.]+),\s*cur=([\-\d.]+)"
)

@dataclass
class DataPoint:
    t:      float
    i_ref:  float
    i_meas: float

class UDPReceiver:
    def __init__(self, cfg: MeasurementConfig):
        self.cfg     = cfg
        self.buffer  = deque(maxlen=int(cfg.fs * cfg.chirp_duration * 1.5))
        self._stop   = threading.Event()
        self._started = threading.Event()
        self._done    = threading.Event()
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

    def get_data(self) -> list[DataPoint]:
        with self._lock:
            return list(self.buffer)

    def stats(self) -> dict:
        return {"received": self._rx, "dropped": self._drop}

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind((self.cfg.udp_host, self.cfg.udp_port))
        sock.settimeout(0.5)
        collecting = False
        while not self._stop.is_set():
            try:
                raw, _ = sock.recvfrom(self.cfg.udp_buffer_size)
                msg = raw.decode("utf-8", errors="replace").strip()

                if "Bandwidth Measurement Started" in msg:
                    collecting = True
                    self._started.set()
                    logger.info("Received: Bandwidth Measurement Started")
                    continue

                if "Bandwidth Measurement Done" in msg:
                    collecting = False
                    self._done.set()
                    logger.info("Received: Bandwidth Measurement Done")
                    continue

                if not collecting:
                    continue

                m = _DATA_RE.search(msg)
                if not m:
                    self._drop += 1
                    continue

                t     = float(m.group(1))
                i_ref = float(m.group(2))
                i_meas = float(m.group(3))

                if abs(i_meas) > self.cfg.max_current * 3:
                    self._drop += 1
                    continue
                with self._lock:
                    self.buffer.append(DataPoint(t, i_ref, i_meas))
                self._rx += 1
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"UDP error: {e}")
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

    - Frequencies f_k are logarithmically spaced in [f_start, f_end]
    - Schroeder phases φ_k = −k(k−1)π/N minimize crest factor
    - Duration matches chirp_duration for fair comparison
    """

    def __init__(self, cfg: MeasurementConfig, n_freqs: int = 60):
        self.cfg     = cfg
        self.n_freqs = n_freqs
        self.freqs   = np.geomspace(cfg.f_start, cfg.f_end, n_freqs)
        self.t_arr   = np.arange(0, cfg.chirp_duration, 1.0 / cfg.fs)
        self.i_ref_arr = self._generate()

    def _generate(self) -> np.ndarray:
        c = self.cfg
        N = self.n_freqs
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
    ) -> dict:
        fs      = self.cfg.fs
        nperseg = min(self.cfg.nperseg, len(i_ref) // 4)
        noverlap= min(self.cfg.noverlap, nperseg // 2)
        kw      = dict(fs=fs, window=self.cfg.window,
                       nperseg=nperseg, noverlap=noverlap)
        eps = 1e-12

        f,  Sxx = signal.welch(i_ref,         **kw)
        _,  Sxy = signal.csd(i_ref, i_meas,   **kw)
        _,  Syy = signal.welch(i_meas,        **kw)

        H          = Sxy / (Sxx + eps)
        mag_db     = 20 * np.log10(np.abs(H) + eps)
        phase_deg  = np.degrees(np.unwrap(np.angle(H)))
        coherence  = np.abs(Sxy)**2 / (Sxx * Syy + eps)
        valid      = coherence > self.cfg.coh_threshold

        bw_hz      = self._bandwidth(f, mag_db, valid)

        return dict(
            f=f, H=H, mag_db=mag_db,
            phase_deg=phase_deg, coherence=coherence,
            valid=valid, bandwidth_hz=bw_hz,
        )

    def _bandwidth(
        self,
        f: np.ndarray,
        mag_db: np.ndarray,
        valid: np.ndarray,
    ) -> float:
        cfg  = self.cfg
        eps  = 1e-12
        rmask = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
        ref_db = float(np.mean(mag_db[rmask])) if np.any(rmask) else 0.0

        drop = valid & (mag_db < ref_db - 3.0) & (f > cfg.ref_f_high)
        if not np.any(drop):
            return float(f[valid][-1]) if np.any(valid) else float(f[-1])

        idx = int(np.argmax(drop))
        if idx > 0:
            f0, f1 = float(f[idx-1]), float(f[idx])
            m0, m1 = float(mag_db[idx-1]), float(mag_db[idx])
            return f0 + (f1-f0) * (ref_db-3.0-m0) / (m1-m0+eps)
        return float(f[idx])


# ════════════════════════════════════════════════════════════
# Preprocessing
# ════════════════════════════════════════════════════════════
def preprocess(
    data: list[DataPoint],
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(data) < 100:
        raise ValueError(f"Too few data points: {len(data)}")

    t_r     = np.array([d.t     for d in data])
    ref_r   = np.array([d.i_ref  for d in data])
    meas_r  = np.array([d.i_meas for d in data])

    idx     = np.argsort(t_r)
    t_r, ref_r, meas_r = t_r[idx], ref_r[idx], meas_r[idx]

    _, uniq = np.unique(t_r, return_index=True)
    t_r, ref_r, meas_r = t_r[uniq], ref_r[uniq], meas_r[uniq]

    t_u   = np.arange(t_r[0], t_r[-1], 1.0/fs)
    i_ref  = np.interp(t_u, t_r, ref_r)
    i_meas = np.interp(t_u, t_r, meas_r)

    expected = int((t_r[-1]-t_r[0]) * fs)
    loss_pct = 100.0 * (1 - len(t_r)/max(expected,1))
    logger.info(f"Packet loss {loss_pct:.1f} % ({expected-len(t_r)}/{expected})")

    return t_u, i_ref, i_meas


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

    f0, f1 = float(f_valid[0]), float(f_valid[-1])
    band   = (f_uni >= f0) & (f_uni <= f1)

    H_uni[band] = (
        np.interp(f_uni[band], f_valid, H_valid.real)
        + 1j * np.interp(f_uni[band], f_valid, H_valid.imag)
    )
    # DC: copy lowest valid value
    H_uni[0] = H_uni[band][0] if np.any(band) else 0.0

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

    return dict(
        t_step=t_step,
        responses=responses,
        avg=avg,
        metrics=metrics,
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
    Two-panel step response figure
    ───────────────────────────────
    Top    : Raw time-domain data (i_ref + i_meas)
    Bottom : Ensemble-averaged normalized step response with metrics
    """
    _apply_style()

    t_step   = step_data["t_step"]
    avg      = step_data["avg"]
    resps    = step_data["responses"]
    metrics  = step_data["metrics"]
    t_ms     = t_step * 1e3

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor(_BG)

    # Suptitle
    parts = []
    if metrics["t_rise"] is not None:
        parts.append(f"t_rise = {metrics['t_rise']*1e3:.2f} ms")
    if metrics["overshoot_pct"] is not None:
        parts.append(f"OS = {metrics['overshoot_pct']:.1f} %")
    if metrics["t_settle"] is not None:
        parts.append(f"t_settle = {metrics['t_settle']*1e3:.2f} ms")
    if metrics.get("ss_error_pct") is not None:
        parts.append(f"SS err = {metrics['ss_error_pct']:.2f} %")

    fig.suptitle(
        "BLDC Current Controller   ·   Step Response (measured)   ·   "
        + "   |   ".join(parts),
        color=_TXT, fontsize=10.5, fontweight="bold",
        y=0.985, x=0.5,
    )

    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.8, 1.2],
        hspace=0.38,
        left=0.07, right=0.97, top=0.94, bottom=0.08,
    )

    # ════════════════════════════════════════════════════════
    # [0]  Raw time-domain data
    # ════════════════════════════════════════════════════════
    ax_raw = fig.add_subplot(gs[0])
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
    # [1]  Ensemble-averaged step response
    # ════════════════════════════════════════════════════════
    ax_step = fig.add_subplot(gs[1])
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
    save_path: str = "bandwidth_result.png",
) -> None:
    """
    Three-panel figure
    ──────────────────
    Left-top    : Bode Magnitude  (log-x, dB)
    Right-top   : Bode Phase      (log-x, deg, unwrapped)
    Bottom-full : Step response   (IFFT-estimated, ms)
    """
    _apply_style()

    cfg  = CFG
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

    # Phase margin  (0 dB crossover)
    phase_margin = None
    f_cross      = None
    cross_mask   = valid & fmsk & (f > cfg.ref_f_high)
    for i in range(len(frf["mag_db"]) - 1):
        if cross_mask[i] and frf["mag_db"][i] >= 0 >= frf["mag_db"][i+1]:
            f_cross      = float(f[i])
            phase_margin = 180.0 + float(phase_uw[i])
            break

    # Step response
    fv            = fmsk & valid
    t_step, s_step = _estimate_step_response(f[fv], H[fv], cfg.fs)
    t_ms           = t_step * 1e3
    metrics        = _step_metrics(t_step, s_step)

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor(_BG)

    # Suptitle
    parts = [f"BW = {bw:.1f} Hz"]
    if phase_margin is not None:
        parts.append(f"PM = {phase_margin:.1f}°")
    if metrics["t_rise"] is not None:
        parts.append(f"t_rise = {metrics['t_rise']*1e3:.2f} ms")
    if metrics["overshoot_pct"] is not None:
        parts.append(f"OS = {metrics['overshoot_pct']:.1f} %")
    if metrics["t_settle"] is not None:
        parts.append(f"t_settle = {metrics['t_settle']*1e3:.2f} ms")

    fig.suptitle(
        "BLDC Current Controller   ·   " + "   |   ".join(parts),
        color=_TXT, fontsize=10.5, fontweight="bold",
        y=0.985, x=0.5,
    )

    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 1.15],
        hspace=0.48, wspace=0.28,
        left=0.07, right=0.97,
        top=0.94, bottom=0.08,
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
        color=_BLUE, lw=1.7, zorder=3, label="|H(jω)|",
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

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ════════════════════════════════════════════════════════
    # [1, :]  Step response  (full width)
    # ════════════════════════════════════════════════════════
    ax_step = fig.add_subplot(gs[1, :])
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
# Demo mode  (synthetic second-order system)
# ════════════════════════════════════════════════════════════
def _simulate_plant(t, i_ref, cfg, rng_seed=42):
    """Pass signal through synthetic 2nd-order plant + noise."""
    wn   = 2 * np.pi * 180.0
    zeta = 0.65
    num  = [wn**2]
    den  = [1, 2*zeta*wn, wn**2]
    sys  = signal.TransferFunction(num, den)
    _, i_meas, _ = signal.lsim(sys, i_ref, t)

    rng     = np.random.default_rng(rng_seed)
    i_meas += rng.normal(0, cfg.amplitude * 0.02, size=len(i_meas))
    return i_meas


def _run_demo(signal_type: str = "chirp") -> None:
    """
    Simulate a 2nd-order current controller:
        H(s) = ωn² / (s² + 2ζωn·s + ωn²)
    with ωn = 2π·180 rad/s, ζ = 0.65

    signal_type: "chirp", "multisine", or "step"
    """
    cfg = CFG

    if signal_type == "step":
        logger.info("── Demo mode ── synthetic plant with Step excitation")
        gen = StepGenerator(cfg)
        t, i_ref = gen.get_full_reference()
        i_meas   = _simulate_plant(t, i_ref, cfg)

        step_data = _analyze_step_response(t, i_ref, i_meas, cfg)
        m = step_data["metrics"]
        logger.info(
            f"Demo Step metrics: t_rise={m['t_rise']*1e3:.2f}ms  "
            f"OS={m['overshoot_pct']:.1f}%  "
            f"t_settle={m['t_settle']*1e3:.2f}ms"
        )

        save_npz = "step_response_raw.npz"
        np.savez(
            save_npz,
            t=t, i_ref=i_ref, i_meas=i_meas,
            t_step=step_data["t_step"],
            avg=step_data["avg"],
            signal_type=np.array("step"),
            **{f"resp_{i}": r for i, r in enumerate(step_data["responses"])},
            **{f"metric_{k}": np.array([v]) for k, v in m.items()
               if v is not None},
        )
        logger.info(f"Raw data saved → {save_npz}")

        plot_step_results(t, i_ref, i_meas, step_data, cfg)
        return

    est = FRFEstimator(cfg)

    if signal_type == "multisine":
        logger.info("── Demo mode ── synthetic plant with Multisine excitation")
        gen = MultisineGenerator(cfg)
    else:
        logger.info("── Demo mode ── synthetic plant with Chirp excitation")
        gen = ChirpGenerator(cfg)

    t, i_ref = gen.get_full_reference()
    i_meas   = _simulate_plant(t, i_ref, cfg)
    frf      = est.estimate(t, i_ref, i_meas)

    logger.info(f"Demo BW estimate : {frf['bandwidth_hz']:.1f} Hz  (true: ~180 Hz)")

    save_npz = f"bandwidth_raw_{signal_type}.npz"
    np.savez(
        save_npz,
        t=t, i_ref=i_ref, i_meas=i_meas,
        f=frf["f"], mag_db=frf["mag_db"],
        phase_deg=frf["phase_deg"],
        coherence=frf["coherence"],
        bandwidth_hz=np.array([frf["bandwidth_hz"]]),
        signal_type=np.array(signal_type),
    )
    logger.info(f"Raw data saved → {save_npz}")

    plot_results(t, i_ref, i_meas, frf,
                 save_path=f"bandwidth_result_{signal_type}.png")


# ════════════════════════════════════════════════════════════
# Live measurement
# ════════════════════════════════════════════════════════════
class BandwidthMeasurement:
    def __init__(self, cfg: MeasurementConfig = CFG, signal_type: str = "chirp"):
        self.cfg         = cfg
        self.signal_type = signal_type
        self.receiver    = UDPReceiver(cfg)
        self.estimator   = FRFEstimator(cfg)

    def run(self) -> dict:
        cfg = self.cfg
        is_step = self.signal_type == "step"

        if is_step:
            total_dur = (cfg.step_settle + cfg.step_hold) * cfg.step_repeats
        else:
            total_dur = cfg.chirp_duration

        logger.info("=" * 58)
        logger.info("BLDC Current Controller  Bandwidth Measurement")
        logger.info(f"  Signal {self.signal_type}  "
                    f"{'settle+hold' if is_step else f'{cfg.f_start}–{cfg.f_end} Hz'}  "
                    f"{total_dur:.1f}s  A={cfg.amplitude}A")
        logger.info(f"  UDP    {cfg.udp_host}:{cfg.udp_port}")
        logger.info("=" * 58)

        self.receiver.start()

        logger.info("Waiting for 'Bandwidth Measurement Started' from STM32 …")
        if not self.receiver.wait_for_start(timeout=60.0):
            self.receiver.stop()
            raise TimeoutError("Timed out waiting for start message from STM32")

        logger.info(f"Recording … (timeout {total_dur + 30.0:.0f} s)")
        if not self.receiver.wait_for_done(timeout=total_dur + 30.0):
            logger.warning("Timed out waiting for done message — using collected data")

        self.receiver.stop()
        logger.info(f"UDP stats: {self.receiver.stats()}")

        data = self.receiver.get_data()
        logger.info(f"Collected {len(data)} samples")

        t, i_ref, i_meas = preprocess(data, cfg.fs)

        if is_step:
            step_data = _analyze_step_response(t, i_ref, i_meas, cfg)
            m = step_data["metrics"]

            logger.info("=" * 58)
            if m["t_rise"] is not None:
                logger.info(f"  ★  Rise time   :  {m['t_rise']*1e3:.2f} ms")
            if m["overshoot_pct"] is not None:
                logger.info(f"  ★  Overshoot   :  {m['overshoot_pct']:.1f} %")
            if m["t_settle"] is not None:
                logger.info(f"  ★  Settle time :  {m['t_settle']*1e3:.2f} ms")
            logger.info("=" * 58)

            save_npz = "step_response_raw.npz"
            np.savez(
                save_npz,
                t=t, i_ref=i_ref, i_meas=i_meas,
                t_step=step_data["t_step"],
                avg=step_data["avg"],
                signal_type=np.array("step"),
                **{f"resp_{i}": r for i, r in enumerate(step_data["responses"])},
                **{f"metric_{k}": np.array([v]) for k, v in m.items()
                   if v is not None},
            )
            logger.info(f"Raw data saved → {save_npz}")

            plot_step_results(t, i_ref, i_meas, step_data, cfg)
            return step_data["metrics"]

        frf = self.estimator.estimate(t, i_ref, i_meas)

        logger.info("=" * 58)
        logger.info(f"  ★  Bandwidth  :  {frf['bandwidth_hz']:.1f} Hz")
        logger.info("=" * 58)

        save_npz = f"bandwidth_raw_{self.signal_type}.npz"
        np.savez(
            save_npz,
            t=t, i_ref=i_ref, i_meas=i_meas,
            f=frf["f"], mag_db=frf["mag_db"],
            phase_deg=frf["phase_deg"],
            coherence=frf["coherence"],
            bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            signal_type=np.array(self.signal_type),
        )
        logger.info(f"Raw data saved → {save_npz}")

        plot_results(t, i_ref, i_meas, frf,
                     save_path=f"bandwidth_result_{self.signal_type}.png")
        return frf


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
        "--signal", choices=["chirp", "multisine", "step"],
        default="chirp",
        help="Excitation signal type (default: chirp). "
             "'step' runs time-domain step response test.",
    )
    parser.add_argument(
        "--compare", nargs=2, metavar="NPZ",
        help="Compare two saved .npz result files. "
             "Example: --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz",
    )
    args = parser.parse_args()

    cfg = CFG

    if args.compare:
        frf_a = _load_frf_from_npz(args.compare[0])
        frf_b = _load_frf_from_npz(args.compare[1])
        logger.info(f"Comparing: {args.compare[0]} vs {args.compare[1]}")
        plot_comparison(frf_a, frf_b, cfg)
    elif args.demo:
        _run_demo(signal_type=args.signal)
    else:
        meas = BandwidthMeasurement(cfg, signal_type=args.signal)
        meas.run()
