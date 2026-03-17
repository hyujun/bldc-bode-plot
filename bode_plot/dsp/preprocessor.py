"""
dsp/preprocessor.py — Adaptive preprocessing and raw data resampling.

- AdaptivePreprocessor: despike → notch → lowpass filter chain
- preprocess():         raw UDP DataPoints → uniform-sampled arrays
"""

from typing import Optional

import numpy as np
from scipy import signal

from ..config import MeasurementConfig, logger
from ..udp_receiver import DataPoint
from .noise_analyzer import NoiseReport


class AdaptivePreprocessor:
    """
    Apply filter chain selected by NoiseAnalyzer.

    Available filters (applied in order):
      1. median   — interpolate impulse spikes (4-sigma outliers)
      2. notch    — 2nd-order IIR notch at each detected tonal peak
      3. lowpass  — Butterworth LPF at f_end * 1.2
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

        # 1. Spike removal (interpolate outliers)
        if "median" in report.recommended_filters:
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

        # 2. Notch filters (tonal peaks)
        if "notch" in report.recommended_filters and report.tonal_peaks:
            fs = self.cfg.fs
            for f_notch in report.tonal_peaks:
                if f_notch <= 0 or f_notch >= fs / 2.0:
                    continue
                Q = max(30.0, f_notch / 3.0)
                b, a = signal.iirnotch(f_notch, Q, fs)
                x = signal.filtfilt(b, a, x)
                applied.append(f"notch({f_notch:.1f} Hz, Q={Q:.0f})")

        # 3. Lowpass filter (OOB noise removal)
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


def preprocess(
    data: list[DataPoint],
    fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Convert raw UDP DataPoints to uniform-sampled arrays.

    Returns (t_uniform, i_ref, i_meas, fs_detected).
    """
    if len(data) < 100:
        raise ValueError(f"Too few data points: {len(data)}")

    t_r     = np.array([d.t     for d in data])
    ref_r   = np.array([d.i_ref  for d in data])
    meas_r  = np.array([d.i_meas for d in data])

    idx     = np.argsort(t_r)
    t_r, ref_r, meas_r = t_r[idx], ref_r[idx], meas_r[idx]

    _, uniq = np.unique(t_r, return_index=True)
    t_r, ref_r, meas_r = t_r[uniq], ref_r[uniq], meas_r[uniq]

    # Auto-detect sampling frequency from timestamps
    dt_arr = np.diff(t_r)
    dt_arr = dt_arr[dt_arr > 0]
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
