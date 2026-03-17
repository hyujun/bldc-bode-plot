"""
generators.py — Signal generators for bandwidth measurement.

- ChirpGenerator:     logarithmic frequency sweep
- MultisineGenerator: Schroeder-phase sum-of-sines (DFT-bin snapped)
- StepGenerator:      repeated 0 → amplitude step
"""

import numpy as np
from scipy import signal

from .config import MeasurementConfig


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


class MultisineGenerator:
    """
    Sum-of-sines with Schroeder phase optimization.

    x(t) = (A / sqrt(N)) * sum cos(2*pi*f_k*t + phi_k) + dc_bias

    - Frequencies f_k are snapped to DFT bins to eliminate spectral leakage
    - Schroeder phases phi_k = -k(k-1)*pi/N minimize crest factor
    """

    def __init__(self, cfg: MeasurementConfig, n_freqs: int = 60):
        self.cfg     = cfg
        self.n_freqs = n_freqs
        self.freqs   = self._snap_to_dft_bins(cfg, n_freqs)
        self.t_arr   = np.arange(0, cfg.chirp_duration, 1.0 / cfg.fs)
        self.i_ref_arr = self._generate()

    @staticmethod
    def _snap_to_dft_bins(cfg: MeasurementConfig, n_freqs: int) -> np.ndarray:
        df          = cfg.fs / cfg.nperseg
        f_targets   = np.geomspace(cfg.f_start, cfg.f_end, n_freqs)
        bin_indices = np.round(f_targets / df).astype(int)
        bin_indices = np.unique(bin_indices)
        freqs       = bin_indices * df
        freqs = freqs[(freqs >= cfg.f_start) & (freqs <= cfg.f_end)]
        return freqs

    def _generate(self) -> np.ndarray:
        c = self.cfg
        N = len(self.freqs)
        self.n_freqs = N
        k      = np.arange(N)
        phases = -k * (k - 1) * np.pi / N

        t_col = self.t_arr[:, np.newaxis]
        f_row = self.freqs[np.newaxis, :]
        p_row = phases[np.newaxis, :]

        x = (c.amplitude / np.sqrt(N)) * np.sum(
            np.cos(2 * np.pi * f_row * t_col + p_row), axis=1
        ) + c.dc_bias

        return np.clip(x, -c.max_current, c.max_current)

    def get_full_reference(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t_arr.copy(), self.i_ref_arr.copy()


class StepGenerator:
    """
    Generates repeated step signals for direct time-domain analysis.

    Each cycle:
      [0, settle_time)   → dc_bias    (baseline)
      [settle_time, end) → amplitude  (step)
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
