"""
dsp/frf_estimator.py — Frequency Response Function estimation.

Welch cross/auto PSD → H1 / H2 / Hv estimators, coherence,
bandwidth (-3 dB), and gain margin.
"""

from typing import Optional

import numpy as np
from scipy import signal

from ..config import MeasurementConfig, logger
from .noise_analyzer import NoiseReport


class FRFEstimator:
    def __init__(self, cfg: MeasurementConfig):
        self.cfg = cfg

    def estimate(
        self,
        t: np.ndarray,
        i_ref: np.ndarray,
        i_meas: np.ndarray,
        noise_report: Optional[NoiseReport] = None,
    ) -> dict:
        fs      = self.cfg.fs

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
        Syx     = np.conj(Sxy)

        # H1 estimator (robust to output noise)
        H1         = Sxy / (Sxx + eps)
        # H2 estimator (robust to input noise)
        H2         = Syy / (Syx + eps)

        use_hv = (noise_report is not None
                  and noise_report.recommended_estimator == "Hv")
        if use_hv:
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
        """Find gain margin at -180 deg phase crossover."""
        cfg = self.cfg
        fmsk = valid & (f >= cfg.f_start) & (f <= cfg.f_end)
        for i in range(len(phase_deg) - 1):
            if not fmsk[i]:
                continue
            if phase_deg[i] >= -180.0 >= phase_deg[i+1]:
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
                gain_margin = -mag_at_cross
                return gain_margin, f_cross
        return None, None
