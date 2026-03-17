"""
dsp/noise_analyzer.py — Automatic noise characterisation and recommendation.

Pipeline
--------
1. Compute error signal  e(t) = i_meas - i_ref
2. PSD of e(t) → detect tonal peaks (PWM harmonics, rotor ripple)
3. Statistics  → spike count, RMS, SNR
4. Quick coherence scan → mean gamma^2 in measurement band
5. Decision rules → recommend notch, median, nperseg, estimator
"""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter as _medfilt1d

from ..config import MeasurementConfig, logger


@dataclass
class NoiseReport:
    """Results of automatic noise characterisation."""
    snr_db: float
    noise_rms: float
    signal_rms: float
    spike_count: int
    spike_ratio: float
    tonal_peaks: list[float]
    tonal_powers_db: list[float]
    mean_coherence: float
    recommended_filters: list[str]
    recommended_nperseg: int
    recommended_estimator: str


class NoiseAnalyzer:
    """Analyse raw i_meas to characterise noise and recommend preprocessing."""

    _PEAK_PROMINENCE_DB = 10.0
    _SPIKE_SIGMA = 4.0
    _COH_HV_THRESHOLD = 0.85
    _SPIKE_RATIO_THRESHOLD = 0.005
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

        # 1. Error signal
        error = i_meas - i_ref

        # 2. Signal / noise RMS
        signal_rms = float(np.sqrt(np.mean(i_ref ** 2)))
        noise_rms  = float(np.sqrt(np.mean(error ** 2)))
        eps = 1e-12
        snr_db = float(20.0 * np.log10(signal_rms / (noise_rms + eps)))

        # 3. Spike detection
        sigma  = float(np.std(error))
        spikes = np.abs(error) > self._SPIKE_SIGMA * sigma
        spike_count = int(np.sum(spikes))
        spike_ratio = spike_count / max(len(error), 1)

        # 4. PSD of error → tonal peak detection
        nperseg_psd = min(2048, len(error) // 4)
        f_psd, Pee = signal.welch(error, fs=fs, nperseg=nperseg_psd)

        Pee_db = 10.0 * np.log10(Pee + eps)

        win = max(15, nperseg_psd // 32)
        if win % 2 == 0:
            win += 1
        Pee_median = _medfilt1d(Pee_db, size=win)

        prominence = Pee_db - Pee_median
        peak_idx, _ = signal.find_peaks(
            prominence,
            height=self._PEAK_PROMINENCE_DB,
            distance=max(3, int(5.0 / (fs / nperseg_psd))),
        )

        valid_peaks = [
            i for i in peak_idx
            if f_psd[i] > cfg.ref_f_high and f_psd[i] < fs / 2.0
        ]
        tonal_peaks     = [float(f_psd[i]) for i in valid_peaks]
        tonal_powers_db = [float(Pee_db[i]) for i in valid_peaks]

        # 5. Quick coherence estimate
        kw = dict(fs=fs, nperseg=nperseg_psd, noverlap=nperseg_psd // 2,
                  window="hann")
        f_c, Sxx = signal.welch(i_ref, **kw)
        _,   Sxy = signal.csd(i_ref, i_meas, **kw)
        _,   Syy = signal.welch(i_meas, **kw)
        coh = np.clip(np.abs(Sxy)**2 / (Sxx * Syy + eps), 0.0, 1.0)

        band = (f_c >= cfg.f_start) & (f_c <= cfg.f_end)
        mean_coh = float(np.mean(coh[band])) if np.any(band) else 0.0

        # 6. Decision rules
        filters = []

        if spike_ratio > self._SPIKE_RATIO_THRESHOLD:
            filters.append("median")

        if tonal_peaks and mean_coh > 0.90:
            filters.append("notch")

        if snr_db < self._LOW_SNR_DB:
            filters.append("lowpass")

        if snr_db < 10.0:
            rec_nperseg = max(512, cfg.nperseg // 2)
        else:
            rec_nperseg = cfg.nperseg

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
