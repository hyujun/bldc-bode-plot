"""
plotting/noise_plot.py — Three-panel noise analysis figure.

Top    : Error PSD (raw vs filtered) with tonal peak markers
Mid    : Time-domain overlay (raw / filtered / reference)
Bottom : Coherence comparison (raw vs filtered)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy import signal

from ..config import MeasurementConfig, logger
from ..dsp.noise_analyzer import NoiseReport
from .style import (apply_style, style_ax,
                     BG, TXT, DIMTXT, BLUE, RED, GREEN, YELLOW)


def plot_noise_analysis(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    i_meas_filtered: np.ndarray,
    report: NoiseReport,
    cfg: MeasurementConfig,
    save_path: str = "noise_analysis.png",
) -> None:
    apply_style()
    fs = cfg.fs

    fig = plt.figure(figsize=(14, 10), dpi=140)
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        f"Noise Analysis   ·   SNR = {report.snr_db:.1f} dB   |   "
        f"Filters: {', '.join(report.recommended_filters) or 'none'}   |   "
        f"Estimator: {report.recommended_estimator}",
        color=TXT, fontsize=10.5, fontweight="bold", y=0.985,
    )

    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 0.8, 0.8],
        hspace=0.45,
        left=0.07, right=0.97, top=0.94, bottom=0.06,
    )

    # [0] Error PSD
    ax_psd = fig.add_subplot(gs[0])
    style_ax(ax_psd)

    error_raw = i_meas - i_ref
    error_filt = i_meas_filtered - i_ref
    npsd = min(2048, len(error_raw) // 4)

    f_p, P_raw  = signal.welch(error_raw,  fs=fs, nperseg=npsd)
    _,   P_filt = signal.welch(error_filt, fs=fs, nperseg=npsd)
    eps = 1e-12

    ax_psd.semilogy(f_p, P_raw,  color=RED,  lw=1.2, alpha=0.7, label="raw error PSD")
    ax_psd.semilogy(f_p, P_filt, color=GREEN, lw=1.5, zorder=3, label="filtered error PSD")

    for fp in report.tonal_peaks:
        ax_psd.axvline(fp, color=YELLOW, lw=0.8, ls=":", alpha=0.7)
    if report.tonal_peaks:
        ax_psd.scatter(
            report.tonal_peaks,
            [P_raw[np.argmin(np.abs(f_p - fp))] for fp in report.tonal_peaks],
            color=YELLOW, s=25, zorder=5, label="tonal peaks",
        )

    ax_psd.set_xlim([0, fs / 2])
    ax_psd.set_xlabel("Frequency [Hz]")
    ax_psd.set_ylabel("PSD [A²/Hz]")
    ax_psd.set_title("Error Power Spectral Density   (i_meas − i_ref)")
    ax_psd.legend(loc="upper right", handlelength=1.6)

    # [1] Time-domain
    ax_time = fig.add_subplot(gs[1])
    style_ax(ax_time)

    n_show = min(int(0.5 * fs), len(t))
    t_ms = t[:n_show] * 1e3

    ax_time.plot(t_ms, i_ref[:n_show], color=BLUE, lw=1.0, alpha=0.6,
                 label="i_ref")
    ax_time.plot(t_ms, i_meas[:n_show], color=RED, lw=0.8, alpha=0.5,
                 label="i_meas (raw)")
    ax_time.plot(t_ms, i_meas_filtered[:n_show], color=GREEN, lw=1.2,
                 zorder=3, label="i_meas (filtered)")

    ax_time.set_xlabel("Time [ms]")
    ax_time.set_ylabel("Current [A]")
    ax_time.set_title("Time Domain   (first 500 ms)")
    ax_time.legend(loc="upper right", handlelength=1.6, ncol=3)

    # [2] Coherence comparison
    ax_coh = fig.add_subplot(gs[2])
    style_ax(ax_coh)

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
    ax_coh.semilogx(f_c[fmsk], coh_raw[fmsk],  color=RED,   lw=1.0,
                    alpha=0.6, label="raw")
    ax_coh.semilogx(f_c[fmsk], coh_filt[fmsk], color=GREEN, lw=1.5,
                    zorder=3, label="filtered")
    ax_coh.axhline(cfg.coh_threshold, color=YELLOW, lw=1.0, ls="--",
                   label=f"threshold γ² = {cfg.coh_threshold}")

    mean_raw  = float(np.mean(coh_raw[fmsk]))
    mean_filt = float(np.mean(coh_filt[fmsk]))
    ax_coh.text(
        0.98, 0.05,
        f"mean γ²:  raw={mean_raw:.3f}   filtered={mean_filt:.3f}   "
        f"Δ={mean_filt-mean_raw:+.3f}",
        transform=ax_coh.transAxes, color=TXT, fontsize=7.5,
        ha="right", va="bottom", fontfamily="monospace",
    )

    ax_coh.set_xlim([cfg.f_start, cfg.f_end])
    ax_coh.set_ylim([-0.05, 1.08])
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylabel("Coherence γ²")
    ax_coh.set_title("Coherence Improvement")
    ax_coh.legend(loc="lower left", handlelength=1.6)
    ax_coh.xaxis.set_minor_formatter(ticker.NullFormatter())

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    logger.info(f"Noise analysis plot saved → {save_path}")
    plt.show()
