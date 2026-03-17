"""
plotting/comparison_plot.py — Four-panel FRF comparison overlay.

Top-left   : Magnitude overlay
Top-right  : Phase overlay
Bot-left   : Coherence overlay
Bot-right  : Step response overlay
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from ..config import MeasurementConfig, CFG, logger
from ..dsp.step_response import estimate_step_response
from .style import (apply_style, style_ax,
                     BG, TXT, DIMTXT, SPINE, BLUE, RED, GREEN, YELLOW, CYAN)


def load_frf_from_npz(path: str) -> dict:
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
    apply_style()

    fig = plt.figure(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor(BG)

    label_a = frf_a.get("signal_type", "A")
    label_b = frf_b.get("signal_type", "B")
    bw_a = float(frf_a["bandwidth_hz"])
    bw_b = float(frf_b["bandwidth_hz"])

    fig.suptitle(
        f"{label_a} vs {label_b}   ·   "
        f"BW_{label_a} = {bw_a:.1f} Hz   |   BW_{label_b} = {bw_b:.1f} Hz",
        color=TXT, fontsize=10.5, fontweight="bold", y=0.985,
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

    rmask  = valid_a & (frf_a["f"] >= cfg.ref_f_low) & (frf_a["f"] <= cfg.ref_f_high)
    ref_db = float(np.mean(frf_a["mag_db"][rmask])) if np.any(rmask) else 0.0

    # ── [0,0] Magnitude ───────────────────────────────────
    ax_mag = fig.add_subplot(gs[0, 0])
    style_ax(ax_mag)

    ax_mag.semilogx(
        frf_a["f"][fmsk_a], frf_a["mag_db"][fmsk_a],
        color=BLUE, lw=1.7, zorder=3, label=f"{label_a}  (BW={bw_a:.1f} Hz)",
    )
    ax_mag.semilogx(
        frf_b["f"][fmsk_b], frf_b["mag_db"][fmsk_b],
        color=CYAN, lw=1.5, ls="--", zorder=3, label=f"{label_b}  (BW={bw_b:.1f} Hz)",
    )

    ax_mag.axhline(ref_db - 3.0, color=RED, lw=0.9, ls="--", zorder=2,
                   label=f"−3 dB  ({ref_db-3:.1f} dB)")
    ax_mag.axhline(0.0, color=SPINE, lw=0.6, zorder=1)
    ax_mag.axvline(bw_a, color=BLUE,  lw=1.0, ls=":", alpha=0.7, zorder=2)
    ax_mag.axvline(bw_b, color=CYAN,  lw=1.0, ls=":", alpha=0.7, zorder=2)

    y_lo = max(float(min(frf_a["mag_db"][fmsk_a].min(),
                         frf_b["mag_db"][fmsk_b].min())) - 6.0, -55.0)
    ax_mag.set_xlim([cfg.f_start, cfg.f_end])
    ax_mag.set_ylim([y_lo, ref_db + 14.0])
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Bode  —  Magnitude")
    ax_mag.legend(loc="lower left", handlelength=1.6)
    ax_mag.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [0,1] Phase ────────────────────────────────────────
    ax_ph = fig.add_subplot(gs[0, 1])
    style_ax(ax_ph)

    ax_ph.semilogx(
        frf_a["f"][fmsk_a], frf_a["phase_deg"][fmsk_a],
        color=BLUE, lw=1.7, zorder=3, label=label_a,
    )
    ax_ph.semilogx(
        frf_b["f"][fmsk_b], frf_b["phase_deg"][fmsk_b],
        color=CYAN, lw=1.5, ls="--", zorder=3, label=label_b,
    )

    ax_ph.axhline(-90,  color=YELLOW, lw=0.9, ls="--", zorder=2, label="−90°")
    ax_ph.axhline(-180, color=RED,    lw=0.7, ls=":",  zorder=1, alpha=0.6, label="−180°")

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [1,0] Coherence ────────────────────────────────────
    ax_coh = fig.add_subplot(gs[1, 0])
    style_ax(ax_coh)

    ax_coh.semilogx(
        frf_a["f"][fmsk_a], frf_a["coherence"][fmsk_a],
        color=BLUE, lw=1.5, zorder=3, label=label_a,
    )
    ax_coh.semilogx(
        frf_b["f"][fmsk_b], frf_b["coherence"][fmsk_b],
        color=CYAN, lw=1.5, ls="--", zorder=3, label=label_b,
    )

    ax_coh.axhline(cfg.coh_threshold, color=YELLOW, lw=1.0, ls="--", zorder=2,
                   label=f"threshold γ² = {cfg.coh_threshold}")
    ax_coh.set_xlim([cfg.f_start, cfg.f_end])
    ax_coh.set_ylim([-0.05, 1.08])
    ax_coh.set_xlabel("Frequency [Hz]")
    ax_coh.set_ylabel("Coherence γ²")
    ax_coh.set_title("Coherence")
    ax_coh.legend(loc="lower left", handlelength=1.6)
    ax_coh.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [1,1] Step response ────────────────────────────────
    ax_step = fig.add_subplot(gs[1, 1])
    style_ax(ax_step)

    H_a = frf_a["H"]
    H_b = frf_b["H"]
    fv_a = fmsk_a & valid_a
    fv_b = fmsk_b & valid_b

    t_sa, s_sa = estimate_step_response(frf_a["f"][fv_a], H_a[fv_a], cfg.fs)
    t_sb, s_sb = estimate_step_response(frf_b["f"][fv_b], H_b[fv_b], cfg.fs)

    ax_step.plot(t_sa * 1e3, s_sa, color=BLUE, lw=1.7, zorder=3, label=label_a)
    ax_step.plot(t_sb * 1e3, s_sb, color=CYAN, lw=1.5, ls="--", zorder=3, label=label_b)

    ax_step.axhline(1.00, color=DIMTXT, lw=0.8, ls=":", zorder=1)
    ax_step.fill_between(t_sa * 1e3, 0.98, 1.02, color=DIMTXT, alpha=0.08, zorder=1)
    ax_step.axhline(0.0, color=SPINE, lw=0.6, zorder=1)

    y_lo_s = min(float(min(s_sa.min(), s_sb.min())) - 0.08, -0.12)
    y_hi_s = max(float(max(s_sa.max(), s_sb.max())) + 0.12, 1.25)
    ax_step.set_xlim([0.0, max(t_sa[-1], t_sb[-1]) * 1e3])
    ax_step.set_ylim([y_lo_s, y_hi_s])
    ax_step.set_xlabel("Time [ms]")
    ax_step.set_ylabel("Normalized amplitude")
    ax_step.set_title("Step Response  (IFFT estimated)")
    ax_step.legend(loc="lower right", handlelength=1.6)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    logger.info(f"Comparison plot saved → {save_path}")
    plt.show()
