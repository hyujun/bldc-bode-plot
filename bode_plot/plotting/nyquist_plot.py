"""
plotting/nyquist_plot.py — Nyquist diagram of H(jw).
"""

import numpy as np
import matplotlib.pyplot as plt

from ..config import MeasurementConfig, CFG, logger
from .style import (apply_style, style_ax,
                     BG, TXT, DIMTXT, SPINE, BLUE, RED, GREEN, YELLOW)


def plot_nyquist(
    frf: dict,
    cfg: MeasurementConfig = None,
    save_path: str = "nyquist_result.png",
) -> None:
    apply_style()
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
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta),
            color=DIMTXT, lw=0.8, ls="--", alpha=0.5, label="unit circle")

    # Nyquist curve
    ax.plot(H_plot.real, H_plot.imag,
            color=BLUE, lw=1.8, zorder=3, label="H(jω)")
    ax.plot(H_plot.real, -H_plot.imag,
            color=BLUE, lw=1.0, ls=":", alpha=0.4, zorder=2,
            label="H(−jω)")

    # Critical point
    ax.scatter([-1], [0], color=RED, s=60, zorder=6, marker="x", linewidths=2)
    ax.annotate("(−1, 0)", xy=(-1, 0), xytext=(-1.15, 0.15),
                color=RED, fontsize=8, fontweight="bold", zorder=5)

    # Frequency markers
    n_markers = 8
    marker_idx = np.linspace(0, len(H_plot) - 1, n_markers, dtype=int)
    for mi in marker_idx:
        ax.scatter([H_plot[mi].real], [H_plot[mi].imag],
                   color=YELLOW, s=18, zorder=5)
        ax.annotate(
            f"{f_plot[mi]:.0f}",
            xy=(H_plot[mi].real, H_plot[mi].imag),
            xytext=(5, 5), textcoords="offset points",
            color=YELLOW, fontsize=6.5, zorder=5,
        )

    ax.scatter([H_plot[0].real], [H_plot[0].imag],
               color=GREEN, s=40, zorder=6, marker="o",
               label=f"f_start = {f_plot[0]:.0f} Hz")
    ax.scatter([H_plot[-1].real], [H_plot[-1].imag],
               color=RED, s=40, zorder=6, marker="s",
               label=f"f_end = {f_plot[-1]:.0f} Hz")

    ax.axhline(0, color=SPINE, lw=0.6)
    ax.axvline(0, color=SPINE, lw=0.6)
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
        transform=ax.transAxes, color=TXT, fontsize=8.5,
        ha="right", va="top", fontweight="bold",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    logger.info(f"Nyquist plot saved → {save_path}")
    plt.show()
