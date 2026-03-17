"""
plotting/step_plot.py — Four-panel step response figure.

[0,0] Bode Magnitude     [0,1] Bode Phase
[1,:] Raw time-domain data (i_ref + i_meas)
[2,:] Ensemble-averaged normalized step response with metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from ..config import MeasurementConfig, logger
from .style import (apply_style, style_ax,
                     BG, TXT, DIMTXT, SPINE, BLUE, GREEN, YELLOW,
                     PURPLE, CYAN)


def plot_step_results(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    step_data: dict,
    cfg: MeasurementConfig,
    save_path: str = "step_response_result.png",
) -> None:
    apply_style()

    t_step   = step_data["t_step"]
    avg      = step_data["avg"]
    resps    = step_data["responses"]
    metrics  = step_data["metrics"]
    frf      = step_data.get("frf")
    t_ms     = t_step * 1e3

    fig = plt.figure(figsize=(14, 16), dpi=140)
    fig.patch.set_facecolor(BG)

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
        color=TXT, fontsize=10.5, fontweight="bold",
        y=0.985, x=0.5,
    )

    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 0.6, 1.2],
        hspace=0.45, wspace=0.28,
        left=0.07, right=0.97, top=0.96, bottom=0.04,
    )

    # ── [0,0] Bode Magnitude ──────────────────────────────
    ax_mag = fig.add_subplot(gs[0, 0])
    style_ax(ax_mag)

    if frf is not None:
        f     = frf["f"]
        mag   = frf["mag_db"]
        coh   = frf["coherence"]
        valid = frf["valid"]
        bw    = frf["bandwidth_hz"]

        ax_mag.semilogx(f, mag, color=DIMTXT, lw=0.7, alpha=0.45, label="all")
        ax_mag.semilogx(f[valid], mag[valid], color=GREEN, lw=1.5,
                        label=f"γ² > {cfg.coh_threshold:.2f}")

        rmask  = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
        ref_db = float(np.mean(mag[rmask])) if np.any(rmask) else 0.0
        ax_mag.axhline(ref_db - 3.0, color=YELLOW, lw=1.0, ls="--",
                       alpha=0.7, label=f"−3 dB ({ref_db-3:.1f} dB)")

        ax_mag.axvline(bw, color=CYAN, lw=1.0, ls=":", alpha=0.8)
        ax_mag.scatter([bw], [ref_db - 3.0], color=CYAN, s=50, zorder=5)
        ax_mag.text(
            bw, ref_db - 3.0 - 2.0,
            f"BW = {bw:.1f} Hz", color=CYAN,
            fontsize=9, fontweight="bold", ha="center", va="top", zorder=5,
        )

    ax_mag.set_xlim([cfg.f_start, cfg.f_end])
    ax_mag.set_xlabel("Frequency [Hz]")
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.set_title("Bode  —  Magnitude  (from step)")
    ax_mag.legend(loc="lower left", handlelength=1.6)
    ax_mag.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [0,1] Bode Phase ──────────────────────────────────
    ax_ph = fig.add_subplot(gs[0, 1])
    style_ax(ax_ph)

    if frf is not None:
        phase = frf["phase_deg"]
        ax_ph.semilogx(f, phase, color=DIMTXT, lw=0.7, alpha=0.45, label="all")
        ax_ph.semilogx(f[valid], phase[valid], color=PURPLE, lw=1.5,
                       label=f"γ² > {cfg.coh_threshold:.2f}")

        if bw > 0:
            ph_at_bw = float(np.interp(bw, f[valid], phase[valid]))
            pm = 180.0 + ph_at_bw
            ax_ph.axvline(bw, color=CYAN, lw=1.0, ls=":", alpha=0.8)
            ax_ph.scatter([bw], [ph_at_bw], color=CYAN, s=50, zorder=5)
            ax_ph.text(
                bw, ph_at_bw - 8,
                f"PM = {pm:.1f}°", color=CYAN,
                fontsize=9, fontweight="bold", ha="center", va="top", zorder=5,
            )

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase  (from step)")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [1,:] Raw time-domain ──────────────────────────────
    ax_raw = fig.add_subplot(gs[1, :])
    style_ax(ax_raw)

    t_plot = t * 1e3
    ax_raw.plot(t_plot, i_ref,  color=BLUE, lw=1.2, alpha=0.8,
                label="i_ref (target)")
    ax_raw.plot(t_plot, i_meas, color=GREEN, lw=1.0, alpha=0.7,
                label="i_meas (measured)")

    ax_raw.set_xlabel("Time [ms]")
    ax_raw.set_ylabel("Current [A]")
    ax_raw.set_title(
        f"Raw Data   ({cfg.step_repeats} cycles × "
        f"{cfg.step_settle}s settle + {cfg.step_hold}s hold)"
    )
    ax_raw.legend(loc="upper right", handlelength=1.6)

    # ── [2,:] Ensemble-averaged step response ──────────────
    ax_step = fig.add_subplot(gs[2, :])
    style_ax(ax_step)

    for i, r in enumerate(resps):
        t_r = np.arange(len(r)) / cfg.fs * 1e3
        ax_step.plot(t_r, r, color=DIMTXT, lw=0.6, alpha=0.4,
                     label="individual" if i == 0 else None)

    ax_step.plot(t_ms, avg, color=GREEN, lw=2.2, zorder=4,
                 label=f"ensemble avg (N={len(resps)})")

    ax_step.axhline(1.00, color=DIMTXT, lw=0.8, ls=":", zorder=1)
    ax_step.fill_between(t_ms, 0.98, 1.02, color=DIMTXT,
                         alpha=0.08, zorder=1)
    ax_step.plot(t_ms, np.full_like(t_ms, 0.98),
                 color=DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5)
    ax_step.plot(t_ms, np.full_like(t_ms, 1.02),
                 color=DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5,
                 label="±2 % band")
    ax_step.axhline(0.0, color=SPINE, lw=0.6, zorder=1)

    if metrics["t_rise"] is not None:
        ss_val = float(np.mean(avg[int(0.80 * len(avg)):]))
        i_lo   = int(np.argmax(avg >= 0.10 * ss_val))
        i_hi   = int(np.argmax(avg >= 0.90 * ss_val))
        tr_ms  = metrics["t_rise"] * 1e3

        ax_step.axvline(t_ms[i_lo], color=CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.axvline(t_ms[i_hi], color=CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.scatter([t_ms[i_lo], t_ms[i_hi]],
                        [avg[i_lo], avg[i_hi]],
                        color=CYAN, s=22, zorder=5)

        mid_t = (t_ms[i_lo] + t_ms[i_hi]) / 2.0
        y_arr = (avg[i_lo] + avg[i_hi]) / 2.0
        ax_step.annotate(
            "", xy=(t_ms[i_hi], y_arr),
            xytext=(t_ms[i_lo], y_arr),
            arrowprops=dict(arrowstyle="<->", color=CYAN, lw=1.2),
            zorder=5,
        )
        ax_step.text(
            mid_t, y_arr + 0.055,
            f"t_rise = {tr_ms:.2f} ms",
            color=CYAN, fontsize=8.5, ha="center", va="bottom",
            fontweight="bold", zorder=5,
        )

    if metrics["t_settle"] is not None:
        ts_ms = metrics["t_settle"] * 1e3
        ax_step.axvline(ts_ms, color=PURPLE, lw=1.3, ls="--", zorder=2,
                        label=f"t_settle = {ts_ms:.2f} ms")
        ax_step.text(
            ts_ms + (t_ms[-1] - t_ms[0]) * 0.007, 0.08,
            f"t_settle\n{ts_ms:.2f} ms",
            color=PURPLE, fontsize=7.5, va="bottom",
        )

    if metrics["overshoot_pct"] is not None and metrics["overshoot_pct"] > 0.3:
        pk_idx = int(np.argmax(avg))
        os_pct = metrics["overshoot_pct"]
        ax_step.scatter([t_ms[pk_idx]], [avg[pk_idx]],
                        color=YELLOW, s=30, zorder=6)
        ax_step.annotate(
            f"OS = {os_pct:.1f} %",
            xy=(t_ms[pk_idx], avg[pk_idx]),
            xytext=(t_ms[pk_idx] + (t_ms[-1] - t_ms[0]) * 0.03,
                    avg[pk_idx] + 0.04),
            color=YELLOW, fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.9),
            zorder=5,
        )

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

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    logger.info(f"Step response plot saved → {save_path}")
    plt.show()
