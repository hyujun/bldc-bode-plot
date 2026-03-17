"""
plotting/bode_plot.py — Four-panel Bode + time-domain + step response figure.

[0,0] Bode Magnitude  (log-x, dB)
[0,1] Bode Phase      (log-x, deg, unwrapped)
[1,:] Time-domain ref vs measured (overview)
[2,:] Time-domain ref vs measured (zoomed)
[3,:] Step response   (IFFT-estimated, ms)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from ..config import MeasurementConfig, CFG, logger
from ..dsp.step_response import estimate_step_response, step_metrics
from .style import (apply_style, style_ax,
                     BG, TXT, DIMTXT, SPINE, BLUE, RED, GREEN, YELLOW,
                     PURPLE, CYAN)


def plot_results(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    frf: dict,
    cfg: MeasurementConfig = None,
    save_path: str = "bandwidth_result.png",
) -> None:
    apply_style()

    if cfg is None:
        cfg = CFG
    f    = frf["f"]
    bw   = float(frf["bandwidth_hz"])
    H    = frf.get("H")

    if H is None:
        mag_lin = 10 ** (frf["mag_db"] / 20.0)
        H = mag_lin * np.exp(1j * np.radians(frf["phase_deg"]))

    valid = frf.get("valid", frf["coherence"] > cfg.coh_threshold)
    fmsk  = (f >= cfg.f_start) & (f <= cfg.f_end)

    rmask  = valid & (f >= cfg.ref_f_low) & (f <= cfg.ref_f_high)
    ref_db = float(np.mean(frf["mag_db"][rmask])) if np.any(rmask) else 0.0

    phase_uw = frf["phase_deg"]

    # Phase margin
    phase_margin = None
    f_cross      = None
    cross_mask   = valid & fmsk & (f > cfg.ref_f_high)
    for i in range(len(frf["mag_db"]) - 1):
        if cross_mask[i] and frf["mag_db"][i] >= ref_db >= frf["mag_db"][i+1]:
            f_cross      = float(f[i])
            phase_margin = 180.0 + float(phase_uw[i])
            break

    gain_margin = frf.get("gain_margin_db")
    f_gm        = frf.get("f_gain_margin")

    # Step response
    fv = fmsk & valid
    if not np.any(fv):
        logger.warning("No valid coherence data — using all frequency data for step response")
        fv = fmsk
    t_step, s_step = estimate_step_response(f[fv], H[fv], cfg.fs)
    t_ms           = t_step * 1e3
    metrics        = step_metrics(t_step, s_step)

    # ── Figure ──────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 16), dpi=140)
    fig.patch.set_facecolor(BG)

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
        color=TXT, fontsize=10.5, fontweight="bold",
        y=0.99, x=0.5,
    )

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.0, 0.6, 0.6, 1.0],
        hspace=0.50, wspace=0.28,
        left=0.07, right=0.97,
        top=0.96, bottom=0.04,
    )

    # ── [0,0] Magnitude ────────────────────────────────────
    ax_mag = fig.add_subplot(gs[0, 0])
    style_ax(ax_mag)

    low_coh = ~valid & fmsk
    if np.any(low_coh):
        ax_mag.fill_between(
            f[fmsk], -65, ref_db + 15,
            where=low_coh[fmsk], color=YELLOW, alpha=0.06,
        )
        ax_mag.text(
            cfg.f_end * 0.92, ref_db + 10,
            "low γ²", color=YELLOW, fontsize=6.5,
            ha="right", va="top", alpha=0.7,
        )

    ax_mag.semilogx(
        f[fmsk], frf["mag_db"][fmsk],
        color=BLUE, lw=1.7, zorder=3, label="|H₁(jω)|",
    )

    H2 = frf.get("H2")
    if H2 is not None:
        eps = 1e-12
        mag_db_h2 = 20 * np.log10(np.abs(H2) + eps)
        ax_mag.semilogx(
            f[fmsk], mag_db_h2[fmsk],
            color=CYAN, lw=1.0, ls=":", alpha=0.6, zorder=2,
            label="|H₂(jω)|",
        )

    ax_mag.axhline(ref_db,       color=DIMTXT, lw=0.7, ls=":",  zorder=1)
    ax_mag.axhline(ref_db - 3.0, color=RED,    lw=0.9, ls="--", zorder=2,
                   label=f"−3 dB  ({ref_db-3:.1f} dB)")
    ax_mag.axhline(0.0,          color=SPINE,  lw=0.6, zorder=1)

    ax_mag.axvline(bw, color=YELLOW, lw=1.4, ls="--", zorder=2,
                   label=f"BW = {bw:.1f} Hz")

    y_ann = ref_db - 3.0
    ax_mag.annotate(
        f"{bw:.1f} Hz", xy=(bw, y_ann),
        xytext=(bw * 1.55, y_ann + 6.0),
        color=YELLOW, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.9,
                        connectionstyle="arc3,rad=-0.2"),
        zorder=5,
    )
    ax_mag.scatter([bw], [y_ann], color=YELLOW, s=25, zorder=6)

    if f_gm is not None and gain_margin is not None:
        mag_at_gm = float(frf["mag_db"][np.argmin(np.abs(f - f_gm))])
        ax_mag.axvline(f_gm, color=GREEN, lw=1.0, ls=":", alpha=0.7, zorder=2)
        ax_mag.scatter([f_gm], [mag_at_gm], color=GREEN, s=25, zorder=6)
        ax_mag.annotate(
            f"GM = {gain_margin:.1f} dB\n@ {f_gm:.1f} Hz",
            xy=(f_gm, mag_at_gm),
            xytext=(f_gm * 0.45, mag_at_gm + 5.0),
            color=GREEN, fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9,
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

    # ── [0,1] Phase ────────────────────────────────────────
    ax_ph = fig.add_subplot(gs[0, 1])
    style_ax(ax_ph)

    ax_ph.semilogx(
        f[fmsk], phase_uw[fmsk],
        color=RED, lw=1.7, zorder=3, label="∠H(jω)",
    )

    ax_ph.axhline( -45,  color=SPINE,  lw=0.6, ls=":",  zorder=1)
    ax_ph.axhline( -90,  color=YELLOW, lw=0.9, ls="--", zorder=2, label="−90°")
    ax_ph.axhline(-135,  color=SPINE,  lw=0.6, ls=":",  zorder=1)
    ax_ph.axhline(-180,  color=RED,    lw=0.7, ls=":",  zorder=1,
                  alpha=0.6, label="−180°")

    if f_cross is not None and phase_margin is not None:
        ph_at_cross = float(phase_uw[np.argmin(np.abs(f - f_cross))])
        ax_ph.axvline(f_cross, color=PURPLE, lw=1.2, ls=":", zorder=2,
                      label=f"f₀dB = {f_cross:.1f} Hz")
        ax_ph.scatter([f_cross], [ph_at_cross],
                      color=PURPLE, s=25, zorder=6)
        ax_ph.annotate(
            f"PM = {phase_margin:.1f}°",
            xy=(f_cross, ph_at_cross),
            xytext=(f_cross * 0.55, ph_at_cross + 22),
            color=PURPLE, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=PURPLE, lw=0.9,
                            connectionstyle="arc3,rad=0.2"),
            zorder=5,
        )

    if f_gm is not None:
        ax_ph.axvline(f_gm, color=GREEN, lw=1.0, ls=":", alpha=0.7, zorder=2,
                      label=f"f_{{−180°}} = {f_gm:.1f} Hz")
        ax_ph.scatter([f_gm], [-180.0], color=GREEN, s=25, zorder=6)

    ax_ph.set_xlim([cfg.f_start, cfg.f_end])
    ax_ph.set_xlabel("Frequency [Hz]")
    ax_ph.set_ylabel("Phase [deg]")
    ax_ph.set_title("Bode  —  Phase")
    ax_ph.legend(loc="lower left", handlelength=1.6)
    ax_ph.xaxis.set_minor_formatter(ticker.NullFormatter())

    # ── [1,:] Time-domain overview ─────────────────────────
    ax_td = fig.add_subplot(gs[1, :])
    style_ax(ax_td)

    t_plot = t * 1e3
    ax_td.plot(t_plot, i_ref,  color=BLUE,  lw=0.8, alpha=0.7,
               label="i_ref (reference)")
    ax_td.plot(t_plot, i_meas, color=GREEN, lw=0.6, alpha=0.6,
               label="i_meas (measured)")

    t_total = t_plot[-1] - t_plot[0]
    zoom_center = t_plot[0] + t_total * 0.5
    zoom_half   = t_total * 0.03
    zoom_lo     = zoom_center - zoom_half
    zoom_hi     = zoom_center + zoom_half
    ax_td.axvspan(zoom_lo, zoom_hi, color=CYAN, alpha=0.12, zorder=1)
    ax_td.text(zoom_center, ax_td.get_ylim()[0] if ax_td.get_ylim()[0] != 0 else 0,
               "▼ zoom", color=CYAN, fontsize=7, ha="center", va="bottom",
               alpha=0.8)

    ax_td.set_xlabel("Time [ms]")
    ax_td.set_ylabel("Current [A]")
    ax_td.set_title("Time Domain  —  Reference vs Measured  (overview)")
    ax_td.legend(loc="upper right", handlelength=1.6)

    # ── [2,:] Time-domain zoomed ───────────────────────────
    ax_zoom = fig.add_subplot(gs[2, :])
    style_ax(ax_zoom)

    zmask = (t_plot >= zoom_lo) & (t_plot <= zoom_hi)
    ax_zoom.plot(t_plot[zmask], i_ref[zmask],  color=BLUE,  lw=1.5, alpha=0.9,
                 label="i_ref")
    ax_zoom.plot(t_plot[zmask], i_meas[zmask], color=GREEN, lw=1.2, alpha=0.8,
                 label="i_meas")

    ax_zoom.set_xlim([zoom_lo, zoom_hi])
    ax_zoom.set_xlabel("Time [ms]")
    ax_zoom.set_ylabel("Current [A]")
    ax_zoom.set_title(
        f"Time Domain  —  Zoomed  [{zoom_lo:.1f} – {zoom_hi:.1f} ms]"
    )
    ax_zoom.legend(loc="upper right", handlelength=1.6)

    # ── [3,:] Step response ────────────────────────────────
    ax_step = fig.add_subplot(gs[3, :])
    style_ax(ax_step)

    ax_step.plot(t_ms, s_step, color=GREEN, lw=2.0, zorder=3,
                 label="step response (IFFT estimated)")

    ax_step.axhline(1.00, color=DIMTXT, lw=0.8, ls=":",  zorder=1)
    ax_step.fill_between(t_ms, 0.98, 1.02, color=DIMTXT,
                         alpha=0.08, zorder=1)
    ax_step.plot(t_ms, np.full_like(t_ms, 0.98),
                 color=DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5)
    ax_step.plot(t_ms, np.full_like(t_ms, 1.02),
                 color=DIMTXT, lw=0.5, ls="--", zorder=1, alpha=0.5,
                 label="±2 % band")
    ax_step.axhline(0.0, color=SPINE, lw=0.6, zorder=1)

    if metrics["t_rise"] is not None:
        ss_val = float(np.mean(s_step[int(0.80*len(s_step)):]))
        i_lo   = int(np.argmax(s_step >= 0.10 * ss_val))
        i_hi   = int(np.argmax(s_step >= 0.90 * ss_val))
        tr_ms  = metrics["t_rise"] * 1e3

        ax_step.axvline(t_ms[i_lo], color=CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.axvline(t_ms[i_hi], color=CYAN, lw=0.8, ls=":", alpha=0.7)
        ax_step.scatter([t_ms[i_lo], t_ms[i_hi]],
                        [s_step[i_lo], s_step[i_hi]],
                        color=CYAN, s=22, zorder=5)

        mid_t  = (t_ms[i_lo] + t_ms[i_hi]) / 2.0
        y_arr  = (s_step[i_lo] + s_step[i_hi]) / 2.0
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
        pk_idx  = int(np.argmax(s_step))
        os_pct  = metrics["overshoot_pct"]
        ax_step.scatter([t_ms[pk_idx]], [s_step[pk_idx]],
                        color=YELLOW, s=30, zorder=6)
        ax_step.annotate(
            f"OS = {os_pct:.1f} %",
            xy=(t_ms[pk_idx], s_step[pk_idx]),
            xytext=(t_ms[pk_idx] + (t_ms[-1]-t_ms[0])*0.03,
                    s_step[pk_idx] + 0.04),
            color=YELLOW, fontsize=8.5, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=YELLOW, lw=0.9),
            zorder=5,
        )

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

    ax_step.text(
        0.995, 0.018,
        "⚠  Estimated under small-signal linearity assumption  —  "
        "validate with hardware step test (TEST_SIG_STEP)",
        transform=ax_step.transAxes,
        color=DIMTXT, fontsize=6.5, ha="right", va="bottom", style="italic",
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    logger.info(f"Plot saved → {save_path}")
    plt.show()
