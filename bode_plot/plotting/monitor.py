"""
plotting/monitor.py — Real-time UDP monitor with scrolling plot.
"""

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..config import MeasurementConfig, CFG, logger
from ..udp_receiver import UDPReceiver
from .style import (apply_style, style_ax,
                     BG, DIMTXT, BLUE, GREEN, RED)


def run_monitor(cfg: MeasurementConfig = None, window_sec: float = 5.0) -> None:
    """Live scrolling plot of UDP i_ref / i_meas.  Press Ctrl+C to stop."""
    if cfg is None:
        cfg = CFG

    apply_style()
    fig, (ax_sig, ax_err) = plt.subplots(2, 1, figsize=(12, 6), dpi=100,
                                          gridspec_kw={"height_ratios": [2, 1],
                                                       "hspace": 0.35})
    fig.patch.set_facecolor(BG)
    style_ax(ax_sig)
    style_ax(ax_err)

    max_pts   = int(window_sec * cfg.fs)
    t_buf     = deque(maxlen=max_pts)
    ref_buf   = deque(maxlen=max_pts)
    meas_buf  = deque(maxlen=max_pts)

    receiver = UDPReceiver(cfg)
    receiver.start()
    logger.info(f"Monitor started  ({window_sec}s window)  —  Ctrl+C to stop")

    line_ref,  = ax_sig.plot([], [], color=BLUE,  lw=1.2, label="i_ref")
    line_meas, = ax_sig.plot([], [], color=GREEN, lw=1.0, label="i_meas")
    line_err,  = ax_err.plot([], [], color=RED,   lw=1.0, label="error")

    ax_sig.set_ylabel("Current [A]")
    ax_sig.set_title("Real-Time UDP Monitor")
    ax_sig.legend(loc="upper right", handlelength=1.2)

    ax_err.set_xlabel("Time [s]")
    ax_err.set_ylabel("Error [A]")
    ax_err.legend(loc="upper right", handlelength=1.2)

    pkt_text = ax_sig.text(
        0.01, 0.97, "", transform=ax_sig.transAxes,
        color=DIMTXT, fontsize=7.5, va="top", fontfamily="monospace",
    )

    def _update(frame):
        data = receiver.get_data()
        if not data:
            return line_ref, line_meas, line_err, pkt_text

        for dp in data:
            t_buf.append(dp.t)
            ref_buf.append(dp.i_ref)
            meas_buf.append(dp.i_meas)

        with receiver._lock:
            receiver.buffer.clear()

        t_arr   = np.array(t_buf)
        ref_arr = np.array(ref_buf)
        meas_arr= np.array(meas_buf)

        line_ref.set_data(t_arr, ref_arr)
        line_meas.set_data(t_arr, meas_arr)
        line_err.set_data(t_arr, ref_arr - meas_arr)

        if len(t_arr) > 1:
            t_lo, t_hi = t_arr[0], t_arr[-1]
            ax_sig.set_xlim(t_lo, t_hi)
            ax_err.set_xlim(t_lo, t_hi)

            y_max = max(abs(ref_arr).max(), abs(meas_arr).max(), 0.01) * 1.1
            ax_sig.set_ylim(-y_max, y_max)

            e_max = max(abs(ref_arr - meas_arr).max(), 0.001) * 1.3
            ax_err.set_ylim(-e_max, e_max)

        stats = receiver.stats()
        pkt_text.set_text(
            f"rx={stats['received']}  drop={stats['dropped']}  "
            f"buf={len(t_buf)}"
        )

        return line_ref, line_meas, line_err, pkt_text

    ani = FuncAnimation(fig, _update, interval=50, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        receiver.stop()
