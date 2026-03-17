"""
dsp/step_response.py — Step response estimation and analysis.

- estimate_step_response(): H(jw) → IFFT → impulse h(t) → cumsum → step s(t)
- step_metrics():           rise time, overshoot, settling time
- analyze_step_response():  extract repeated steps, ensemble average, compute metrics
"""

import numpy as np

from ..config import MeasurementConfig, logger
from .frf_estimator import FRFEstimator


def estimate_step_response(
    f_valid: np.ndarray,
    H_valid: np.ndarray,
    fs: float,
    n_pts: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """
    H(jw) on valid band → IFFT → impulse h(t) → cumsum*dt = step s(t)

    Steps:
      1. Interpolate H onto uniform grid [0, fs/2]
      2. Hermitian extension → irfft → h(t)
      3. cumsum * dt → s(t), normalize to unit steady-state
    """
    f_uni  = np.linspace(0, fs / 2.0, n_pts // 2 + 1)
    H_uni  = np.zeros(len(f_uni), dtype=complex)

    if len(f_valid) == 0:
        t_step = np.arange(n_pts) * (1.0 / fs)
        crop   = min(int(0.025 * fs) + 1, n_pts)
        return t_step[:crop], np.zeros(crop)

    f0, f1 = float(f_valid[0]), float(f_valid[-1])
    band   = (f_uni >= f0) & (f_uni <= f1)

    H_uni[band] = (
        np.interp(f_uni[band], f_valid, H_valid.real)
        + 1j * np.interp(f_uni[band], f_valid, H_valid.imag)
    )
    # DC: unity gain
    H_uni[0] = 1.0 + 0j

    # Smooth edges to reduce Gibbs ringing
    ramp_len = max(1, int(0.05 * np.sum(band)))
    band_idx = np.where(band)[0]
    if len(band_idx) > ramp_len * 2:
        ramp   = np.linspace(0, 1, ramp_len)
        H_uni[band_idx[:ramp_len]]  *= ramp
        H_uni[band_idx[-ramp_len:]] *= ramp[::-1]

    h      = np.fft.irfft(H_uni, n=n_pts)
    dt     = 1.0 / fs
    s      = np.cumsum(h) * dt

    ss     = float(np.mean(s[int(0.80*len(s)):]))
    s_norm = s / ss if abs(ss) > 1e-9 else s

    t_step = np.arange(n_pts) * dt
    crop   = min(int(0.025 * fs) + 1, n_pts)
    return t_step[:crop], s_norm[:crop]


def step_metrics(t: np.ndarray, s: np.ndarray) -> dict:
    """Compute rise time, overshoot, settling time from step response."""
    m = dict(t_rise=None, overshoot_pct=None, t_settle=None)
    if len(s) < 10:
        return m

    ss  = float(np.mean(s[int(0.80*len(s)):]))
    pk  = float(s.max())

    lo_val, hi_val = 0.10 * ss, 0.90 * ss
    i_lo = int(np.argmax(s >= lo_val)) if np.any(s >= lo_val) else None
    i_hi = int(np.argmax(s >= hi_val)) if np.any(s >= hi_val) else None
    if i_lo is not None and i_hi is not None and i_hi > i_lo:
        m["t_rise"] = float(t[i_hi] - t[i_lo])

    if ss > 1e-9:
        m["overshoot_pct"] = max(0.0, (pk/ss - 1.0) * 100.0)

    tol     = 0.02 * abs(ss)
    outside = np.where(np.abs(s - ss) > tol)[0]
    if len(outside):
        m["t_settle"] = float(t[outside[-1]])

    return m


def analyze_step_response(
    t: np.ndarray,
    i_ref: np.ndarray,
    i_meas: np.ndarray,
    cfg: MeasurementConfig,
) -> dict:
    """
    Extract individual step responses from repeated step signal,
    ensemble-average them, and compute metrics.

    Returns dict with keys:
        t_step, responses, avg, metrics, frf
    """
    cycle_samples  = int((cfg.step_settle + cfg.step_hold) * cfg.fs)
    settle_samples = int(cfg.step_settle * cfg.fs)
    hold_samples   = int(cfg.step_hold * cfg.fs)

    responses = []
    for i in range(cfg.step_repeats):
        start = i * cycle_samples + settle_samples
        end   = start + hold_samples
        if end > len(i_meas):
            break

        base_start = i * cycle_samples + int(settle_samples * 0.8)
        base_end   = i * cycle_samples + settle_samples
        baseline   = float(np.mean(i_meas[base_start:base_end]))

        resp = i_meas[start:end]
        target = cfg.amplitude - baseline

        if abs(target) > 1e-9:
            resp_norm = (resp - baseline) / target
        else:
            resp_norm = resp - baseline

        responses.append(resp_norm)

    if not responses:
        raise ValueError("No valid step responses found")

    min_len = min(len(r) for r in responses)
    responses = [r[:min_len] for r in responses]
    avg = np.mean(responses, axis=0)

    t_step = np.arange(min_len) / cfg.fs

    metrics = step_metrics(t_step, avg)

    # Steady-state error (last 20%)
    ss_val = float(np.mean(avg[int(0.8 * len(avg)):]))
    metrics["ss_error_pct"] = abs(1.0 - ss_val) * 100.0

    # FRF estimation from step data
    est = FRFEstimator(cfg)
    frf = est.estimate(t, i_ref, i_meas)

    return dict(
        t_step=t_step,
        responses=responses,
        avg=avg,
        metrics=metrics,
        frf=frf,
    )
