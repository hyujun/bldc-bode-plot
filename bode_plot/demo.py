"""
demo.py — Synthetic second-order plant simulation for testing.

Simulates H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
with wn = 2*pi*180 rad/s, zeta = 0.65
"""

from typing import Optional

import numpy as np
from scipy import signal

from .config import MeasurementConfig, CFG, logger
from .output_manager import OutputManager
from .generators import ChirpGenerator, MultisineGenerator, StepGenerator
from .dsp import (FRFEstimator, NoiseAnalyzer, AdaptivePreprocessor,
                   analyze_step_response)
from .plotting import plot_results, plot_step_results, plot_noise_analysis
from .export import export_csv, export_json


def _simulate_plant(t, i_ref, cfg, rng_seed=42, noisy=False):
    """Pass signal through synthetic 2nd-order plant + noise."""
    wn   = 2 * np.pi * 180.0
    zeta = 0.65
    num  = [wn**2]
    den  = [1, 2*zeta*wn, wn**2]
    sys  = signal.TransferFunction(num, den)
    _, i_meas, _ = signal.lsim(sys, i_ref, t)

    rng = np.random.default_rng(rng_seed)

    if noisy:
        fs = cfg.fs
        i_meas += rng.normal(0, cfg.amplitude * 0.05, size=len(i_meas))

        for f_tone in [47.0, 153.0, 347.0]:
            amp_tone = cfg.amplitude * 0.15
            i_meas += amp_tone * np.sin(2 * np.pi * f_tone * t
                                        + rng.uniform(0, 2*np.pi))

        f_rotor = 23.5
        i_meas += cfg.amplitude * 0.05 * np.sin(2 * np.pi * f_rotor * t)

        n_spikes = max(10, int(len(t) * 0.008))
        spike_idx = rng.choice(len(t), size=n_spikes, replace=False)
        spike_amp = rng.choice([-1, 1], size=n_spikes) * cfg.amplitude * 2.5
        i_meas[spike_idx] += spike_amp
    else:
        i_meas += rng.normal(0, cfg.amplitude * 0.02, size=len(i_meas))

    return i_meas


def _run_demo_single(signal_type: str, cfg: MeasurementConfig,
                     noisy: bool = False,
                     out: Optional[OutputManager] = None) -> Optional[dict]:
    """Run demo for a single signal type."""
    est = FRFEstimator(cfg)
    noise_tag = " (noisy + adaptive)" if noisy else ""
    _p = out.path if out else lambda f: f

    if signal_type == "step":
        logger.info(f"── Demo ── Step excitation{noise_tag}")
        gen = StepGenerator(cfg)
        t, i_ref = gen.get_full_reference()
        i_meas   = _simulate_plant(t, i_ref, cfg, noisy=noisy)

        if noisy:
            analyzer = NoiseAnalyzer(cfg)
            report   = analyzer.analyze(t, i_ref, i_meas)
            preprocessor = AdaptivePreprocessor(cfg)
            i_meas_clean = preprocessor.apply(i_meas, report, i_ref=i_ref)
            plot_noise_analysis(t, i_ref, i_meas, i_meas_clean, report, cfg,
                                save_path=_p("noise_analysis_step.png"))
            i_meas = i_meas_clean

        step_data = analyze_step_response(t, i_ref, i_meas, cfg)
        m   = step_data["metrics"]
        frf = step_data.get("frf")

        if frf is not None:
            logger.info(f"  BW estimate : {frf['bandwidth_hz']:.1f} Hz  (true: ~180 Hz)")
        logger.info(
            f"  Step metrics: t_rise={m['t_rise']*1e3:.2f}ms  "
            f"OS={m['overshoot_pct']:.1f}%  "
            f"t_settle={m['t_settle']*1e3:.2f}ms"
        )

        save_npz = _p("step_response_raw.npz")
        save_dict = dict(
            t=t, i_ref=i_ref, i_meas=i_meas,
            t_step=step_data["t_step"],
            avg=step_data["avg"],
            signal_type=np.array("step"),
        )
        save_dict.update(
            {f"resp_{i}": r for i, r in enumerate(step_data["responses"])}
        )
        save_dict.update(
            {f"metric_{k}": np.array([v]) for k, v in m.items()
             if v is not None}
        )
        if frf is not None:
            save_dict.update(
                f=frf["f"], mag_db=frf["mag_db"],
                phase_deg=frf["phase_deg"],
                coherence=frf["coherence"],
                bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            )
        np.savez(save_npz, **save_dict)
        logger.info(f"  Raw data saved → {save_npz}")
        plot_step_results(t, i_ref, i_meas, step_data, cfg,
                          save_path=_p("step_response_result.png"))
        return step_data["metrics"]

    if signal_type == "multisine":
        logger.info(f"── Demo ── Multisine excitation{noise_tag}")
        gen = MultisineGenerator(cfg)
    else:
        logger.info(f"── Demo ── Chirp excitation{noise_tag}")
        gen = ChirpGenerator(cfg)

    t, i_ref = gen.get_full_reference()
    i_meas   = _simulate_plant(t, i_ref, cfg, noisy=noisy)

    noise_report = None
    if noisy:
        analyzer = NoiseAnalyzer(cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            cfg, save_path=_p(f"noise_analysis_{signal_type}.png"))

        i_meas = i_meas_filtered

    frf = est.estimate(t, i_ref, i_meas, noise_report=noise_report)

    logger.info(f"  BW estimate : {frf['bandwidth_hz']:.1f} Hz  (true: ~180 Hz)")

    save_npz = _p(f"bandwidth_raw_{signal_type}.npz")
    np.savez(
        save_npz,
        t=t, i_ref=i_ref, i_meas=i_meas,
        f=frf["f"], mag_db=frf["mag_db"],
        phase_deg=frf["phase_deg"],
        coherence=frf["coherence"],
        bandwidth_hz=np.array([frf["bandwidth_hz"]]),
        signal_type=np.array(signal_type),
    )
    logger.info(f"  Raw data saved → {save_npz}")
    plot_results(t, i_ref, i_meas, frf, cfg=cfg,
                 save_path=_p(f"bandwidth_result_{signal_type}.png"))
    return frf


def run_demo(signal_type: str = "all", noisy: bool = False) -> None:
    """Run synthetic demo with 2nd-order plant simulation."""
    cfg = CFG
    out = OutputManager()

    if signal_type == "all":
        logger.info("── Demo mode ── sequential: chirp → multisine → step")
        for sig in ("chirp", "multisine", "step"):
            frf = _run_demo_single(sig, cfg, noisy=noisy, out=out)
            if isinstance(frf, dict) and "f" in frf:
                export_csv(frf, path=out.path(f"bandwidth_result_{sig}.csv"))
                export_json(frf, path=out.path(f"bandwidth_result_{sig}.json"))
    else:
        logger.info(f"── Demo mode ── single signal: {signal_type}")
        frf = _run_demo_single(signal_type, cfg, noisy=noisy, out=out)
        if isinstance(frf, dict) and "f" in frf:
            export_csv(frf, path=out.path(f"bandwidth_result_{signal_type}.csv"))
            export_json(frf, path=out.path(f"bandwidth_result_{signal_type}.json"))

    out.log_structure()
