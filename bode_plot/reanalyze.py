"""
reanalyze.py — Re-analyze saved .npz data from a previous measurement.

Loads raw time-domain data (t, i_ref, i_meas) from .npz files in a
specified output folder and re-runs the full analysis pipeline with
current MeasurementConfig parameters.

Usage:
  python -m bode_plot --reanalyze 260317_1430
  python -m bode_plot --reanalyze 260317_1430 --signal chirp
"""

import os
import glob
from typing import Optional

import numpy as np

from .config import MeasurementConfig, CFG, logger
from .output_manager import OutputManager
from .dsp import (FRFEstimator, NoiseAnalyzer, AdaptivePreprocessor,
                   analyze_step_response)
from .plotting import (plot_results, plot_step_results, plot_noise_analysis)
from .export import export_csv, export_json


def _find_npz_files(folder: str) -> dict[str, str]:
    """Discover .npz files in the folder's data/ subdirectory.

    Returns mapping: signal_type → file_path
    """
    data_dir = os.path.join(folder, "data")
    if not os.path.isdir(data_dir):
        data_dir = folder

    found = {}
    for path in glob.glob(os.path.join(data_dir, "*.npz")):
        basename = os.path.basename(path)
        if "bandwidth_raw_chirp" in basename:
            found["chirp"] = path
        elif "bandwidth_raw_multisine" in basename:
            found["multisine"] = path
        elif "step_response_raw" in basename:
            found["step"] = path

    return found


def _reanalyze_frf(npz_path: str, signal_type: str,
                   cfg: MeasurementConfig,
                   out: OutputManager) -> Optional[dict]:
    """Re-analyze a chirp or multisine .npz file."""
    logger.info(f"[reanalyze] Loading {signal_type}: {npz_path}")
    d = np.load(npz_path)

    if "t" not in d or "i_ref" not in d or "i_meas" not in d:
        logger.warning(f"  .npz missing raw time-domain data (t, i_ref, i_meas) — "
                       f"cannot re-analyze, skipping")
        return None

    t      = d["t"]
    i_ref  = d["i_ref"]
    i_meas = d["i_meas"]

    logger.info(f"  Loaded {len(t)} samples, duration={t[-1]-t[0]:.2f}s")

    # Noise analysis
    analyzer = NoiseAnalyzer(cfg)
    noise_report = analyzer.analyze(t, i_ref, i_meas)

    # Adaptive preprocessing
    preprocessor = AdaptivePreprocessor(cfg)
    i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

    plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report, cfg,
                        save_path=out.path(f"noise_analysis_{signal_type}_reanalyzed.png"))

    # FRF estimation
    est = FRFEstimator(cfg)
    frf = est.estimate(t, i_ref, i_meas_filtered, noise_report=noise_report)

    logger.info(f"  ★  [{signal_type}] Bandwidth : {frf['bandwidth_hz']:.1f} Hz  "
                f"(estimator={frf.get('estimator', 'H1')}, "
                f"nperseg={frf.get('nperseg_used', cfg.nperseg)})")

    # Save re-analyzed results
    save_npz = out.path(f"bandwidth_raw_{signal_type}_reanalyzed.npz")
    np.savez(
        save_npz,
        t=t, i_ref=i_ref, i_meas=i_meas_filtered,
        f=frf["f"], mag_db=frf["mag_db"],
        phase_deg=frf["phase_deg"],
        coherence=frf["coherence"],
        bandwidth_hz=np.array([frf["bandwidth_hz"]]),
        signal_type=np.array(signal_type),
    )
    logger.info(f"  Re-analyzed data saved → {save_npz}")

    plot_results(t, i_ref, i_meas_filtered, frf, cfg=cfg,
                 save_path=out.path(f"bandwidth_result_{signal_type}_reanalyzed.png"))

    export_csv(frf, path=out.path(f"bandwidth_result_{signal_type}_reanalyzed.csv"))
    export_json(frf, path=out.path(f"bandwidth_result_{signal_type}_reanalyzed.json"))

    return frf


def _reanalyze_step(npz_path: str, cfg: MeasurementConfig,
                    out: OutputManager) -> Optional[dict]:
    """Re-analyze a step response .npz file."""
    logger.info(f"[reanalyze] Loading step: {npz_path}")
    d = np.load(npz_path)

    if "t" not in d or "i_ref" not in d or "i_meas" not in d:
        logger.warning("  .npz missing raw time-domain data — cannot re-analyze, skipping")
        return None

    t      = d["t"]
    i_ref  = d["i_ref"]
    i_meas = d["i_meas"]

    logger.info(f"  Loaded {len(t)} samples, duration={t[-1]-t[0]:.2f}s")

    # Noise analysis
    analyzer = NoiseAnalyzer(cfg)
    noise_report = analyzer.analyze(t, i_ref, i_meas)

    # Adaptive preprocessing
    preprocessor = AdaptivePreprocessor(cfg)
    i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

    plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report, cfg,
                        save_path=out.path("noise_analysis_step_reanalyzed.png"))

    # Step response analysis
    step_data = analyze_step_response(t, i_ref, i_meas_filtered, cfg)
    m   = step_data["metrics"]
    frf = step_data.get("frf")

    if frf is not None:
        logger.info(f"  ★  [step] Bandwidth   : {frf['bandwidth_hz']:.1f} Hz")
    if m["t_rise"] is not None:
        logger.info(f"  ★  [step] Rise time   : {m['t_rise']*1e3:.2f} ms")
    if m["overshoot_pct"] is not None:
        logger.info(f"  ★  [step] Overshoot   : {m['overshoot_pct']:.1f} %")
    if m["t_settle"] is not None:
        logger.info(f"  ★  [step] Settle time : {m['t_settle']*1e3:.2f} ms")

    # Save
    save_npz = out.path("step_response_raw_reanalyzed.npz")
    save_dict = dict(
        t=t, i_ref=i_ref, i_meas=i_meas_filtered,
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
    logger.info(f"  Re-analyzed data saved → {save_npz}")

    plot_step_results(t, i_ref, i_meas_filtered, step_data, cfg,
                      save_path=out.path("step_response_result_reanalyzed.png"))

    return step_data["metrics"]


def run_reanalyze(folder: str, signal_type: str = "all",
                  cfg: MeasurementConfig = None) -> dict:
    """Load raw .npz data from folder and re-run analysis pipeline.

    Parameters
    ----------
    folder : str
        Path to a YYMMDD_HHMM output directory.
    signal_type : str
        "all", "chirp", "multisine", or "step".
    cfg : MeasurementConfig, optional
        Override configuration (uses global CFG if None).

    Returns
    -------
    dict with results per signal type.
    """
    if cfg is None:
        cfg = CFG

    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory not found: {folder}")

    # Output goes to the same folder
    out = OutputManager.from_existing(folder)

    npz_files = _find_npz_files(folder)

    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in {folder} or {folder}/data/. "
            f"Expected bandwidth_raw_*.npz or step_response_raw.npz"
        )

    logger.info("=" * 58)
    logger.info("Re-analysis mode")
    logger.info(f"  Source : {folder}")
    logger.info(f"  Found  : {', '.join(npz_files.keys())}")
    logger.info(f"  Signal : {signal_type}")
    logger.info("=" * 58)

    results = {}

    if signal_type in ("all", "chirp") and "chirp" in npz_files:
        frf = _reanalyze_frf(npz_files["chirp"], "chirp", cfg, out)
        if frf:
            results["chirp"] = frf

    if signal_type in ("all", "multisine") and "multisine" in npz_files:
        frf = _reanalyze_frf(npz_files["multisine"], "multisine", cfg, out)
        if frf:
            results["multisine"] = frf

    if signal_type in ("all", "step") and "step" in npz_files:
        step_m = _reanalyze_step(npz_files["step"], cfg, out)
        if step_m:
            results["step"] = step_m

    logger.info("=" * 58)
    logger.info("Re-analysis complete")
    out.log_structure()
    logger.info("=" * 58)

    return results
