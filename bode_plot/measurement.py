"""
measurement.py — Live bandwidth measurement orchestrator.

Manages the full UDP-based measurement session:
  chirp → multisine → step (sequential)
"""

from typing import Optional

import numpy as np

from .config import MeasurementConfig, CFG, logger
from .output_manager import OutputManager
from .udp_receiver import UDPReceiver
from .dsp import (FRFEstimator, NoiseAnalyzer, AdaptivePreprocessor,
                   preprocess, analyze_step_response)
from .plotting import plot_results, plot_step_results, plot_noise_analysis
from .export import export_csv, export_json


class BandwidthMeasurement:
    """Full bandwidth measurement: chirp → multisine → step (sequential).

    The STM32 sends all three signal types in a single session.
    Each phase is analysed independently and saved to separate files.
    """

    def __init__(self, cfg: MeasurementConfig = CFG, signal_type: str = "all"):
        self.cfg         = cfg
        self.signal_type = signal_type
        self.receiver    = UDPReceiver(cfg)
        self.estimator   = FRFEstimator(cfg)
        self.out         = OutputManager()

    def _analyze_frf_phase(self, phase: str, data: list) -> Optional[dict]:
        """Preprocess + FRF estimation for a chirp/multisine phase."""
        if not data:
            logger.warning(f"[{phase}] no data collected — skipping")
            return None

        logger.info(f"[{phase}] collected {len(data)} samples")
        t, i_ref, i_meas, fs_det = preprocess(data, self.cfg.fs)
        self.cfg.fs = fs_det

        analyzer = NoiseAnalyzer(self.cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(self.cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            self.cfg,
                            save_path=self.out.path(f"noise_analysis_{phase}.png"))

        i_meas = i_meas_filtered

        frf = self.estimator.estimate(t, i_ref, i_meas,
                                      noise_report=noise_report)
        logger.info(f"  ★  [{phase}] Bandwidth : {frf['bandwidth_hz']:.1f} Hz  "
                    f"(estimator={frf.get('estimator', 'H1')}, "
                    f"nperseg={frf.get('nperseg_used', self.cfg.nperseg)})")

        save_npz = self.out.path(f"bandwidth_raw_{phase}.npz")
        np.savez(
            save_npz,
            t=t, i_ref=i_ref, i_meas=i_meas,
            f=frf["f"], mag_db=frf["mag_db"],
            phase_deg=frf["phase_deg"],
            coherence=frf["coherence"],
            bandwidth_hz=np.array([frf["bandwidth_hz"]]),
            signal_type=np.array(phase),
        )
        logger.info(f"  Raw data saved → {save_npz}")

        plot_results(t, i_ref, i_meas, frf, cfg=self.cfg,
                     save_path=self.out.path(f"bandwidth_result_{phase}.png"))

        export_csv(frf, path=self.out.path(f"bandwidth_result_{phase}.csv"))
        export_json(frf, path=self.out.path(f"bandwidth_result_{phase}.json"))
        return frf

    def _analyze_step_phase(self, data: list) -> Optional[dict]:
        """Preprocess + step response analysis."""
        if not data:
            logger.warning("[step] no data collected — skipping")
            return None

        logger.info(f"[step] collected {len(data)} samples")
        t, i_ref, i_meas, fs_det = preprocess(data, self.cfg.fs)
        self.cfg.fs = fs_det

        analyzer = NoiseAnalyzer(self.cfg)
        noise_report = analyzer.analyze(t, i_ref, i_meas)

        preprocessor = AdaptivePreprocessor(self.cfg)
        i_meas_filtered = preprocessor.apply(i_meas, noise_report, i_ref=i_ref)

        plot_noise_analysis(t, i_ref, i_meas, i_meas_filtered, noise_report,
                            self.cfg,
                            save_path=self.out.path("noise_analysis_step.png"))

        i_meas = i_meas_filtered

        step_data = analyze_step_response(t, i_ref, i_meas, self.cfg)
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

        save_npz = self.out.path("step_response_raw.npz")
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

        plot_step_results(t, i_ref, i_meas, step_data, self.cfg,
                          save_path=self.out.path("step_response_result.png"))
        return step_data["metrics"]

    def run(self) -> dict:
        cfg = self.cfg

        chirp_dur = cfg.chirp_duration
        msine_dur = cfg.chirp_duration
        step_dur  = (cfg.step_settle + cfg.step_hold) * cfg.step_repeats
        total_dur = chirp_dur + msine_dur + step_dur

        logger.info("=" * 58)
        logger.info("BLDC Current Controller  Bandwidth Measurement")
        if self.signal_type == "all":
            logger.info(f"  Mode   : sequential (chirp → multisine → step)")
        else:
            logger.info(f"  Mode   : single ({self.signal_type})")
        logger.info(f"  Freq   : {cfg.f_start}–{cfg.f_end} Hz  A={cfg.amplitude}A")
        logger.info(f"  UDP    : {cfg.udp_host}:{cfg.udp_port}")
        logger.info("=" * 58)

        self.receiver.start()

        logger.info("Waiting for 'bandwidth measure start' from STM32 …")
        if not self.receiver.wait_for_start(timeout=60.0):
            self.receiver.stop()
            raise TimeoutError("Timed out waiting for start message from STM32")

        logger.info(f"Recording … (timeout {total_dur + 60.0:.0f} s)")
        if not self.receiver.wait_for_done(timeout=total_dur + 60.0):
            logger.warning("Timed out waiting for done message — using collected data")

        self.receiver.stop()
        logger.info(f"UDP stats: {self.receiver.stats()}")

        results: dict = {}

        if self.signal_type in ("all", "chirp"):
            data = self.receiver.get_phase_data("chirp")
            frf_chirp = self._analyze_frf_phase("chirp", data)
            if frf_chirp:
                results["chirp"] = frf_chirp

        if self.signal_type in ("all", "multisine"):
            data = self.receiver.get_phase_data("multisine")
            frf_msine = self._analyze_frf_phase("multisine", data)
            if frf_msine:
                results["multisine"] = frf_msine

        if self.signal_type in ("all", "step"):
            data = self.receiver.get_phase_data("step")
            step_metrics = self._analyze_step_phase(data)
            if step_metrics:
                results["step"] = step_metrics

        logger.info("=" * 58)
        logger.info("Measurement complete — all phases processed")
        self.out.log_structure()
        logger.info("=" * 58)
        return results
