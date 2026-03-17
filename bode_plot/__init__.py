"""
bode_plot — BLDC Current Controller Bandwidth Measurement Package.

Modules:
  config          — MeasurementConfig and logging
  output_manager  — Timestamped output directory management
  udp_receiver    — UDP data reception from STM32
  generators      — Chirp, Multisine, Step signal generators
  dsp/            — FRF estimation, noise analysis, preprocessing
  plotting/       — Visualization (Bode, step, noise, Nyquist, etc.)
  export          — CSV/JSON export
  measurement     — Live measurement orchestrator
  demo            — Synthetic plant simulation
  reanalyze       — Re-analyze saved .npz data
  cli             — Command-line interface
"""

from .config import MeasurementConfig, CFG

__all__ = ["MeasurementConfig", "CFG"]
