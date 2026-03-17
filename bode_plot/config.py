"""
config.py — Measurement configuration and logging setup.
"""

import logging
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bode_plot")


@dataclass
class MeasurementConfig:
    # UDP
    udp_host: str        = "0.0.0.0"
    udp_port: int        = 55150
    udp_buffer_size: int = 1024

    # Signal
    fs: float              = 500.0    # [Hz]  2 ms interval
    f_start: float         = 5.0      # [Hz]
    f_end: float           = 400.0    # [Hz]
    chirp_duration: float  = 30.0     # [s]
    amplitude: float       = 0.3      # [A]   15 % of nominal
    dc_bias: float         = 0.0      # [A]

    # Step test
    step_settle: float     = 0.5      # [s]  pre-step settle
    step_hold: float       = 0.2      # [s]  step hold duration
    step_repeats: int      = 5        # number of step cycles

    # FRF
    nperseg: int   = 2048
    noverlap: int  = 1024
    window: str    = "hann"

    # Safety
    max_current: float     = 1.0      # [A]
    nominal_current: float = 2.0      # [A]

    # Plot
    coh_threshold: float   = 0.80
    ref_f_low: float       = 10.0     # [Hz]  reference-level band
    ref_f_high: float      = 30.0     # [Hz]


CFG = MeasurementConfig()
