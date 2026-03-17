"""
udp_receiver.py — UDP data reception and protocol handling.

Receives sequential chirp → multisine → step data from STM32 over UDP.
"""

import re
import socket
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .config import MeasurementConfig, logger

# ── Protocol regex ────────────────────────────────────────
_DATA_RE = re.compile(
    r"(?:chirp|MSine|Step):\s*t=([\d.]+),\s*ref=([\-\d.]+),\s*cur=([\-\d.]+)",
    re.IGNORECASE,
)
_PHASE_RE = re.compile(
    r"(chirp|MSine|Step):", re.IGNORECASE,
)

# Protocol message constants
_MSG_START           = "bandwidth measure start"
_MSG_CHIRP_DONE      = "chirp done, transition to multisine"
_MSG_MULTISINE_DONE  = "Multisine done, transition to step"
_MSG_ALL_DONE        = "step done, bandwidth measure completed"

# Legacy protocol (kept for backward compat)
_MSG_LEGACY_START = "Bandwidth Measurement Started"
_MSG_LEGACY_DONE  = "Bandwidth Measurement Done"


@dataclass
class DataPoint:
    t:      float
    i_ref:  float
    i_meas: float


class UDPReceiver:
    """Receives sequential chirp → multisine → step data over UDP.

    Data is stored in per-phase buffers accessible via ``get_phase_data()``.
    The legacy single-signal protocol is also supported for backward compat.
    """

    PHASES = ("chirp", "multisine", "step")

    def __init__(self, cfg: MeasurementConfig):
        self.cfg     = cfg
        max_samples = int(cfg.fs * cfg.chirp_duration * 1.5)
        self._phase_buffers: dict[str, deque] = {
            p: deque(maxlen=max_samples) for p in self.PHASES
        }
        self.buffer  = deque(maxlen=max_samples)

        self._stop   = threading.Event()
        self._started = threading.Event()
        self._done    = threading.Event()
        self._phase_done: dict[str, threading.Event] = {
            p: threading.Event() for p in self.PHASES
        }
        self._current_phase: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._lock   = threading.Lock()
        self._rx     = 0
        self._drop   = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"UDP receiver  {self.cfg.udp_host}:{self.cfg.udp_port}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def wait_for_start(self, timeout: float = 60.0) -> bool:
        return self._started.wait(timeout=timeout)

    def wait_for_done(self, timeout: float = 120.0) -> bool:
        return self._done.wait(timeout=timeout)

    def wait_for_phase(self, phase: str, timeout: float = 120.0) -> bool:
        ev = self._phase_done.get(phase)
        if ev is None:
            return False
        return ev.wait(timeout=timeout)

    def get_data(self) -> list[DataPoint]:
        with self._lock:
            return list(self.buffer)

    def get_phase_data(self, phase: str) -> list[DataPoint]:
        with self._lock:
            buf = self._phase_buffers.get(phase, [])
            return list(buf)

    def stats(self) -> dict:
        phase_counts = {}
        with self._lock:
            for p in self.PHASES:
                phase_counts[p] = len(self._phase_buffers[p])
        return {"received": self._rx, "dropped": self._drop,
                "per_phase": phase_counts}

    @staticmethod
    def _classify_phase(msg: str) -> Optional[str]:
        m = _PHASE_RE.search(msg)
        if not m:
            return None
        tag = m.group(1).lower()
        if tag in ("chirp",):
            return "chirp"
        if tag in ("msine",):
            return "multisine"
        if tag in ("step",):
            return "step"
        return None

    def _run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind((self.cfg.udp_host, self.cfg.udp_port))
            sock.settimeout(0.5)
            collecting = False

            while not self._stop.is_set():
                try:
                    raw, _ = sock.recvfrom(self.cfg.udp_buffer_size)
                    msg = raw.decode("utf-8", errors="replace").strip()
                    msg_lower = msg.lower()

                    if _MSG_START.lower() in msg_lower or _MSG_LEGACY_START.lower() in msg_lower:
                        collecting = True
                        self._current_phase = "chirp"
                        self._started.set()
                        logger.info(f"Received: {msg}")
                        continue

                    if _MSG_CHIRP_DONE.lower() in msg_lower:
                        self._phase_done["chirp"].set()
                        self._current_phase = "multisine"
                        logger.info(f"Received: {msg}")
                        continue

                    if _MSG_MULTISINE_DONE.lower() in msg_lower:
                        self._phase_done["multisine"].set()
                        self._current_phase = "step"
                        logger.info(f"Received: {msg}")
                        continue

                    if (_MSG_ALL_DONE.lower() in msg_lower
                            or _MSG_LEGACY_DONE.lower() in msg_lower):
                        self._phase_done["step"].set()
                        collecting = False
                        self._current_phase = None
                        self._done.set()
                        logger.info(f"Received: {msg}")
                        continue

                    if not collecting:
                        continue

                    m = _DATA_RE.search(msg)
                    if not m:
                        self._drop += 1
                        continue

                    t_val  = float(m.group(1))
                    i_ref  = float(m.group(2))
                    i_meas = float(m.group(3))

                    if abs(i_meas) > self.cfg.max_current * 3:
                        self._drop += 1
                        continue

                    phase = self._classify_phase(msg) or self._current_phase
                    with self._lock:
                        self.buffer.append(DataPoint(t_val, i_ref, i_meas))
                        if phase and phase in self._phase_buffers:
                            self._phase_buffers[phase].append(
                                DataPoint(t_val, i_ref, i_meas))
                    self._rx += 1
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"UDP error: {e}")
        finally:
            sock.close()
