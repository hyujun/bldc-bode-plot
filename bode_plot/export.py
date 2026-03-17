"""
export.py — CSV and JSON export for FRF data.
"""

import csv
import json

from .config import logger


def export_csv(frf: dict, path: str = "bandwidth_result.csv") -> None:
    """Export FRF data to CSV for MATLAB / Excel interoperability."""
    f         = frf["f"]
    mag_db    = frf["mag_db"]
    phase_deg = frf["phase_deg"]
    coherence = frf["coherence"]

    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["frequency_hz", "magnitude_db", "phase_deg", "coherence"])
        for i in range(len(f)):
            writer.writerow([
                f"{f[i]:.6f}",
                f"{mag_db[i]:.6f}",
                f"{phase_deg[i]:.6f}",
                f"{coherence[i]:.6f}",
            ])
    logger.info(f"CSV exported → {path}")


def export_json(frf: dict, path: str = "bandwidth_result.json") -> None:
    """Export FRF summary + data to JSON."""
    out = {
        "bandwidth_hz": float(frf["bandwidth_hz"]),
        "gain_margin_db": frf.get("gain_margin_db"),
        "f_gain_margin_hz": frf.get("f_gain_margin"),
        "data": {
            "frequency_hz": frf["f"].tolist(),
            "magnitude_db": frf["mag_db"].tolist(),
            "phase_deg": frf["phase_deg"].tolist(),
            "coherence": frf["coherence"].tolist(),
        },
    }
    with open(path, "w") as fp:
        json.dump(out, fp, indent=2, default=float)
    logger.info(f"JSON exported → {path}")
