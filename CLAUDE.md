# BLDC Current Controller Bandwidth Measurement Tool

## Project Overview

Single-file Python tool (`bandwidth_measure.py`, ~789 lines) that measures the **frequency response** of a BLDC motor current controller via Bode plot analysis. It receives real-time current data over UDP from an embedded controller, performs FRF (Frequency Response Function) estimation using Welch's method, and produces a 3-panel visualization: Bode magnitude, Bode phase, and IFFT-based step response.

## Quick Start

```bash
# Live measurement (requires hardware sending UDP packets on port 9000)
python bandwidth_measure.py

# Synthetic demo (no hardware, simulates 2nd-order plant: ωn=180Hz, ζ=0.65)
python bandwidth_measure.py --demo
```

## Dependencies

Python 3.9+ with: `numpy`, `scipy`, `matplotlib`. No requirements.txt — install manually.

## Architecture

```
bandwidth_measure.py (single file, all-in-one)
├── MeasurementConfig    — dataclass with all tunable parameters
├── UDPReceiver          — threaded UDP listener, binary packet parser (<dff: 16B)
├── ChirpGenerator       — log-sweep 5–400 Hz, 30s excitation signal
├── FRFEstimator         — Welch PSD/CSD → H(jω), coherence, -3dB bandwidth
├── preprocess()         — sort, deduplicate, interpolate raw UDP data to uniform grid
├── _estimate_step_response() — IFFT of H(jω) → impulse → cumsum → step response
├── _step_metrics()      — rise time, overshoot, settling time extraction
├── plot_results()       — 3-panel dark-themed visualization (~290 lines)
├── _run_demo()          — synthetic 2nd-order system simulation
├── BandwidthMeasurement — live measurement orchestrator
└── __main__             — argparse entry point (--demo flag)
```

### Data Flow

**Live mode:** Hardware → UDP packets (timestamp, i_ref, i_meas) → UDPReceiver (threaded, deque buffer) → preprocess (sort/dedup/interp) → FRFEstimator → plot_results

**Demo mode:** ChirpGenerator → scipy.signal.lsim (2nd-order TF) + noise → FRFEstimator → plot_results

## Key Domain Concepts

- **FRF (Frequency Response Function):** H(jω) = Sxy(ω) / Sxx(ω) computed via Welch's cross/auto spectral density
- **Bandwidth:** -3dB point relative to reference level (mean magnitude in 10–30 Hz band)
- **Coherence:** γ² = |Sxy|² / (Sxx·Syy), threshold 0.80 filters unreliable data
- **Step response:** Estimated via IFFT of H(jω), not from time-domain measurement
- **Phase margin:** 180° + phase at 0dB gain crossover frequency

## Output Files

- `bandwidth_result.png` — 3-panel figure (14×9 inch, 140 dpi, dark theme)
- `bandwidth_raw.npz` — NumPy archive with f, mag_db, phase_deg, coherence, bandwidth_hz

## UDP Protocol

- Format: `<dff` (little-endian: double timestamp_s, float i_ref, float i_meas) = 16 bytes
- Default: `0.0.0.0:9000`, 1 MB receive buffer
- Safety: packets with |i_meas| > 3× max_current are dropped

## Configuration (MeasurementConfig defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| fs | 1000 Hz | Sampling rate (matches CAN FD rate) |
| f_start/f_end | 5–400 Hz | Chirp sweep range |
| chirp_duration | 30 s | Measurement duration |
| amplitude | 0.3 A | Chirp amplitude (15% of 2A nominal) |
| nperseg/noverlap | 2048/1024 | Welch window parameters (hann) |
| coh_threshold | 0.80 | Coherence validity filter |
| max_current | 1.0 A | Safety clamp |

## Code Conventions

- **Style:** Type-hinted Python 3.9+, dataclasses for config, stdlib logging
- **Naming:** Signal vars use `i_ref`, `i_meas`, `t`, `f`; spectral: `Sxx`, `Sxy`, `Syy`, `H`; suffixes `_r` (raw), `_u` (uniform), `_uw` (unwrapped)
- **Constants:** ALL_CAPS, color palette prefixed with `_` (`_BG`, `_BLUE`, etc.)
- **Structure:** Sections delimited by `════` comment blocks
- **Visualization:** Dark instrument-grade aesthetic, monospace font, color palette defined at module top

## When Modifying This Code

- Keep everything in the single file — this is intentionally a standalone script
- The Welch parameters (nperseg, noverlap, window) are tuned for 30s of 1kHz data; changing chirp_duration or fs may require re-tuning
- The step response estimation uses IFFT with edge windowing to suppress Gibbs ringing — be careful modifying `_estimate_step_response()`
- The plot function is large (~290 lines) due to extensive annotations; modify in sections marked by inner `════` comment blocks
- Safety bounds (max_current, amplitude clamping) exist for hardware protection — do not remove
