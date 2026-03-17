# BLDC Current Controller Bandwidth Measurement

BLDC 모터 PI 전류 제어기의 주파수 응답(Bode plot)과 스텝 응답을 측정하는 도구입니다.

**STM32H723** 에서 excitation signal을 생성하고, **UDP**로 실시간 전류 데이터를 전송하면, **Python** 스크립트가 수신하여 FRF 추정 및 시각화를 수행합니다.

```
STM32H723 (hand controller)          Desktop PC (0.0.0.0)
┌─────────────────────────┐          ┌──────────────────────────────┐
│ chirp_generator.h       │          │ bandwidth_measure.py         │
│  → signal generation    │──UDP──→  │  → noise analysis (adaptive) │
│  → FDCAN TX/RX          │ :55150   │  → FRF estimation (H1/Hv)   │
│  → current measurement  │ (text)   │  → Bode / Step / Nyquist    │
└─────────────────────────┘          └──────────────────────────────┘
```

## Quick Start

### Python (Desktop)

```bash
# venv 생성 및 의존성 설치
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r bode_plot/requirements.txt

# Demo 모드 (하드웨어 없이 2차 시스템 시뮬레이션)
python bode_plot/bandwidth_measure.py --demo                            # 전체 (chirp→multisine→step)
python bode_plot/bandwidth_measure.py --demo --signal chirp             # chirp 단독
python bode_plot/bandwidth_measure.py --demo --signal multisine         # multisine 단독
python bode_plot/bandwidth_measure.py --demo --signal step              # step response 단독
python bode_plot/bandwidth_measure.py --demo --noisy                    # BLDC 노이즈 시뮬레이션 + adaptive pipeline

# Live 측정 (STM32에서 UDP 패킷 수신)
python bode_plot/bandwidth_measure.py                                   # 전체 sequential (chirp→multisine→step)
python bode_plot/bandwidth_measure.py --signal chirp                    # chirp 단독
python bode_plot/bandwidth_measure.py --signal step                     # step 단독

# 실시간 모니터링
python bode_plot/bandwidth_measure.py --monitor

# 두 실험 결과 비교
python bode_plot/bandwidth_measure.py --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz

# Nyquist plot 생성
python bode_plot/bandwidth_measure.py --nyquist bandwidth_raw_chirp.npz

# 데이터 내보내기
python bode_plot/bandwidth_measure.py --export-csv bandwidth_raw_chirp.npz
python bode_plot/bandwidth_measure.py --export-json bandwidth_raw_chirp.npz
```

### STM32H723 (Embedded)

```c
// 컴파일 타임 설정 오버라이드 (선택)
#define CHIRP_AMPLITUDE     0.5f
#define STEP_N_REPEATS      10
#include "chirp_generator.h"
```

사용 예제는 `example_stm32h723.c` 참조 (FreeRTOS 기반).

## Excitation Signals

| Signal | 용도 | 측정 시간 | 출력 |
|--------|------|----------|------|
| **Chirp** | 주파수 응답 (Bode plot) | 30s | `bandwidth_result_chirp.png` |
| **Multisine** | 주파수 응답 (Bode plot) | 30s | `bandwidth_result_multisine.png` |
| **Step** | 시간 영역 성능 측정 | 3.5s | `step_response_result.png` |

### Chirp (Log Sweep)
5~400 Hz 로그 스윕. 넓은 주파수 범위를 순차적으로 커버하여 깔끔한 Bode plot 생성.

### Multisine (Schroeder Phase)
DFT bin-aligned 주파수를 동시 여기. Schroeder 위상 최적화로 낮은 crest factor. Chirp과 다른 관점의 FRF 추정치를 제공하여 비교 검증에 활용. 스펙트럼 누설(leakage) 없이 정확한 측정.

### Step
0 → amplitude 스텝 입력을 5회 반복 후 앙상블 평균. Rise time, overshoot, settling time, steady-state error를 직접 측정.

## Adaptive Noise Pipeline

BLDC 모터 전류 데이터의 노이즈 특성을 자동으로 분석하고, 적절한 전처리를 적용합니다.

```
i_meas (raw) → NoiseAnalyzer → NoiseReport → AdaptivePreprocessor → i_meas (clean) → FRF
                    │                              │
                    ├─ SNR 측정                     ├─ Despike (4σ outlier interpolation)
                    ├─ Tonal peak 검출              ├─ Notch filter (IIR, Q ∝ freq)
                    ├─ Spike 검출 (4σ)              └─ Lowpass (Butterworth 4th-order)
                    ├─ Coherence 분석
                    └─ Filter / Estimator 자동 추천
```

### 자동 판단 기준

| 조건 | 조치 |
|------|------|
| Spike 비율 > 0.5% | Despike (outlier interpolation) |
| Tonal peak 검출 + γ² > 0.90 | Notch filter |
| SNR < 15 dB | Lowpass filter (f_end × 1.2) |
| SNR < 10 dB | nperseg 축소 (더 많은 averaging) |
| Mean γ² < 0.70 | Hv estimator (H1·H2의 기하평균) |

### Demo에서 테스트

```bash
python bode_plot/bandwidth_measure.py --demo --noisy
```

`--noisy` 플래그는 PWM switching ripple, rotor ripple, EMI 스파이크, broadband noise를 시뮬레이션하여 adaptive pipeline을 검증합니다.

## Measurement Metrics

### Frequency Domain (Chirp / Multisine)

| Metric | Description |
|--------|-------------|
| **Bandwidth** | -3 dB point relative to reference level (10~30 Hz band) |
| **Phase Margin** | 180° + phase at 0 dB gain crossover |
| **Gain Margin** | -|H| at -180° phase crossover |
| **Coherence** | γ² > 0.80 threshold for data validity |

### Time Domain (Step)

| Metric | Description |
|--------|-------------|
| **Rise time** | 10% → 90% of steady-state |
| **Overshoot** | (peak - steady) / steady × 100% |
| **Settling time** | Time to enter ±2% band |
| **SS error** | Steady-state error from target |

## FRF Estimators

| Estimator | 수식 | 사용 시점 |
|-----------|------|----------|
| **H1** | Sxy / Sxx | 기본 (output noise에 robust) |
| **H2** | Syy / Syx | Input noise에 robust |
| **Hv** | √(H1 · H2) | 양쪽 noise 존재 시 (γ² < 0.70) |

## File Structure

```
bldc-bode-plot/
├── bode_plot/
│   ├── bandwidth_measure.py    — Python 분석 도구 (단일 파일, ~2700 lines)
│   ├── requirements.txt        — Python 의존성 (numpy, scipy, matplotlib)
│   ├── data/                   — 저장된 측정 결과 (.npz)
│   └── results/                — 시각화 결과 (.png)
├── stm32h723_udp/              — STM32H723 UDP 통신 구현 (별도 README)
├── chirp_generator.h           — STM32H723 header-only signal generator
├── example_stm32h723.c         — FreeRTOS 사용 예제
├── CLAUDE.md                   — AI assistant용 프로젝트 문서
└── README.md
```

## Configuration

### Python (`MeasurementConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | 500 Hz | Sampling rate |
| `f_start` / `f_end` | 5 / 400 Hz | Chirp sweep range |
| `chirp_duration` | 30 s | Chirp / multisine duration |
| `amplitude` | 0.3 A | Excitation amplitude |
| `step_settle` | 0.5 s | Step pre-settle time |
| `step_hold` | 0.2 s | Step hold time |
| `step_repeats` | 5 | Step repetition count |
| `nperseg` / `noverlap` | 2048 / 1024 | Welch window params |
| `coh_threshold` | 0.80 | Coherence validity filter |
| `max_current` | 1.0 A | Safety clamp |
| `udp_host` | 0.0.0.0 | UDP listen address |
| `udp_port` | 55150 | UDP listen port |

### C Header (compile-time `#define`)

| Define | Default | Description |
|--------|---------|-------------|
| `CHIRP_FS` | 1000 Hz | Sampling rate |
| `CHIRP_F_START` / `CHIRP_F_END` | 5 / 400 Hz | Sweep range |
| `CHIRP_DURATION` | 30 s | Signal duration |
| `CHIRP_AMPLITUDE` | 0.3 A | Peak amplitude |
| `CHIRP_MAX_CURRENT` | 1.0 A | Safety clamp |
| `MULTISINE_N_FREQS` | 60 | Number of sine components |
| `STEP_SETTLE_TIME` | 0.5 s | Pre-step settle |
| `STEP_HOLD_TIME` | 0.2 s | Step hold |
| `STEP_N_REPEATS` | 5 | Repetitions |

## UDP Protocol

- **Format:** Text-based (UTF-8)
- **Data line:** `chirp: t=0.001, ref=0.050, cur=0.048`
- **Phase tags:** `chirp:`, `MSine:`, `Step:`
- **Direction:** STM32H723 → Desktop PC (`0.0.0.0:55150`)
- **Safety:** packets with |i_meas| > 3 × max_current are dropped

### Control Messages

| Message | 의미 |
|---------|------|
| `bandwidth measure start` | 측정 시작 |
| `chirp done, transition to multisine` | Chirp → Multisine 전환 |
| `Multisine done, transition to step` | Multisine → Step 전환 |
| `step done, bandwidth measure completed` | 전체 측정 완료 |

## Output Files

| File | Description |
|------|-------------|
| `bandwidth_result_chirp.png` | Chirp Bode plot (3-panel) |
| `bandwidth_result_multisine.png` | Multisine Bode plot (3-panel) |
| `step_response_result.png` | Step response plot |
| `noise_analysis_*.png` | Noise analysis plot (3-panel) |
| `bandwidth_comparison.png` | Chirp vs Multisine overlay (4-panel) |
| `nyquist_result.png` | Nyquist diagram |
| `bandwidth_raw_*.npz` | Raw FRF data (NumPy archive) |
| `bandwidth_raw_*.csv` | FRF data (CSV export) |
| `bandwidth_raw_*.json` | FRF summary + data (JSON export) |
| `step_response_raw.npz` | Raw step response data |

## CLI Reference

| Flag | Description |
|------|-------------|
| `--demo` | 2차 시스템 시뮬레이션 (하드웨어 불필요) |
| `--signal {all,chirp,multisine,step}` | 신호 유형 선택 (기본: `all`) |
| `--noisy` | (demo only) BLDC 노이즈 시뮬레이션 + adaptive pipeline |
| `--monitor` | 실시간 UDP 모니터 (scrolling plot) |
| `--compare NPZ NPZ` | 두 .npz 결과 비교 (4-panel overlay) |
| `--nyquist NPZ` | Nyquist diagram 생성 |
| `--export-csv NPZ` | .npz → CSV 내보내기 |
| `--export-json NPZ` | .npz → JSON 내보내기 |

## Typical Workflow

```bash
# 1. STM32에서 전체 시퀀스 실행, desktop에서 수신
python bode_plot/bandwidth_measure.py

# 2. 자동으로 chirp → multisine → step 순차 처리
#    각 phase마다 noise analysis → adaptive preprocessing → FRF estimation

# 3. 두 Bode plot 비교
python bode_plot/bandwidth_measure.py --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz

# 4. Nyquist diagram으로 안정성 확인
python bode_plot/bandwidth_measure.py --nyquist bandwidth_raw_chirp.npz

# 5. 결과 내보내기 (MATLAB / Excel)
python bode_plot/bandwidth_measure.py --export-csv bandwidth_raw_chirp.npz
```

## Dependencies

- **Python:** 3.9+, `numpy`, `scipy`, `matplotlib`
- **STM32:** ARM GCC, `math.h` (single-precision FPU), FreeRTOS
- **Network:** Ethernet, lwIP

## License

MIT
