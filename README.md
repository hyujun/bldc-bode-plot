# BLDC Current Controller Bandwidth Measurement

BLDC 모터 PI 전류 제어기의 주파수 응답(Bode plot)과 스텝 응답을 측정하는 도구입니다.

**STM32H723** 에서 excitation signal을 생성하고, **UDP**로 실시간 전류 데이터를 전송하면, **Python** 스크립트가 수신하여 FRF 추정 및 시각화를 수행합니다.

```
STM32H723 (hand controller)          Desktop PC (192.168.1.2)
┌─────────────────────────┐          ┌──────────────────────────┐
│ chirp_generator.h       │          │ bandwidth_measure.py     │
│  → signal generation    │──UDP──→  │  → FRF estimation        │
│  → FDCAN TX/RX          │ :55150   │  → Bode plot / Step plot │
│  → current measurement  │          │  → .npz data save        │
└─────────────────────────┘          └──────────────────────────┘
```

## Quick Start

### Python (Desktop)

```bash
# venv 생성 및 의존성 설치
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Demo 모드 (하드웨어 없이 2차 시스템 시뮬레이션)
python bandwidth_measure.py --demo                        # chirp (기본)
python bandwidth_measure.py --demo --signal multisine     # multisine
python bandwidth_measure.py --demo --signal step          # step response

# Live 측정 (STM32에서 UDP 패킷 수신)
python bandwidth_measure.py --signal chirp
python bandwidth_measure.py --signal multisine
python bandwidth_measure.py --signal step

# 두 실험 결과 비교
python bandwidth_measure.py --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz
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
60개 로그 간격 주파수를 동시 여기. Schroeder 위상 최적화로 낮은 crest factor. Chirp과 다른 관점의 FRF 추정치를 제공하여 비교 검증에 활용.

### Step
0 → amplitude 스텝 입력을 5회 반복 후 앙상블 평균. Rise time, overshoot, settling time, steady-state error를 직접 측정.

## Measurement Metrics

### Frequency Domain (Chirp / Multisine)
| Metric | Description |
|--------|-------------|
| **Bandwidth** | -3 dB point relative to reference level (10~30 Hz band) |
| **Phase Margin** | 180° + phase at 0 dB gain crossover |
| **Coherence** | γ² > 0.80 threshold for data validity |

### Time Domain (Step)
| Metric | Description |
|--------|-------------|
| **Rise time** | 10% → 90% of steady-state |
| **Overshoot** | (peak - steady) / steady × 100% |
| **Settling time** | Time to enter ±2% band |
| **SS error** | Steady-state error from target |

## File Structure

```
bldc-bode-plot/
├── bandwidth_measure.py    — Python 분석 도구 (단일 파일, ~1200 lines)
├── chirp_generator.h       — STM32H723 header-only signal generator
├── example_stm32h723.c     — FreeRTOS 사용 예제
├── requirements.txt        — Python 의존성 (numpy, scipy, matplotlib)
├── CLAUDE.md               — AI assistant용 프로젝트 문서
└── README.md
```

## Configuration

### Python (`MeasurementConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fs` | 1000 Hz | Sampling rate |
| `f_start` / `f_end` | 5 / 400 Hz | Chirp sweep range |
| `chirp_duration` | 30 s | Chirp / multisine duration |
| `amplitude` | 0.3 A | Excitation amplitude |
| `step_settle` | 0.5 s | Step pre-settle time |
| `step_hold` | 0.2 s | Step hold time |
| `step_repeats` | 5 | Step repetition count |
| `nperseg` / `noverlap` | 2048 / 1024 | Welch window params |
| `coh_threshold` | 0.80 | Coherence validity filter |
| `max_current` | 1.0 A | Safety clamp |
| `udp_host` | 192.168.1.2 | UDP listen address |
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

- **Format:** `<dff` (little-endian: `double` timestamp_s + `float` i_ref + `float` i_meas)
- **Packet size:** 16 bytes
- **Direction:** STM32H723 → Desktop PC (`192.168.1.2:55150`)
- **Safety:** packets with |i_meas| > 3 × max_current are dropped

## Output Files

| File | Description |
|------|-------------|
| `bandwidth_result_chirp.png` | Chirp Bode plot (3-panel) |
| `bandwidth_result_multisine.png` | Multisine Bode plot (3-panel) |
| `step_response_result.png` | Step response plot (2-panel) |
| `bandwidth_comparison.png` | Chirp vs Multisine overlay (4-panel) |
| `bandwidth_raw_*.npz` | Raw FRF data (NumPy archive) |
| `step_response_raw.npz` | Raw step response data |

## Typical Workflow

```bash
# 1. STM32에서 chirp 실행, desktop에서 수신
python bandwidth_measure.py --signal chirp

# 2. STM32에서 multisine 실행, desktop에서 수신
python bandwidth_measure.py --signal multisine

# 3. 두 Bode plot 비교
python bandwidth_measure.py --compare bandwidth_raw_chirp.npz bandwidth_raw_multisine.npz

# 4. Step response 측정
python bandwidth_measure.py --signal step
```

## Dependencies

- **Python:** 3.9+, `numpy`, `scipy`, `matplotlib`
- **STM32:** ARM GCC, `math.h` (single-precision FPU), FreeRTOS
- **Network:** Ethernet, lwIP

## License

MIT
