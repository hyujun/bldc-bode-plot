# bandwidth_measure.py 리팩토링 계획

## 목표
1. 2,700줄 모놀리식 파일을 기능별 모듈로 분리
2. 출력 폴더(YYMMDD_HHMM) 최대 10개 제한 (자동 정리)
3. `--reanalyze <folder>` 기능 추가 (저장된 .npz 데이터 재분석)

---

## Phase 1: 모듈 분리 구조

```
bode_plot/
├── __init__.py                  # 패키지 초기화
├── config.py                    # MeasurementConfig, 상수, 색상 정의
├── output_manager.py            # OutputManager (폴더 생성, 10개 제한 로직)
├── udp_receiver.py              # DataPoint, UDPReceiver, 프로토콜 상수/정규식
├── generators.py                # ChirpGenerator, MultisineGenerator, StepGenerator
├── dsp/
│   ├── __init__.py
│   ├── frf_estimator.py         # FRFEstimator (Welch PSD, H1/H2/Hv, BW, GM)
│   ├── noise_analyzer.py        # NoiseReport, NoiseAnalyzer
│   ├── preprocessor.py          # AdaptivePreprocessor, preprocess()
│   └── step_response.py         # _estimate_step_response, _step_metrics, _analyze_step_response
├── plotting/
│   ├── __init__.py
│   ├── style.py                 # 색상 상수, _apply_style(), _style_ax()
│   ├── bode_plot.py             # plot_results()
│   ├── step_plot.py             # plot_step_results()
│   ├── noise_plot.py            # plot_noise_analysis()
│   ├── comparison_plot.py       # plot_comparison(), _load_frf_from_npz()
│   ├── nyquist_plot.py          # plot_nyquist()
│   └── monitor.py               # run_monitor()
├── export.py                    # export_csv(), export_json()
├── measurement.py               # BandwidthMeasurement 클래스
├── demo.py                      # _simulate_plant(), _run_demo_single(), _run_demo()
├── reanalyze.py                 # 새 기능: 저장된 .npz 데이터 재분석
├── cli.py                       # argparse + main() 진입점
└── requirements.txt             # (기존 유지)
```

### 모듈 간 의존성 방향 (단방향 DAG)
```
cli.py
 ├── config.py
 ├── output_manager.py ← config
 ├── measurement.py ← udp_receiver, dsp/*, generators, plotting/*, export, output_manager
 ├── demo.py ← generators, dsp/*, plotting/*, export, output_manager
 ├── reanalyze.py ← dsp/*, plotting/*, export, output_manager   ★ NEW
 └── plotting/monitor.py ← udp_receiver, config
```

---

## Phase 2: OutputManager 폴더 제한 (10개)

**위치**: `output_manager.py`

- `OutputManager.__init__()` 에서 새 폴더 생성 직후 `_enforce_max_dirs(max_dirs=10)` 호출
- 로직:
  1. 현재 working directory에서 `YYMMDD_HHMM` 패턴(`^\d{6}_\d{4}$`)에 매칭되는 디렉토리 탐색
  2. mtime 기준 정렬
  3. `max_dirs`를 초과하는 가장 오래된 폴더부터 `shutil.rmtree()`로 삭제
  4. 삭제 시 logger.info()로 기록

---

## Phase 3: `--reanalyze <folder>` 기능

**위치**: `reanalyze.py`

### 동작 방식
```bash
# 기존 폴더의 .npz 데이터를 다시 분석 (새 결과는 같은 폴더에 저장)
python -m bode_plot --reanalyze 260317_1430

# 분석 옵션 조합 가능
python -m bode_plot --reanalyze 260317_1430 --signal chirp
```

### 구현
1. 지정된 폴더의 `data/` 하위 디렉토리에서 `.npz` 파일 탐색
2. `bandwidth_raw_chirp.npz`, `bandwidth_raw_multisine.npz`, `step_response_raw.npz` 를 로드
3. 각 .npz에서 `t`, `i_ref`, `i_meas` 원본 시계열 데이터를 추출
4. 현재 `MeasurementConfig` 파라미터로 NoiseAnalyzer → AdaptivePreprocessor → FRFEstimator 파이프라인 재실행
5. 결과물(plots, csv, json)은 같은 폴더에 `_reanalyzed` 접미사 또는 별도 서브폴더에 저장
6. 기존 원본 데이터(.npz)는 보존

### 핵심 함수
```python
def run_reanalyze(folder: str, signal_type: str = "all", cfg: MeasurementConfig = None) -> dict:
    """Load raw .npz data from folder and re-run analysis pipeline."""
```

---

## Phase 4: CLI 진입점 변경

**위치**: `cli.py` + `__main__.py`

- `python -m bode_plot` 로 실행 가능하도록 `__main__.py` 추가
- 기존 `python bandwidth_measure.py` 호환 위해 `cli.py`에 `if __name__ == "__main__"` 유지
- 새 인자: `--reanalyze FOLDER`

---

## 구현 순서

| Step | 작업 | 설명 |
|------|------|------|
| 1 | `config.py` | MeasurementConfig + 로깅 설정 추출 |
| 2 | `plotting/style.py` | 색상 상수, _apply_style, _style_ax 추출 |
| 3 | `output_manager.py` | OutputManager + 10개 제한 로직 추가 |
| 4 | `udp_receiver.py` | DataPoint, UDPReceiver, 프로토콜 상수 추출 |
| 5 | `generators.py` | 3개 Generator 클래스 추출 |
| 6 | `dsp/*.py` | FRF, Noise, Preprocessor, Step 추출 |
| 7 | `plotting/*.py` | 6개 plot 함수 각각 분리 |
| 8 | `export.py` | CSV/JSON export 추출 |
| 9 | `measurement.py` | BandwidthMeasurement 클래스 추출 |
| 10 | `demo.py` | 데모 관련 함수 추출 |
| 11 | `reanalyze.py` | 새 재분석 기능 구현 |
| 12 | `cli.py` + `__main__.py` | CLI 진입점 + --reanalyze 추가 |
| 13 | `__init__.py` | 패키지 초기화 및 public API 정의 |
| 14 | 검증 | `--demo`, `--reanalyze` 등 주요 모드 실행 확인 |
