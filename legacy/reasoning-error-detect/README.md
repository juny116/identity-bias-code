# Reasoning Error Detection

언어모델이 풀이 과정의 오류를 찾아내고 답안을 평가하도록 하는 코드입니다. 오류 예시를 만들고, 모델별 탐지 응답과 정오 판단을 비교합니다.

## 코드 구조

| 경로 | 역할 |
|---|---|
| `detect_reasoning_errors.py` | 입력 풀이에 대한 모델의 오류 탐지 응답 생성 |
| `generate_responses.py` | 모델 실행 진입점 |
| `build_error_examples.py`, `build_error_ablation.py` | 오류 예시 생성과 변형 |
| `score_predictions.py`, `binned_detection_accuracy.py` | 응답 평가와 이진 판단 분석 |
| `models/` | 모델 API 어댑터 |
| `data/` | 수학 문제 로더와 JSONL 입출력 |
| `analysis/` | 오류 탐지 응답 분석 |
| `resample_model_outputs.py` | 지정한 인덱스로 샘플 재선택 |

## 사용

`detect_reasoning_errors.py`는 Hydra의 `conf/config`에서 모델 연결 정보, 프롬프트와 처리할 예시 수를 읽습니다. 이 디렉터리에서 실행하며, 모델 어댑터는 `models/`, 입력 데이터 처리는 `data/`를 사용합니다.

`resample_model_outputs.py`는 `RESAMPLE_INDICES_PATH` 환경변수로 지정한 JSON 인덱스 파일을 읽습니다.

데이터 전처리는 이 디렉터리에서 `python -m data.prepare_math` 또는 `python -m data.prepare_gsm8k`로 실행합니다. `score_correlation.py --input scores.csv --x COLUMN_A --y COLUMN_B`는 두 점수 열의 Pearson 상관계수를 계산합니다.
