# Reasoning Error Detection

언어모델이 풀이 과정의 오류를 찾아내고 답안을 평가하도록 하는 코드입니다. 오류 예시를 만들고, 모델별 탐지 응답과 정오 판단을 비교합니다.

## 코드 구조

| 경로 | 역할 |
|---|---|
| `error_detect.py` | 입력 풀이에 대한 모델의 오류 탐지 응답 생성 |
| `main.py` | 모델 실행 진입점 |
| `sample_error_data.py`, `sample_error_ablation.py` | 오류 예시 생성과 변형 |
| `eval.py`, `eval_bin.py` | 응답 평가와 이진 판단 분석 |
| `models/` | 모델 API 어댑터 |
| `data/` | 수학 문제 로더와 JSONL 입출력 |
| `analysis/` | 오류 탐지 응답 분석 |
| `resample.py` | 지정한 인덱스로 샘플 재선택 |

## 사용

`error_detect.py`는 Hydra의 `conf/config`에서 모델 연결 정보, 프롬프트와 처리할 예시 수를 읽습니다. 이 디렉터리에서 실행하며, 모델 어댑터는 `models/`, 입력 데이터 처리는 `data/`를 사용합니다.

`resample.py`는 `RESAMPLE_INDICES_PATH` 환경변수로 지정한 JSON 인덱스 파일을 읽습니다.
