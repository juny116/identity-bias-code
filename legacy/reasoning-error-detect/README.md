# Reasoning error detection prototypes

추론 오류 탐지, 답안 평가, 오류 예시 생성과 모델 어댑터 코드.

## 주요 코드

- [`error_detect.py`](error_detect.py)
- [`main.py`](main.py)
- [`sample_error_data.py`](sample_error_data.py)
- [`models`](models)

## 보존 범위

서버 미커밋 구현을 보존. 결과 수치가 포함된 plotting 스크립트는 검토본에서 제외.

소스 31개. 기준 Git HEAD: `cdad7f4571ed31108f3fd402576b9583fef1d6ca` + 당시 디스크 변경분.


## 실행과 검토

이 디렉터리는 코드 검토용 보존본입니다. 데이터·결과·로그·체크포인트·논문 원고·노트북 출력·외부 설정 파일·실행 셸 설정은 포함하지 않았습니다. 원래 상대 경로와 import 구조를 유지했지만, 비공개 설정과 의존성을 복원하기 전에는 실행 가능한 배포본이 아닙니다. API 키는 개인 환경에서 공급해야 하며, `REDACTED_*`와 `/path/to/user`는 제거된 값의 표시입니다.

Python 소스 문법 검사를 수행했습니다. 모델 다운로드, 학습, API 호출, 통합 실행 및 성능 재현은 하지 않았습니다. 소스에 있는 분석 함수는 결과 자체가 아니며 성능 주장을 의미하지 않습니다.

## 출처와 공개 상태

코드와 문서만 공개합니다. 개인 설정과 실험 산출물은 포함하지 않습니다. 기존 저작권 표기와 upstream 출처는 유지하며, 이 모음에 새 라이선스를 일괄 적용하지 않습니다. 외부 코드와 사용자 변경의 권리 확인은 실제 공개 대상별로 진행합니다.

`resample.py`의 개인 실험 sample index는 제거했습니다. 사용할 경우 `RESAMPLE_INDICES_PATH`로 개인 JSON index 파일을 제공하세요.
