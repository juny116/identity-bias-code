# Identity Bias and Critic Evaluation

언어모델이 답안을 생성하고 평가하는 과정에서 작성자 정체성과 평가 인터페이스를
어떻게 다룰지 연구한 코드입니다. 결과 수치나 논문 원고는 포함하지 않습니다.

| 코드 | 역할 |
|---|---|
| `identity_bias/` | 모델 클라이언트, solver·critic, 평가 및 기록 모듈 |
| `experiments/run_solver_critic_matrix.py` | 모델이 자신의 답과 다른 모델의 답을 평가하는 흐름 |
| `experiments/run_identity_swap.py` | 정체성 조건을 바꾸는 실험 진입점 |
| `experiments/interface_confound/` | 평가 인터페이스 조건 |
| `analysis/` | 비공개 실행 결과를 입력으로 받는 분석 코드 |
| `legacy/reasoning-error-detect/` | 답안 오류 탐지·검증을 탐색했던 초기 구현 |
| `archive/old-framing/` | 과거 연구 구성의 분석 구현 |

오류 탐지 초기 코드는 현재 평가 연구와 주제가 연결되므로 이곳에 보존했습니다.
동일한 실험의 이전 버전이라는 증거가 있는 것은 아니며, 설정·모델·출력 형식이
다르므로 현재 파이프라인에 섞어 실행하지 않습니다. 초기 스크립트의 상대 import와
원래 파일 배치를 유지했습니다.

이 저장소는 코드 보존본입니다. 개인 모델 preset, 데이터, 환경 설정, 논문 소스와
결과 파일은 제외했습니다. 기존 코드가 참조하는 `identity_bias.config` 등 비공개
설정 인터페이스를 복원해야 전체 실험이 실행됩니다. 소스 공개와 재현용 배포는
구분하며, 공개할 경우 이 제한도 명시합니다.
