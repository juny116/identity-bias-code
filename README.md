# Identity Bias and Critic Evaluation

언어모델이 답안을 평가할 때 작성자 정보에 따라 판단을 바꾸는지 살펴보는 실험 코드입니다. 같은 답안을 자기 답안, 다른 모델의 답안, 사람의 답안 등으로 제시하고 정오 판단, 오류 설명, 수정 답안과 confidence를 비교합니다.

## 주요 기능

- Solver 모델을 이용한 문제 풀이와 답안 생성
- 동일한 답안의 작성자 라벨을 바꾸는 identity-swap 평가
- Solver와 critic 모델 조합별 교차 평가
- 평가 인터페이스와 질문 형식에 따른 판단 비교
- 정오 판단, 답안 수정, confidence 및 항목별 불일치 분석

## 코드 구조

| 경로 | 역할 |
|---|---|
| `identity_bias/solver/` | 문제 풀이와 답안 생성 |
| `identity_bias/critic/` | 작성자 조건별 프롬프트와 critic 응답 파싱 |
| `identity_bias/llm/` | OpenAI, Anthropic, Gemini, vLLM 모델 클라이언트 |
| `identity_bias/data/` | 수학·지식 벤치마크 로더와 답안 처리 |
| `identity_bias/evaluation/` | 조건별 평가 지표 계산 |
| `identity_bias/logging/` | 실행 기록 저장 |
| `experiments/` | 답안 생성, identity swap, 모델 조합 및 인터페이스 실험 |
| `analysis/` | 항목별 판단 차이, 라벨 오류와 조건별 분석 |
| `legacy/reasoning-error-detect/` | 추론 오류 탐지와 오류 예시 생성 |
| `archive/old-framing/` | 추가 분석 및 실험 스크립트 |

## 실행 흐름

1. `experiments/run_solver.py`로 문제별 답안을 생성합니다.
2. `experiments/run_identity_swap.py`로 같은 답안을 여러 작성자 조건에서 평가합니다.
3. `experiments/run_solver_critic_matrix.py`로 solver–critic 조합을 비교합니다.
4. `analysis/`의 스크립트로 판단과 답안 수정 양상을 분석합니다.

실험 진입점은 `identity_bias.config`의 모델 preset과 데이터·작성자 조건 정의를 사용합니다. Identity-swap 실행에서는 `--critic-model`, `--solutions-file`, `--conditions`, `--dataset`으로 평가 대상을 선택합니다. `--two-stage`는 정오 판단만 요청하는 호출과 판단·수정을 함께 요청하는 호출을 각각 수행합니다.
