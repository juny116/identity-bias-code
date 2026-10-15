# Identity Bias and Critic Evaluation

언어모델이 같은 풀이를 평가할 때 작성자 정보에 따라 판단을 바꾸는지 살펴보는 코드입니다. 풀이를 자기 답안, 다른 모델의 답안, 사람의 답안 등으로 제시하고 정오 판단, 수정 답안, confidence를 비교합니다. 평가 응답의 형식과 답안 제시 순서가 판단에 주는 영향도 분석합니다.

## 구조

```text
identity_bias/
├── workflows/       # 풀이 생성과 평가 실행
│   └── extended/    # 세션 문맥, 다수결, 판단 후 수정 실험
├── solver/          # 풀이 생성과 최종 답안 추출
├── critic/          # 작성자별 프롬프트, 평가 형식, 응답 파싱
├── llm/             # 모델 제공자별 API 어댑터
├── data/            # 벤치마크 로더와 정답 비교
├── evaluation/      # 오류 탐지·답안 수정·confidence 지표
├── analysis/        # 판단 불일치, 라벨 보정, 평가 형식 비교
│   └── extended/    # 회귀, 길이 민감도, 보정·다수결 분석
├── io/              # JSONL 입출력과 재개용 기록
├── runtime/         # 외부 모델 프로필, 작업 경로, 토큰 예산
└── types.py         # 작성자 조건, 데이터셋, 모델 인터페이스 타입
legacy/reasoning-error-detect/ # 오류 예시 생성과 탐지 도구
```

`workflows`는 데이터 로딩, 모델 생성, 실행 순서와 기록을 담당합니다. `CoTSolver`는 풀이에서 최종 답안을 추출하고, `IdentityCritic`은 작성자 조건에 맞는 메시지를 만들어 판단과 수정 답안을 파싱합니다. `ResultLogger`는 문제·조건별 완료 기록을 읽어 재개할 항목을 결정합니다.

## 설치

Python 3.10 이상에서 저장소 루트를 기준으로 실행합니다.

```bash
pip install -r requirements.txt
python -m identity_bias list
```

Anthropic과 Gemini 어댑터를 사용할 때는 각각 `anthropic`, `google-genai`를 추가 설치합니다. 통계 분석과 그림 생성은 `pip install -r requirements-analysis.txt`로 설치합니다.

## 풀이 생성과 작성자 평가

```bash
python -m identity_bias --models-file /path/to/models.json run generate-solutions \
  --model solver --dataset aime --log-dir /path/to/work/logs

python -m identity_bias --models-file /path/to/models.json run evaluate-authorship \
  --critic-model critic --dataset aime \
  --solutions-file /path/to/work/logs/solutions_solver_aime.jsonl \
  --conditions self other_model anonymous human --log-dir /path/to/work/logs
```

`--models-file`은 프로필 이름을 키로 갖는 JSON 객체를 읽습니다. 각 프로필에는 `provider`, `model_name`, `max_tokens`가 필요합니다. 연결 주소는 `base_url`, API 키를 읽을 환경변수 이름은 `api_key_env`로 지정합니다. `temperature`, `top_p`, `top_k`, `enable_thinking`, `reasoning_effort`도 지정할 수 있습니다. 지원 provider는 `openai`, `openai_responses`, `anthropic`, `google`, `vllm`입니다. `--base-url`은 해당 실행의 주소를 덮어씁니다.

생성·작성자 평가 명령은 저장된 완료 항목을 건너뜁니다. `--fresh`를 지정하면 해당 실행의 기존 기록을 지우고 다시 시작합니다. `evaluate-authorship --two-stage`는 정오 판단만 요청하는 호출과 판단·수정을 요청하는 호출을 각각 수행합니다.

## 주요 실행 명령

모든 명령은 `python -m identity_bias` 뒤에 붙입니다. `--models-file`과 `--workspace`는 `run` 또는 `analyze` 앞에 지정합니다.

| 명령 | 역할 |
|---|---|
| `run generate-solutions` | 문제 풀이 생성 |
| `run evaluate-authorship` | 같은 풀이의 작성자 조건 비교 |
| `run compare-solver-critic` | solver와 critic 조합별 지표 비교 |
| `run evaluate-verdict-interfaces` | 정오 판단·rubric·수정 답안 등의 출력 형식 비교 |
| `run evaluate-pairwise-interfaces` | 답안 쌍의 선호도와 제시 순서 효과 비교 |
| `data prepare-rewardbench` | 쌍대 선호도 평가용 문제 준비 |
| `analyze verdict-interface-metrics` | 오수락·오거절률, 순위, paired bootstrap, 회귀 분석 |
| `analyze verdict-disagreement` | 같은 항목의 평가 형식별 판단 불일치 |
| `analyze compare-reasoning-modes` | 추론 모드별 판단과 순위 비교 |
| `analyze compare-reference-answers` | 정답 제공 여부에 따른 판단 비교 |
| `analyze pairwise-preference-metrics` | 답안 선호도 변화와 위치 편향 분석 |
| `analyze estimate-label-errors` | 최종 답안 정규화에 따른 정답 라벨 변화 확인 |
| `analyze relabel-choice-answers` | 선택지 답안의 라벨 보정표 생성 |

```bash
python -m identity_bias --workspace /path/to/work analyze verdict-interface-metrics
python -m identity_bias run evaluate-authorship --help
python -m identity_bias analyze compare-reference-answers --help
```

`--workspace`의 기본값은 현재 디렉터리입니다. 평가 형식 분석은 그 아래 `logs/interface_confound_pilot`, 추론 모드 비교는 `logs/interface_confound_think`, 정답 제공 비교는 `logs/interface_confound_reference`를 읽습니다. 쌍대 선호도 분석은 `logs/subjective_confound`를 읽습니다. 라벨 보정표는 `paper/tables/corrected_labels.json`에서 읽어 분석 데이터의 복사본에 적용합니다.

추가 실험과 분석은 `run-extended`, `analyze-extended`에 모았습니다. 전체 이름은 `list`로 확인할 수 있습니다. [이전 파일과 새 모듈 대응표](docs/module-map.md), [추론 오류 탐지 도구](legacy/reasoning-error-detect/README.md)도 참고하세요.

## 테스트

```bash
pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

가짜 모델 응답으로 풀이→평가→저장→재개 흐름, 두 단계 평가, 답안 순서 교환, 잘린 응답 처리, 라벨 보정과 명령 진입점을 검사합니다.
