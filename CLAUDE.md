# CLAUDE.md

이 파일은 Claude Code가 이 저장소에서 작업할 때 따르는 연구 운영 지침이다.

## 저장소 개요

두 개의 연구 라인이 공존한다 (자세한 내용은 `README.md` 상단 NOTICE 참고):

- **활성 라인 — OpenVLA spatial diagnostics** (`tools/openvla/`): `openvla/openvla-7b-finetuned-libero-spatial`를 대상으로 spatial perturbation 실패 원인을 분석한다.
- **레거시 — SUREFlow** (`SUREFlow/`, `configs/`, `dataloader/`, `run.py`, `tools/dry_run_*.py`): IROS 2026 발표 아티팩트. 재현성 유지 목적으로만 존재하며 현재 실험 대상이 아니다.
- `LIBERO-PRO/`: perturbation 벤치마크 서브모듈.
- `tests/`: pytest 유닛 테스트.

## 연구 목적

VLA 정책이 공간 변화(spatial perturbation)에서 실패하는 원인을 분석한다.

핵심 질문:
- 모델이 실제로 이미지를 사용하는가?
- 비전 인코더 내부에 target 위치 정보가 존재하는가?
- 모델이 영상 대신 proprioception 또는 궤적을 외우는가?
- LIBERO-Pro swap 및 position perturbation에서 왜 성능이 무너지는가?

## 작업 원칙

- 가설 없이 실험부터 실행하지 않는다.
- 한 번에 하나의 독립변수만 바꾼다.
- baseline 결과를 먼저 확보한다.
- 학습 데이터와 평가 데이터의 누수를 확인한다.
- checkpoint, seed, dataset, config를 반드시 기록한다.
- 성공 사례뿐 아니라 실패 사례도 보존한다.
- 결과를 먼저 보고 가설을 바꾸는 일(사후 가설 조정)을 피한다.

## 코드 수정 원칙

- 최소한의 코드만 수정한다.
- 기존 동작을 보존한다.
- 변경 전에 관련 코드를 읽는다.
- 실험 코드(`tools/openvla/`)와 핵심 모델/레거시 코드(`SUREFlow/`)는 가능하면 분리해서 다룬다.
- 임시 디버깅 코드를 남기지 않는다.
- 큰 리팩터링과 실험 변경을 동시에 하지 않는다.

## 결과 보고 형식

실험을 보고할 때는 다음 항목을 모두 포함한다:

```
연구 질문 / 가설
변경한 변수 / 통제한 변수
사용한 데이터셋 / 사용한 checkpoint
실행 명령 / seed
평가 지표
성공 또는 실패
관찰 결과 / 해석 / 해석의 한계
다음 실험
생성된 파일 경로
```

## 설명 방식

사용자에게 개념을 설명할 때는 다음 순서를 사용한다: (1) 쉬운 직관 → (2) 실제 코드 흐름 → (3) 전문 용어 → (4) 연구적으로 어떤 의미인지. 단순한 개념은 길게 설명하지 않는다.

## 연구용 서브에이전트 & 커맨드

`.claude/agents/`에 역할이 겹치지 않는 4개의 서브에이전트가 있다:

| 에이전트 | 역할 | memory |
|---|---|---|
| `research-planner` | 연구 질문 → 가설/실험 설계. 실행하지 않는다 | project |
| `experiment-runner` | 승인된 계획만 구현·실행 | project |
| `result-reviewer` | 실험자와 독립된 시각에서 결과 교차 검토 | project |
| `repository-auditor` | 저장소/실험 진행 상태를 추측 없이 분류 | project |

`.claude/commands/`의 `/research-cycle`, `/plan-experiment`, `/run-experiment`, `/review-results`가 이 에이전트들을 순서대로 orchestrate한다. 자세한 절차는 각 커맨드 파일 참고.

각 에이전트의 memory는 `.claude/agent-memory/<agent-name>/MEMORY.md`에 저장되고 매 세션 시작 시 앞부분이 자동 주입된다. 기록 규칙:

- `FACT:` — 실행으로 검증된 내용만 (확정 데이터셋/checkpoint 경로, 실제로 성공한 실행 명령, 반복 오류와 해결법, 지표 계산 방식)
- `HYPOTHESIS:` — 아직 실험으로 검증되지 않은 내용
- API 키, 토큰, 비밀번호, 개인정보, 한 번뿐인 로그 원문은 기록하지 않는다.
- 기존 memory 파일이 있으면 삭제·초기화하지 않고 이어서 갱신한다.

## Tasks 운영 (지원되는 경우)

연구 작업은 `backlog / planned / running / blocked / review / completed` 상태로 관리한다. 코드 파일이 존재한다는 이유만으로 `completed`로 표시하지 않는다 — 실제 실행 로그나 결과 파일이 있어야 `completed`로 판단한다. 각 task에는 목적, 선행 조건, 실행 명령, 예상/실제 결과, 생성 파일, 차단 원인, 다음 행동을 남긴다.

## 권한/안전 정책

`.claude/settings.json`에 최소한의 자동 허용(읽기/검색, `git status`/`diff`/`log`/`show`/`branch`)만 설정되어 있다. 파일 삭제, 저장소 밖 파일 수정, `sudo`, 패키지 설치, 모델/대용량 데이터 다운로드, `git commit`/`push`/`reset --hard`/`clean`, 원격 업로드, 장시간 GPU 실험은 항상 사용자 승인을 받는다. `Bash(*)`, `Edit(*)`, `Write(*)`, `bypassPermissions` 같은 전체 자동 허용은 이 저장소에 절대 추가하지 않는다.

## 하위 CLAUDE.md

- `tools/CLAUDE.md` — 도구 스크립트(dry-run, 실패 스크리닝 등) 작성 지침.

`experiments/`, `analysis/` 폴더는 저장소에 아직 존재하지 않아 하위 CLAUDE.md를 만들지 않았다. 해당 폴더가 실제로 생기면 이 파일의 하위 CLAUDE.md 규칙(재현 가능한 실행 명령, 별도 출력 디렉터리, 원본 결과 미변경 등)을 적용해 새로 추가한다.
