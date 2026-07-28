# tools/ 작업 지침

`tools/`와 `tools/openvla/`에는 학습/평가 파이프라인을 건드리지 않는 진단용 스크립트가 들어 있다 (`dry_run_*.py`, `run_single_vanilla_rollout.py`, `run_failure_screening.py`, `inspect_openvla_architecture.py` 등).

- 도구 스크립트는 기본적으로 dry-run(또는 `--max-steps 1` 같은 최소 실행) 모드를 지원하도록 만들거나 유지한다.
- 이 스크립트가 LIBERO 시뮬레이터 상태를 변경하는지(rollout 실행 등) 아니면 순수 읽기 전용인지(아키텍처 검사 등) 코드나 docstring에 명시한다.
- 실행 전에 입력 경로(checkpoint, dataset, bddl 파일)와 출력 경로를 먼저 출력한다.
- `spatial_task_resolver.py`, `model_input_transform.py`, `probe_hooks.py`, `task_phase_resolver.py`는 여러 스크립트가 공유하는 단일 진실 공급원(single source of truth)이다 — 로직을 복제하지 말고 반드시 import해서 쓴다.
- 객체 이름이나 좌표를 하드코딩하지 않는다. BDDL/시뮬레이터 상태에서 값을 유도한다 (`tools/openvla/README.md` 참고).
