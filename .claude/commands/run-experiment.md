---
description: Execute an already-approved experiment plan (not for designing new experiments)
argument-hint: [approved plan, or reference to one already in this conversation]
---

Execute the following already-approved experiment plan for VLA-Spatial-Diagnostics: $ARGUMENTS

This command only runs plans the user has already approved. If no concrete, approved plan is available (e.g. the user just described an idea without going through `/plan-experiment` or `/research-cycle`), stop and ask for the plan or run `/plan-experiment` first instead of guessing what to execute.

Use the Agent tool with `subagent_type: experiment-runner`. Before it runs anything at full scale, it must:

1. Verify checkpoint/dataset/config/GPU/environment/output-directory state and report what it found.
2. Perform a dry-run or a minimal (e.g. 1-step / small-N) execution first if the tooling supports it (most scripts under `tools/` do — check for a `--dry-run` or similarly scoped flag before assuming one doesn't exist).
3. Only then run the full approved experiment.

Do not let it expand scope beyond what was approved (e.g. running additional conditions "while it's at it") — if it identifies a good reason to do more, it should report that as a suggestion for the next cycle, not act on it unilaterally. Long-running GPU execution, `git commit`/`push`, package installs, and anything touching files outside the repo remain gated by `.claude/settings.json` permission prompts — do not attempt to bypass them.

Report back using experiment-runner's standard format: 성공/실패, 실행 명령, 변경 파일, 생성 파일, 중요 로그, 오류, 관찰 결과, 다음 조치.
