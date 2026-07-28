---
description: Independently cross-check the most recent (or specified) experiment results with fresh eyes
argument-hint: [result/log path or description, optional — defaults to the most recent run discussed]
---

Independently review the following experiment results for VLA-Spatial-Diagnostics: $ARGUMENTS

Use the Agent tool with `subagent_type: result-reviewer`. Give it enough to locate the actual result files and logs (paths, or what was just discussed in this conversation) — do not hand it only a prose summary and expect it to trust that summary. It reads the raw evidence itself, and its job is specifically to be skeptical of the experimenter's own interpretation.

It must walk the full checklist (설계가 가설을 검증하는지 / 독립변수 1개만 변경 / baseline 동일 조건 / seed 차이 가능성 / checkpoint 동일 여부 / 평가 데이터 누수 / 성공 판정 기준의 사후 변경 여부 / 결과와 로그의 일치 여부) and attach a confidence label (높음/중간/낮음) to every conclusion, with the specific gap named whenever confidence is not 높음.

Present its findings to the user as-is, including anything it flags as 낮음 confidence — do not soften or omit a low-confidence finding to make the result look more settled than it is.
