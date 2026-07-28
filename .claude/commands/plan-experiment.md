---
description: Turn a research question into a structured, executable experiment plan (design only, no execution)
argument-hint: [research question]
---

Turn the following research question into a structured experiment plan for VLA-Spatial-Diagnostics: $ARGUMENTS

Use the Agent tool with `subagent_type: research-planner` and give it this question plus any relevant context already established in this conversation (e.g. results of a prior `/research-cycle` step). Require its output to cover all of:

```
가설
변수 (독립변수 / 종속변수 / 통제변수)
baseline
실험군
평가 지표
실행 순서
필요 코드
예상 비용
중단 조건 (가설이 틀렸다고 판단할 조건)
```

Present the plan to the user as-is. Do not implement or run anything from this command — that is `/run-experiment`'s job, and only after the user approves this plan.
