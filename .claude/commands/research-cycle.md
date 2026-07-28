---
description: Run one full research cycle for VLA-Spatial-Diagnostics — audit, plan, approve, run, review, summarize
argument-hint: [research question or focus area]
---

Run one full research cycle for VLA-Spatial-Diagnostics on: $ARGUMENTS

Follow this order. Do not skip a step or merge steps to save time — each step exists specifically to keep experiments honest.

1. **Audit current state.** Use the Agent tool with `subagent_type: repository-auditor` to find out what's actually done vs. only coded vs. never run, relevant to this focus area. Do not proceed on an assumption the audit could check.
2. **Design the next experiment.** Use the Agent tool with `subagent_type: research-planner`, giving it the audit result and the research question. It returns a plan (hypothesis, variables, baseline, experimental/control groups, metrics, expected result, falsification condition, required code changes, estimated cost) — it does not execute anything.
3. **Report the plan and risks to the user, and stop for approval.** Summarize the plan plainly, flag anything expensive (long GPU run, large code change, anything touching `git commit`/`push`, package installs, or files outside the repo), and wait for explicit user go-ahead before continuing. Do not run a long-running GPU experiment or a large code change without this approval, regardless of how confident the plan looks.
4. **Implement and run only the approved scope.** Use the Agent tool with `subagent_type: experiment-runner`, passing exactly what the user approved — not a superset of it. It verifies checkpoint/dataset/config/GPU/output-directory state before running, prefers a dry-run or minimal run first, and reports command/files/logs/errors/observations.
5. **Independent review.** Use the Agent tool with `subagent_type: result-reviewer` to check the results with fresh eyes — it did not run the experiment and does not take the experimenter's interpretation at face value. It returns a verdict with a confidence label (높음/중간/낮음) per conclusion.
6. **Final summary.** Combine the plan, execution report, and independent review into one summary using the result-reporting format from the root `CLAUDE.md` (연구 질문 / 가설 / 변수 / 데이터셋 / checkpoint / 실행 명령 / seed / 지표 / 성공 또는 실패 / 관찰 / 해석 / 해석의 한계 / 다음 실험 / 생성된 파일 경로). Note any point where the reviewer's confidence was 중간 or 낮음.

If at any point a step reveals the plan doesn't make sense (e.g. the audit shows the "next" experiment already ran, or the planner flags an open question that blocks design), stop and surface that to the user instead of pushing forward.
