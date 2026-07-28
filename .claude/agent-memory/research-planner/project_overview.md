---
name: Project overview and active research line
description: Two research lines in this repo; OpenVLA diagnostics is active, SUREFlow is legacy-only
type: project
---

Active line: OpenVLA spatial diagnostics under `tools/openvla/`. Target model: `openvla/openvla-7b-finetuned-libero-spatial` @ revision `962318cec55ac10993ff0f5f43eda9a270b4c873`. Output root: `/home/hwkim/env-audit/`.

Legacy line: SUREFlow (`SUREFlow/`, `configs/`, `dataloader/`, `run.py`, `tools/dry_run_*`). IROS 2026 artifact, reproducibility only. Do not import or mix with OpenVLA tools.

Core research questions (from tools/openvla/README.md):
1. Where does spatial information weaken between vision encoder and action generation under LIBERO-PRO position perturbations?
2. Does the final action fail to use spatial information even when it remains in internal representations?
3. Can targeted repairs improve LIBERO-PRO generalization while preserving vanilla LIBERO performance?

Key LIBERO-PRO result driving the diagnostics: OpenVLA achieves 0.97 normalized SR on vanilla Spatial-Obj but 0.00 on Spatial-Pos (position perturbation). This near-total collapse is what we are diagnosing.

**Why:** Understanding the collapse mechanism before attempting any repair prevents wasted fine-tuning runs.
**How to apply:** Every plan should anchor to one of the three research questions above and must have a falsification condition tied to the position-perturbation failure.
