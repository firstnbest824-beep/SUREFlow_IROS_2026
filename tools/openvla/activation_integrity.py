"""Integrity checks for a collected activation episode.

Deliberately simulator-free and model-free: everything here reads the artifacts
an episode directory already contains, so the checks can be unit-tested against
synthetic fixtures and re-run on any past collection without a GPU.

A file existing is never treated as success. Every check returns an explicit
PASS / WARNING / FAIL with the numbers that produced the verdict.

The six check groups mirror the collection plan:

* A -- count consistency (timesteps vs. metrics rows vs. observations vs.
  actions vs. per-hook activation files)
* B -- shape verification against the specs in ``STAGE_SPECS``
* C -- value sanity (NaN/Inf, all-zero, frozen-across-time, extreme magnitude)
* D -- temporal alignment (the activation, the action and the phase label must
  all belong to the same observation; off-by-one detection)
* E -- hook execution (each required hook fires the expected number of times per
  timestep, and no timestep reuses the previous timestep's tensor)
* F -- episode sanity (the scene actually changed, phases are not all unknown,
  success is not left null)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PASS = "PASS"
WARNING = "WARNING"
FAIL = "FAIL"

_SEVERITY = {PASS: 0, WARNING: 1, FAIL: 2}


def worst(statuses: Sequence[str]) -> str:
    """The most severe status in ``statuses`` (PASS when empty)."""
    return max(statuses, key=lambda s: _SEVERITY.get(s, 0), default=PASS)


# -----------------------------------------------------------------------------
# Expected shapes
#
# Verified against the real checkpoint (see tools/openvla/README.md):
#   DINOv2  penultimate block, prefix stripped -> [1, 256, 1024]
#   SigLIP  penultimate block                  -> [1, 256, 1152]
#   concat(dim=2)                              -> [1, 256, 2176]
#   projector output                           -> [1, 256, 4096]
#   LLM prefill                                -> [1, 1 + 256 + n_text, 4096]
#
# `tokens=None` means "not fixed a priori, but must be identical across every
# timestep of the episode" -- the LLM sequence length depends on how many text
# tokens the task instruction produces, so hard-coding 291 would wrongly fail a
# different task.
# -----------------------------------------------------------------------------
@dataclass
class StageSpec:
    ndim: int
    batch: Optional[int]
    tokens: Optional[int]
    hidden: Optional[int]
    calls_per_timestep: int
    required: bool = True
    # Multi-call stages save the prefill full sequence plus a last-token stack.
    multi_call: bool = False


VISUAL_TOKENS = 256
LLM_HIDDEN = 4096

STAGE_SPECS: Dict[str, StageSpec] = {
    "final_vision_dinov2": StageSpec(3, 1, VISUAL_TOKENS, 1024, 1),
    "final_vision_siglip": StageSpec(3, 1, VISUAL_TOKENS, 1152, 1),
    "projector_input": StageSpec(3, 1, VISUAL_TOKENS, 2176, 1),
    "projector_output": StageSpec(3, 1, VISUAL_TOKENS, LLM_HIDDEN, 1),
    "llm_early": StageSpec(3, 1, None, LLM_HIDDEN, 7, multi_call=True),
    "llm_middle": StageSpec(3, 1, None, LLM_HIDDEN, 7, multi_call=True),
    "llm_late": StageSpec(3, 1, None, LLM_HIDDEN, 7, multi_call=True),
    "pre_action_hidden": StageSpec(3, 1, None, LLM_HIDDEN, 7, multi_call=True),
    # Optional: only saved with --save_lm_head_logits.
    "lm_head_logits": StageSpec(3, 1, None, None, 7, required=False, multi_call=True),
}

REQUIRED_STAGES = [name for name, spec in STAGE_SPECS.items() if spec.required]

# Value-sanity thresholds.
#
# LLaMA-family models carry "massive activations": a handful of fixed channels
# hold values orders of magnitude above the rest. Measured on this checkpoint,
# `llm_middle` and `llm_late` put ~1.5e4 into exactly 2 of 4096 channels (indices
# 2533 and 1415) while the median is ~0.4-1.25 and p99.99 is only ~25-43;
# `llm_early` shows none, and `pre_action_hidden` (post-RMSNorm) peaks at ~76.
# That is healthy model behaviour, so thresholding on the *maximum* would fire on
# every run and train everyone to ignore the check.
#
# The bulk percentile is what actually distinguishes numerical blow-up from
# normal outlier channels: if p99.9 is large, the whole tensor is diverging.
BULK_EXTREME_ABS_VALUE = 1.0e3   # p99.9 above this means large values are widespread
ABSOLUTE_EXTREME_VALUE = 1.0e6   # any value above this is genuine blow-up territory
MIN_TIMESTEP_DELTA_L2 = 1.0e-6
MAX_ZERO_FRACTION = 0.999


# -----------------------------------------------------------------------------
# Result containers
# -----------------------------------------------------------------------------
@dataclass
class CheckResult:
    check_id: str
    name: str
    status: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


def tensor_stats(array: np.ndarray) -> Dict[str, Any]:
    """Per-timestep summary statistics saved alongside every activation."""
    flat = np.asarray(array, dtype=np.float64).ravel()
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    return {
        "shape": [int(v) for v in np.asarray(array).shape],
        "dtype": str(np.asarray(array).dtype),
        "count": int(flat.size),
        "mean": float(finite_values.mean()) if finite_values.size else None,
        "std": float(finite_values.std()) if finite_values.size else None,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
        "l2_norm": float(np.linalg.norm(finite_values)) if finite_values.size else None,
        # Bulk magnitude. Separates "a few outlier channels" (normal for LLaMA)
        # from "the whole tensor is diverging" (a real problem).
        "abs_p99_9": (
            float(np.percentile(np.abs(finite_values), 99.9)) if finite_values.size else None
        ),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "zero_fraction": float((flat == 0).sum() / flat.size) if flat.size else None,
    }


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_episode(episode_dir: str) -> Dict[str, Any]:
    """Read the artifacts of one episode directory."""
    metadata_path = os.path.join(episode_dir, "episode_metadata.json")
    metrics_path = os.path.join(episode_dir, "per_step_metrics.jsonl")
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    rows: List[Dict[str, Any]] = []
    if os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return {"episode_dir": episode_dir, "metadata": metadata, "rows": rows}


def _activation_files(episode_dir: str, stage: str) -> List[str]:
    stage_dir = os.path.join(episode_dir, "activations", stage)
    if not os.path.isdir(stage_dir):
        return []
    return sorted(f for f in os.listdir(stage_dir) if f.endswith(".npy"))


def _primary_files(episode_dir: str, stage: str) -> List[str]:
    """One file per timestep: the single-call tensor or the prefill tensor."""
    return [
        f
        for f in _activation_files(episode_dir, stage)
        if not f.endswith("_last_token_stack.npy")
    ]


# -----------------------------------------------------------------------------
# A. Count consistency
# -----------------------------------------------------------------------------
def check_counts(episode: Dict[str, Any]) -> CheckResult:
    meta, rows, ep_dir = episode["metadata"], episode["rows"], episode["episode_dir"]
    declared = int(meta.get("num_timesteps", len(rows)))

    counts: Dict[str, int] = {
        "declared_timesteps": declared,
        "per_step_metrics_rows": len(rows),
        "actions_model": sum(1 for r in rows if r.get("action_model") is not None),
        "actions_applied": sum(1 for r in rows if r.get("action_applied") is not None),
        "observations_agentview": sum(
            1 for r in rows if (r.get("observations") or {}).get("agentview")
        ),
        "observations_eye_in_hand": sum(
            1 for r in rows if (r.get("observations") or {}).get("eye_in_hand")
        ),
    }
    saved_stages = meta.get("saved_stages") or REQUIRED_STAGES
    for stage in saved_stages:
        counts[f"activation_{stage}"] = len(_primary_files(ep_dir, stage))

    values = set(counts.values())
    if len(values) == 1:
        return CheckResult(
            "A", "count consistency", PASS,
            f"all {len(counts)} counters agree at {declared} timesteps", counts,
        )
    mismatches = {k: v for k, v in counts.items() if v != declared}
    reason = meta.get("count_mismatch_reason")
    if reason:
        return CheckResult(
            "A", "count consistency", WARNING,
            f"counts differ from {declared}; collector recorded a reason: {reason}",
            dict(counts, mismatches=mismatches, reason=reason),
        )
    return CheckResult(
        "A", "count consistency", FAIL,
        f"counters disagree with declared {declared} timesteps and no reason was recorded: {mismatches}",
        dict(counts, mismatches=mismatches),
    )


# -----------------------------------------------------------------------------
# B. Shape verification
# -----------------------------------------------------------------------------
def check_shapes(episode: Dict[str, Any]) -> CheckResult:
    meta, rows = episode["metadata"], episode["rows"]
    saved_stages = meta.get("saved_stages") or REQUIRED_STAGES
    problems: List[str] = []
    observed: Dict[str, Any] = {}

    for stage in saved_stages:
        spec = STAGE_SPECS.get(stage)
        if spec is None:
            problems.append(f"{stage}: no expected-shape spec")
            continue
        shapes = []
        for row in rows:
            entry = (row.get("activations") or {}).get(stage)
            if entry and entry.get("shape"):
                shapes.append(tuple(entry["shape"]))
        if not shapes:
            problems.append(f"{stage}: no shapes recorded")
            continue
        unique = sorted(set(shapes))
        observed[stage] = {
            "unique_shapes": [list(s) for s in unique],
            "expected": {
                "ndim": spec.ndim, "batch": spec.batch,
                "tokens": spec.tokens, "hidden": spec.hidden,
            },
        }
        if len(unique) != 1:
            problems.append(f"{stage}: shape varies across timesteps: {unique}")
            continue
        shape = unique[0]
        if len(shape) != spec.ndim:
            problems.append(f"{stage}: ndim {len(shape)} != expected {spec.ndim}")
            continue
        if spec.batch is not None and shape[0] != spec.batch:
            problems.append(f"{stage}: batch {shape[0]} != {spec.batch}")
        if spec.tokens is not None and shape[1] != spec.tokens:
            problems.append(f"{stage}: tokens {shape[1]} != {spec.tokens}")
        if spec.hidden is not None and shape[2] != spec.hidden:
            problems.append(f"{stage}: hidden {shape[2]} != {spec.hidden}")

    # Vision concat invariant: dinov2 hidden + siglip hidden == projector_input hidden.
    def hidden_of(stage: str) -> Optional[int]:
        entry = observed.get(stage)
        if not entry or len(entry["unique_shapes"]) != 1:
            return None
        return entry["unique_shapes"][0][2]

    d, s, p = hidden_of("final_vision_dinov2"), hidden_of("final_vision_siglip"), hidden_of("projector_input")
    if None not in (d, s, p):
        observed["concat_invariant"] = {"dinov2": d, "siglip": s, "sum": d + s, "projector_input": p}
        if d + s != p:
            problems.append(f"concat invariant broken: {d} + {s} != {p}")

    if problems:
        return CheckResult("B", "shape verification", FAIL, "; ".join(problems), observed)
    return CheckResult(
        "B", "shape verification", PASS,
        f"all {len(saved_stages)} stages match the expected shape spec, concat invariant holds",
        observed,
    )


# -----------------------------------------------------------------------------
# C. Value sanity
# -----------------------------------------------------------------------------
def check_values(episode: Dict[str, Any]) -> CheckResult:
    meta, rows = episode["metadata"], episode["rows"]
    saved_stages = meta.get("saved_stages") or REQUIRED_STAGES
    failures: List[str] = []
    warnings: List[str] = []
    summary: Dict[str, Any] = {}

    for stage in saved_stages:
        stats = [
            (row.get("activations") or {}).get(stage, {}).get("stats")
            for row in rows
        ]
        stats = [s for s in stats if s]
        if not stats:
            failures.append(f"{stage}: no statistics recorded")
            continue
        nan_total = sum(s.get("nan_count") or 0 for s in stats)
        inf_total = sum(s.get("inf_count") or 0 for s in stats)
        norms = [s.get("l2_norm") for s in stats if s.get("l2_norm") is not None]
        max_abs = max((abs(s.get("max") or 0.0) for s in stats), default=0.0)
        min_abs = max((abs(s.get("min") or 0.0) for s in stats), default=0.0)
        peak_abs = max(max_abs, min_abs)
        bulk = [s.get("abs_p99_9") for s in stats if s.get("abs_p99_9") is not None]
        max_bulk = max(bulk) if bulk else None
        zero_fracs = [s.get("zero_fraction") for s in stats if s.get("zero_fraction") is not None]

        summary[stage] = {
            "nan_total": nan_total,
            "inf_total": inf_total,
            "l2_norm_min": min(norms) if norms else None,
            "l2_norm_max": max(norms) if norms else None,
            "max_abs_value": peak_abs,
            "abs_p99_9_max": max_bulk,
            "zero_fraction_max": max(zero_fracs) if zero_fracs else None,
        }
        if nan_total:
            failures.append(f"{stage}: {nan_total} NaN values")
        if inf_total:
            failures.append(f"{stage}: {inf_total} Inf values")
        if norms and all(n == 0.0 for n in norms):
            failures.append(f"{stage}: every timestep is all-zero")
        if zero_fracs and max(zero_fracs) > MAX_ZERO_FRACTION:
            warnings.append(f"{stage}: zero_fraction up to {max(zero_fracs):.4f}")
        if max_bulk is not None and max_bulk > BULK_EXTREME_ABS_VALUE:
            warnings.append(
                f"{stage}: bulk magnitude is extreme (p99.9={max_bulk:.3e}); "
                "this is widespread, not a few outlier channels"
            )
        if peak_abs > ABSOLUTE_EXTREME_VALUE:
            warnings.append(f"{stage}: peak magnitude {peak_abs:.3e} suggests numerical blow-up")
        # Frozen across time: identical L2 norm at every timestep is a strong
        # signal that the same tensor was written repeatedly.
        if len(norms) > 1 and len(set(round(n, 10) for n in norms)) == 1:
            failures.append(f"{stage}: L2 norm identical at all {len(norms)} timesteps (frozen tensor?)")

    if failures:
        return CheckResult("C", "value sanity", FAIL, "; ".join(failures), summary)
    if warnings:
        return CheckResult("C", "value sanity", WARNING, "; ".join(warnings), summary)
    return CheckResult(
        "C", "value sanity", PASS,
        "no NaN/Inf, no all-zero stage, activations vary across timesteps", summary,
    )


# -----------------------------------------------------------------------------
# D. Temporal alignment
# -----------------------------------------------------------------------------
def check_alignment(episode: Dict[str, Any]) -> CheckResult:
    """The activation, the action and the phase must describe one observation.

    The collector captures activations from ``obs_pre`` -- the observation the
    policy actually saw -- and must label that timestep with the phase computed
    from the *same* ``obs_pre``. Labelling with the post-step observation (which
    the plain rollout runners do) shifts every label one control step ahead.
    """
    rows = episode["rows"]
    meta = episode["metadata"]
    problems: List[str] = []
    data: Dict[str, Any] = {
        "activation_source": meta.get("activation_source"),
        "phase_source": meta.get("phase_source"),
    }

    if meta.get("activation_source") != "obs_pre":
        problems.append(f"activation_source is {meta.get('activation_source')!r}, expected 'obs_pre'")
    if meta.get("phase_source") != "obs_pre":
        problems.append(f"phase_source is {meta.get('phase_source')!r}, expected 'obs_pre' to match the activation")

    # Timesteps must be contiguous and start at 0.
    timesteps = [r.get("timestep") for r in rows]
    if timesteps != list(range(len(rows))):
        problems.append(f"timesteps are not contiguous from 0: {timesteps[:8]}...")
    data["timestep_range"] = [timesteps[0], timesteps[-1]] if timesteps else []

    # Every row must agree on which env step it came from, and pre/post must
    # differ by exactly one env step.
    bad_steps = []
    for row in rows:
        pre, post = row.get("obs_step_index_pre"), row.get("obs_step_index_post")
        if pre is None or post is None:
            bad_steps.append((row.get("timestep"), pre, post))
        elif post != pre + 1:
            bad_steps.append((row.get("timestep"), pre, post))
    if bad_steps:
        problems.append(f"{len(bad_steps)} rows where obs_step_index_post != pre + 1, e.g. {bad_steps[:3]}")
    data["step_index_pairs_checked"] = len(rows)

    # Off-by-one detector: the proprioceptive state stored for this timestep must
    # be the PRE state. If it accidentally holds the POST state it will equal the
    # next row's pre-state instead.
    mismatched = 0
    shifted = 0
    for i, row in enumerate(rows[:-1]):
        this_pre = row.get("eef_pos_pre")
        this_post = row.get("eef_pos_post")
        next_pre = rows[i + 1].get("eef_pos_pre")
        if this_post is not None and next_pre is not None:
            if not _close(this_post, next_pre):
                mismatched += 1
        if this_pre is not None and next_pre is not None and _close(this_pre, next_pre):
            shifted += 1
    if mismatched:
        problems.append(
            f"{mismatched} timesteps where obs_post state != next timestep's obs_pre state "
            "(the rollout is not a contiguous chain)"
        )
    data["post_equals_next_pre_violations"] = mismatched
    data["consecutive_identical_pre_states"] = shifted
    if rows and shifted == len(rows) - 1:
        problems.append("every consecutive pre-state is identical; the robot never moved")

    # The proprio vector fed to the model must be the pre-state.
    proprio_mismatch = 0
    for row in rows:
        proprio, pre = row.get("proprio_state"), row.get("eef_pos_pre")
        if proprio and pre and not _close(proprio[:3], pre):
            proprio_mismatch += 1
    if proprio_mismatch:
        problems.append(f"{proprio_mismatch} rows where the model's proprio input != obs_pre eef position")
    data["proprio_vs_pre_mismatches"] = proprio_mismatch

    if problems:
        return CheckResult("D", "temporal alignment", FAIL, "; ".join(problems), data)
    return CheckResult(
        "D", "temporal alignment", PASS,
        "activation, action, proprio and phase all derive from the same pre-step observation; "
        "obs_post chains to the next obs_pre with no off-by-one",
        data,
    )


def _close(a: Sequence[float], b: Sequence[float], tol: float = 1e-9) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# -----------------------------------------------------------------------------
# E. Hook execution
# -----------------------------------------------------------------------------
def check_hooks(episode: Dict[str, Any]) -> CheckResult:
    meta, rows, ep_dir = episode["metadata"], episode["rows"], episode["episode_dir"]
    saved_stages = meta.get("saved_stages") or REQUIRED_STAGES
    problems: List[str] = []
    warnings: List[str] = []
    data: Dict[str, Any] = {}

    for stage in saved_stages:
        spec = STAGE_SPECS.get(stage)
        expected = spec.calls_per_timestep if spec else None
        call_counts = [
            (row.get("activations") or {}).get(stage, {}).get("call_count") for row in rows
        ]
        call_counts = [c for c in call_counts if c is not None]
        unique = sorted(set(call_counts))
        data[stage] = {"expected_calls_per_timestep": expected, "observed": unique}
        if not call_counts:
            problems.append(f"{stage}: no call counts recorded")
            continue
        if expected is not None and unique != [expected]:
            # A differing count is only acceptable if the collector explained it.
            note = (meta.get("hook_call_notes") or {}).get(stage)
            if note:
                warnings.append(f"{stage}: calls {unique} != {expected} ({note})")
            else:
                problems.append(f"{stage}: calls per timestep {unique}, expected {expected}")

    # Tensor reuse: consecutive timesteps must not be byte-identical.
    reuse: Dict[str, int] = {}
    for stage in saved_stages:
        files = _primary_files(ep_dir, stage)
        identical = 0
        previous: Optional[np.ndarray] = None
        for name in files:
            array = np.load(os.path.join(ep_dir, "activations", stage, name), mmap_mode="r")
            current = np.asarray(array, dtype=np.float32)
            if previous is not None and previous.shape == current.shape:
                if np.array_equal(previous, current):
                    identical += 1
            previous = current
        reuse[stage] = identical
        if files and identical == len(files) - 1 and len(files) > 1:
            problems.append(f"{stage}: every consecutive pair is byte-identical (previous tensor reused)")
        elif identical:
            warnings.append(f"{stage}: {identical} consecutive byte-identical pairs")
    data["consecutive_identical_pairs"] = reuse

    missing = [s for s in REQUIRED_STAGES if s not in saved_stages]
    if missing:
        problems.append(f"required hooks never saved: {missing}")
    data["missing_required_stages"] = missing

    if problems:
        return CheckResult("E", "hook execution", FAIL, "; ".join(problems), data)
    if warnings:
        return CheckResult("E", "hook execution", WARNING, "; ".join(warnings), data)
    return CheckResult(
        "E", "hook execution", PASS,
        "every required hook fired the expected number of times per timestep and no tensor was reused",
        data,
    )


# -----------------------------------------------------------------------------
# F. Episode sanity
# -----------------------------------------------------------------------------
def check_episode(episode: Dict[str, Any]) -> CheckResult:
    meta, rows = episode["metadata"], episode["rows"]
    problems: List[str] = []
    warnings: List[str] = []
    data: Dict[str, Any] = {}

    if not rows:
        return CheckResult("F", "episode sanity", FAIL, "no timesteps recorded", {})

    first, last = rows[0], rows[-1]
    eef_first, eef_last = first.get("eef_pos_pre"), last.get("eef_pos_post")
    if eef_first and eef_last:
        moved = float(np.linalg.norm(np.array(eef_last) - np.array(eef_first)))
        data["eef_total_displacement_m"] = moved
        if moved < 1e-4:
            problems.append(f"end-effector never moved (total displacement {moved:.2e} m)")
    else:
        problems.append("missing end-effector positions on the first/last timestep")

    src_first = first.get("source_position_pre")
    src_last = last.get("source_position_post") or last.get("source_position_pre")
    if src_first and src_last:
        data["source_object_displacement_m"] = float(
            np.linalg.norm(np.array(src_last) - np.array(src_first))
        )

    frames = [(r.get("observations") or {}).get("agentview") for r in rows]
    data["frames_recorded"] = sum(1 for f in frames if f)
    if data["frames_recorded"] < len(rows):
        problems.append(f"only {data['frames_recorded']}/{len(rows)} agentview frames saved")
    if meta.get("first_last_frame_identical") is True:
        problems.append("first and last agentview frames are pixel-identical")
    data["first_last_frame_identical"] = meta.get("first_last_frame_identical")

    phases = [r.get("task_phase") for r in rows]
    distribution: Dict[str, int] = {}
    for phase in phases:
        distribution[str(phase)] = distribution.get(str(phase), 0) + 1
    data["phase_distribution"] = distribution
    if all(p in (None, "unknown") for p in phases):
        problems.append("every timestep has an unknown/None task phase")

    success = meta.get("task_success")
    data["task_success"] = success
    data["termination_reason"] = meta.get("termination_reason")
    if success is None:
        problems.append("task_success was left null")
    if not meta.get("termination_reason"):
        problems.append("no termination reason recorded")
    if success is False:
        warnings.append(
            f"episode failed ({meta.get('termination_reason')}) -- valid for collection, recorded explicitly"
        )

    if problems:
        return CheckResult("F", "episode sanity", FAIL, "; ".join(problems), data)
    if warnings:
        return CheckResult("F", "episode sanity", WARNING, "; ".join(warnings), data)
    return CheckResult("F", "episode sanity", PASS, "scene changed, phases assigned, success recorded", data)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
ALL_CHECKS = (check_counts, check_shapes, check_values, check_alignment, check_hooks, check_episode)


def run_all_checks(episode_dir: str) -> Dict[str, Any]:
    episode = load_episode(episode_dir)
    results = [check(episode) for check in ALL_CHECKS]
    overall = worst([r.status for r in results])
    return {
        "episode_dir": episode_dir,
        "episode_id": episode["metadata"].get("episode_id"),
        "overall": overall,
        "checks": [r.to_dict() for r in results],
        "summary": {r.check_id: r.status for r in results},
    }


def build_integrity_report(run_dir: str) -> Dict[str, Any]:
    """Run every episode's checks and aggregate into one report."""
    episodes_root = os.path.join(run_dir, "episodes")
    # Only completed episodes are analysed. In-flight artifacts -- `.partial`
    # (interrupted write) and anything dot-prefixed such as `.transfer_tmp` --
    # are skipped so a crashed or still-running collection never corrupts a report.
    episode_dirs = sorted(
        os.path.join(episodes_root, d)
        for d in os.listdir(episodes_root)
        if os.path.isdir(os.path.join(episodes_root, d))
        and not d.startswith(".")
        and not d.endswith(".partial")
    ) if os.path.isdir(episodes_root) else []

    per_episode = [run_all_checks(d) for d in episode_dirs]
    overall = worst([e["overall"] for e in per_episode]) if per_episode else FAIL
    return {
        "run_dir": run_dir,
        "num_episodes": len(per_episode),
        "overall": overall,
        "episodes": per_episode,
    }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Verify a collected activation run.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="defaults to <run_dir>/integrity_report.json")
    args = parser.parse_args()

    report = build_integrity_report(args.run_dir)
    output = args.output or os.path.join(args.run_dir, "integrity_report.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Overall: {report['overall']}")
    for episode in report["episodes"]:
        print(f"  {os.path.basename(episode['episode_dir'])}: {episode['overall']}")
        for check in episode["checks"]:
            print(f"    [{check['status']:7s}] {check['check_id']} {check['name']}: {check['detail'][:110]}")
    print(f"Saved: {output}")
    return 0 if report["overall"] != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(_main())
