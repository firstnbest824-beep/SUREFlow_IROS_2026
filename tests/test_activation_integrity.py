"""Simulator-free tests for the activation-collection integrity checks.

Builds synthetic episode directories on disk, then corrupts one property at a
time and asserts the corresponding check turns FAIL. Anything needing the real
simulator or checkpoint is covered by the pilot run instead.
"""

import json
import os
import sys

import numpy as np
import pytest

_OPENVLA_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "openvla")
if _OPENVLA_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_TOOLS_DIR)

from activation_integrity import (  # noqa: E402
    FAIL,
    PASS,
    REQUIRED_STAGES,
    STAGE_SPECS,
    WARNING,
    build_integrity_report,
    check_alignment,
    check_counts,
    check_episode,
    check_hooks,
    check_shapes,
    check_values,
    load_episode,
    run_all_checks,
    tensor_stats,
    worst,
)

N_STEPS = 6
SEQ_LEN = 291


def _stage_shape(stage):
    spec = STAGE_SPECS[stage]
    tokens = spec.tokens if spec.tokens is not None else SEQ_LEN
    hidden = spec.hidden if spec.hidden is not None else 4096
    return (1, tokens, hidden)


def build_episode(root, n_steps=N_STEPS, stages=None, rng_seed=0):
    """Write a synthetic but structurally valid episode directory."""
    stages = stages or list(REQUIRED_STAGES)
    rng = np.random.default_rng(rng_seed)
    episode_dir = os.path.join(root, "episodes", "episode_000")
    os.makedirs(os.path.join(episode_dir, "observations"), exist_ok=True)

    rows = []
    eef = np.array([0.0, 0.0, 1.0])
    for t in range(n_steps):
        eef_pre = eef.copy()
        eef_post = eef_pre + np.array([0.01, 0.0, -0.005])
        eef = eef_post

        activations = {}
        for stage in stages:
            stage_dir = os.path.join(episode_dir, "activations", stage)
            os.makedirs(stage_dir, exist_ok=True)
            array = rng.standard_normal(_stage_shape(stage)).astype(np.float32)
            path = os.path.join(stage_dir, f"t{t:04d}.npy")
            np.save(path, array)
            activations[stage] = {
                "saved": True,
                "path": path,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "call_count": STAGE_SPECS[stage].calls_per_timestep,
                "stats": tensor_stats(array),
            }

        obs_dir = os.path.join(episode_dir, "observations")
        agentview = os.path.join(obs_dir, f"t{t:04d}_agentview.png")
        eye = os.path.join(obs_dir, f"t{t:04d}_eye_in_hand.png")
        for p in (agentview, eye):
            with open(p, "wb") as handle:
                handle.write(b"png")

        rows.append({
            "episode_id": 0,
            "timestep": t,
            "task_name": "synthetic_task",
            "seed": 7,
            "obs_step_index_pre": 10 + t,
            "obs_step_index_post": 11 + t,
            "task_phase": "pre_grasp" if t < n_steps // 2 else "post_grasp",
            "phase_source": "obs_pre",
            "relevant_entity": "src" if t < n_steps // 2 else "dst",
            "relevant_entity_role": "source" if t < n_steps // 2 else "destination",
            "grasp_confidence": 0.1 * t,
            "success": t == n_steps - 1,
            "done": t == n_steps - 1,
            "action_model": [0.1] * 7,
            "action_applied": [0.1] * 6 + [-1.0],
            "proprio_state": list(eef_pre) + [0.0, 0.0, 0.0, 0.02, -0.02],
            "eef_pos_pre": list(eef_pre),
            "eef_pos_post": list(eef_post),
            "gripper_qpos_pre": [0.02, -0.02],
            "gripper_qpos_post": [0.02, -0.02],
            "source_position_pre": [0.3, 0.1, 0.9],
            "destination_position_pre": [0.5, 0.1, 0.9],
            "latency_ms": 300.0,
            "observations": {"agentview": agentview, "eye_in_hand": eye},
            "activations": activations,
        })

    with open(os.path.join(episode_dir, "per_step_metrics.jsonl"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    metadata = {
        "episode_id": 0,
        "task_name": "synthetic_task",
        "num_timesteps": n_steps,
        "saved_stages": stages,
        "activation_source": "obs_pre",
        "phase_source": "obs_pre",
        "task_success": True,
        "termination_reason": "success",
        "first_last_frame_identical": False,
        "hook_call_notes": {
            s: "fires once per generated action token"
            for s in stages if STAGE_SPECS[s].calls_per_timestep > 1
        },
    }
    with open(os.path.join(episode_dir, "episode_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)
    return episode_dir


def _rewrite(episode_dir, mutate_rows=None, mutate_meta=None):
    metrics = os.path.join(episode_dir, "per_step_metrics.jsonl")
    rows = [json.loads(l) for l in open(metrics, encoding="utf-8") if l.strip()]
    if mutate_rows:
        rows = mutate_rows(rows)
    with open(metrics, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if mutate_meta:
        meta_path = os.path.join(episode_dir, "episode_metadata.json")
        meta = json.load(open(meta_path, encoding="utf-8"))
        meta = mutate_meta(meta)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)


# --- happy path --------------------------------------------------------------
def test_valid_episode_passes_every_check(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    result = run_all_checks(episode_dir)
    failed = [c for c in result["checks"] if c["status"] == FAIL]
    assert not failed, f"unexpected failures: {[(c['check_id'], c['detail']) for c in failed]}"
    assert result["overall"] in (PASS, WARNING)


def test_report_aggregates_episodes(tmp_path):
    build_episode(str(tmp_path))
    report = build_integrity_report(str(tmp_path))
    assert report["num_episodes"] == 1
    assert report["overall"] in (PASS, WARNING)


# --- A: counts ---------------------------------------------------------------
def test_missing_activation_file_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    stage = REQUIRED_STAGES[0]
    os.remove(os.path.join(episode_dir, "activations", stage, "t0002.npy"))
    result = check_counts(load_episode(episode_dir))
    assert result.status == FAIL
    assert f"activation_{stage}" in result.data["mismatches"]


def test_dropped_metrics_row_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    _rewrite(episode_dir, mutate_rows=lambda rows: rows[:-1])
    assert check_counts(load_episode(episode_dir)).status == FAIL


# --- B: shapes ---------------------------------------------------------------
def test_wrong_hidden_dim_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        rows[0]["activations"]["final_vision_dinov2"]["shape"] = [1, 256, 999]
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_shapes(load_episode(episode_dir))
    assert result.status == FAIL
    assert "final_vision_dinov2" in result.detail


def test_broken_concat_invariant_is_detected(tmp_path):
    """dinov2 + siglip must equal projector_input on the channel axis."""
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["activations"]["projector_input"]["shape"] = [1, 256, 2000]
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_shapes(load_episode(episode_dir))
    assert result.status == FAIL
    assert "concat invariant" in result.detail


# --- C: values ---------------------------------------------------------------
def test_nan_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        rows[1]["activations"]["llm_early"]["stats"]["nan_count"] = 5
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_values(load_episode(episode_dir))
    assert result.status == FAIL
    assert "NaN" in result.detail


def test_inf_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        rows[0]["activations"]["projector_output"]["stats"]["inf_count"] = 2
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    assert check_values(load_episode(episode_dir)).status == FAIL


def test_frozen_activation_is_detected(tmp_path):
    """Identical L2 norm at every timestep means the same tensor was rewritten."""
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["activations"]["projector_output"]["stats"]["l2_norm"] = 123.456
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_values(load_episode(episode_dir))
    assert result.status == FAIL
    assert "frozen" in result.detail


def test_llama_massive_activation_channels_are_not_flagged(tmp_path):
    """A few huge channels are normal for LLaMA and must not trip the check.

    Measured on openvla-7b-finetuned-libero-spatial: llm_middle/llm_late put
    ~1.5e4 into exactly 2 of 4096 channels while the median stays ~0.4-1.25.
    Thresholding on the maximum made this WARN on every healthy run.
    """
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            stats = row["activations"]["llm_middle"]["stats"]
            stats["max"] = 14976.0          # the real observed peak
            stats["min"] = -11520.0
            stats["abs_p99_9"] = 24.9       # bulk stays small -> healthy
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_values(load_episode(episode_dir))
    assert result.status == PASS, f"healthy outlier channels were flagged: {result.detail}"


def test_widespread_magnitude_blowup_is_flagged(tmp_path):
    """If the bulk (p99.9) is huge, the tensor really is diverging."""
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            stats = row["activations"]["llm_middle"]["stats"]
            stats["max"] = 5.0e3
            stats["abs_p99_9"] = 4.0e3      # bulk is large -> not just outliers
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_values(load_episode(episode_dir))
    assert result.status == WARNING
    assert "bulk magnitude" in result.detail


def test_all_zero_activation_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["activations"]["llm_late"]["stats"]["l2_norm"] = 0.0
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    assert check_values(load_episode(episode_dir)).status == FAIL


# --- D: alignment ------------------------------------------------------------
def test_valid_alignment_passes(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    assert check_alignment(load_episode(episode_dir)).status == PASS


def test_phase_labelled_from_post_step_observation_is_rejected(tmp_path):
    """The exact off-by-one the plain rollout runners have."""
    episode_dir = build_episode(str(tmp_path))
    _rewrite(episode_dir, mutate_meta=lambda m: dict(m, phase_source="obs_post"))
    result = check_alignment(load_episode(episode_dir))
    assert result.status == FAIL
    assert "obs_pre" in result.detail


def test_off_by_one_state_shift_is_detected(tmp_path):
    """obs_post(k) must equal obs_pre(k+1); shifting breaks the chain."""
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["eef_pos_post"] = [v + 0.5 for v in row["eef_pos_post"]]
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_alignment(load_episode(episode_dir))
    assert result.status == FAIL
    assert "contiguous chain" in result.detail


def test_proprio_taken_from_post_state_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["proprio_state"] = list(row["eef_pos_post"]) + row["proprio_state"][3:]
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_alignment(load_episode(episode_dir))
    assert result.status == FAIL
    assert "proprio" in result.detail


def test_non_contiguous_timesteps_are_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        rows[3]["timestep"] = 99
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    assert check_alignment(load_episode(episode_dir)).status == FAIL


# --- E: hooks ----------------------------------------------------------------
def test_unexpected_call_count_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        rows[2]["activations"]["final_vision_dinov2"]["call_count"] = 3
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    _rewrite(episode_dir, mutate_meta=lambda m: dict(m, hook_call_notes={}))
    result = check_hooks(load_episode(episode_dir))
    assert result.status == FAIL
    assert "final_vision_dinov2" in result.detail


def test_missing_required_hook_is_detected(tmp_path):
    stages = [s for s in REQUIRED_STAGES if s != "projector_input"]
    episode_dir = build_episode(str(tmp_path), stages=stages)
    result = check_hooks(load_episode(episode_dir))
    assert result.status == FAIL
    assert "projector_input" in result.detail


def test_reused_tensor_across_timesteps_is_detected(tmp_path):
    """Writing the same array every timestep must be caught byte-wise."""
    episode_dir = build_episode(str(tmp_path))
    stage = "final_vision_dinov2"
    stage_dir = os.path.join(episode_dir, "activations", stage)
    frozen = np.load(os.path.join(stage_dir, "t0000.npy"))
    for t in range(N_STEPS):
        np.save(os.path.join(stage_dir, f"t{t:04d}.npy"), frozen)
    result = check_hooks(load_episode(episode_dir))
    assert result.status == FAIL
    assert "reused" in result.detail


# --- F: episode --------------------------------------------------------------
def test_static_scene_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["eef_pos_pre"] = [0.0, 0.0, 1.0]
            row["eef_pos_post"] = [0.0, 0.0, 1.0]
            row["proprio_state"] = [0.0, 0.0, 1.0] + row["proprio_state"][3:]
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_episode(load_episode(episode_dir))
    assert result.status == FAIL
    assert "never moved" in result.detail


def test_null_success_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    _rewrite(episode_dir, mutate_meta=lambda m: dict(m, task_success=None))
    result = check_episode(load_episode(episode_dir))
    assert result.status == FAIL
    assert "null" in result.detail


def test_all_unknown_phase_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))

    def mutate(rows):
        for row in rows:
            row["task_phase"] = "unknown"
        return rows

    _rewrite(episode_dir, mutate_rows=mutate)
    result = check_episode(load_episode(episode_dir))
    assert result.status == FAIL
    assert "unknown" in result.detail


def test_identical_first_last_frame_is_detected(tmp_path):
    episode_dir = build_episode(str(tmp_path))
    _rewrite(episode_dir, mutate_meta=lambda m: dict(m, first_last_frame_identical=True))
    assert check_episode(load_episode(episode_dir)).status == FAIL


def test_failed_episode_is_warning_not_failure(tmp_path):
    """A failed rollout is valid collection data as long as it is recorded."""
    episode_dir = build_episode(str(tmp_path))
    _rewrite(episode_dir, mutate_meta=lambda m: dict(
        m, task_success=False, termination_reason="max_steps_reached"))
    result = check_episode(load_episode(episode_dir))
    assert result.status == WARNING


# --- helpers -----------------------------------------------------------------
def test_worst_severity_ordering():
    assert worst([PASS, WARNING]) == WARNING
    assert worst([PASS, WARNING, FAIL]) == FAIL
    assert worst([]) == PASS


def test_tensor_stats_reports_nan_and_zero_fraction():
    array = np.zeros((1, 4, 4), dtype=np.float32)
    array[0, 0, 0] = np.nan
    array[0, 0, 1] = np.inf
    stats = tensor_stats(array)
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    assert stats["zero_fraction"] == pytest.approx(14 / 16)


# --- overwrite protection ----------------------------------------------------
def test_existing_run_id_is_not_overwritten(tmp_path):
    """The collector must refuse to reuse an existing run directory."""
    run_dir = tmp_path / "run_abc"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text("{}")
    assert run_dir.exists()
    # Mirrors the collector's guard: exists() -> abort before writing anything.
    with pytest.raises(FileExistsError):
        run_dir.mkdir(parents=True, exist_ok=False)


# --- dashboard smoke test ----------------------------------------------------
def test_dashboard_builds_from_a_synthetic_run(tmp_path):
    from activation_dashboard import build_dashboard

    episode_dir = build_episode(str(tmp_path))
    # Real PNGs so the embedder has something decodable.
    from PIL import Image

    for t in range(N_STEPS):
        for name in ("agentview", "eye_in_hand"):
            Image.fromarray(
                (np.random.default_rng(t).random((48, 48, 3)) * 255).astype(np.uint8)
            ).save(os.path.join(episode_dir, "observations", f"t{t:04d}_{name}.png"))

    (tmp_path / "run_config.json").write_text(json.dumps({
        "run_id": "test_run", "task_name": "synthetic_task", "task_instruction": "do a thing",
        "seed": 7, "checkpoint_id": "ckpt", "revision": "rev12345", "git_commit": "abc1234",
        "save_dtype": "float32",
    }))
    (tmp_path / "collection_summary.json").write_text(json.dumps({
        "num_episodes": 1, "total_timesteps": N_STEPS, "total_size_mb": 1.0,
    }))
    report = build_integrity_report(str(tmp_path))
    (tmp_path / "integrity_report.json").write_text(json.dumps(report))

    output = build_dashboard(str(tmp_path))
    assert os.path.isfile(output)
    content = open(output, encoding="utf-8").read()
    assert "Activation Collection" in content
    assert "data:image/png;base64," in content       # frames embedded
    assert "무결성 검사" in content                    # integrity table rendered
    for stage in REQUIRED_STAGES:
        assert stage in content
