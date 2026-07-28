"""Tests for the activation storage backend (direct / staged / auto).

All tests use small temporary directories -- no simulator, no model, no real
activation data. They cover the properties that protect the data: transfers are
verified before the staging copy is dropped, failures preserve the source,
backpressure blocks instead of dropping, interrupted runs recover, and completed
episodes are never overwritten.
"""

import json
import os
import shutil
import sys
import threading
import time

import pytest

_OPENVLA_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "openvla")
if _OPENVLA_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _OPENVLA_TOOLS_DIR)

from activation_storage import (  # noqa: E402
    MANIFEST_NAME,
    MODE_AUTO,
    MODE_DIRECT,
    MODE_STAGED,
    ActivationStorage,
    build_manifest,
    describe_filesystem,
    verify_against_manifest,
)


def write_episode(path, n_files=5, size=2048):
    """Simulate a collector writing an episode into ``path``."""
    os.makedirs(os.path.join(path, "activations", "stage_a"), exist_ok=True)
    os.makedirs(os.path.join(path, "observations"), exist_ok=True)
    for i in range(n_files):
        with open(os.path.join(path, "activations", "stage_a", f"t{i:04d}.npy"), "wb") as fh:
            fh.write(os.urandom(size))
    with open(os.path.join(path, "observations", "t0000.png"), "wb") as fh:
        fh.write(os.urandom(256))
    with open(os.path.join(path, "episode_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"num_timesteps": n_files}, fh)


@pytest.fixture
def distinct_filesystems(monkeypatch):
    """Make staging and final look like different devices.

    A real staged setup needs two filesystems, but `tmp_path` puts both on one,
    so `auto` would (correctly) fall back to direct and the staged transfer logic
    would never be exercised. Patching the device probe lets these tests run
    anywhere without depending on the host's mount layout.
    """
    import activation_storage

    # Match on a distinctive token, not the bare word "stage": pytest's tmp_path
    # embeds the test name, and names like `test_staged_...` would otherwise make
    # both roots look like the staging device.
    monkeypatch.setattr(
        activation_storage, "device_of",
        lambda path: "stage-dev" if "STGROOT" in str(path) else "final-dev",
    )


def make_storage(tmp_path, mode=MODE_STAGED, **kwargs):
    params = dict(
        run_id="testrun",
        output_root=str(tmp_path / "FINROOT"),
        storage_mode=mode,
        staging_root=str(tmp_path / "STGROOT"),
        staging_max_gb=1.0,
        staging_min_free_gb=0.0001,
        transfer_queue_size=2,
        log=lambda *_: None,
    )
    params.update(kwargs)
    return ActivationStorage(**params)


# --- manifest ----------------------------------------------------------------
def test_manifest_round_trip(tmp_path):
    src = tmp_path / "ep"
    write_episode(str(src))
    manifest = build_manifest(str(src))
    assert manifest["file_count"] == 7
    assert manifest["total_bytes"] > 0
    assert verify_against_manifest(str(src), manifest) == []


def test_manifest_detects_missing_file(tmp_path):
    src = tmp_path / "ep"
    write_episode(str(src))
    manifest = build_manifest(str(src))
    os.remove(os.path.join(src, "activations", "stage_a", "t0000.npy"))
    problems = verify_against_manifest(str(src), manifest)
    assert any("missing file" in p or "file count" in p for p in problems)


def test_manifest_detects_truncated_file(tmp_path):
    src = tmp_path / "ep"
    write_episode(str(src))
    manifest = build_manifest(str(src))
    target = os.path.join(src, "activations", "stage_a", "t0001.npy")
    with open(target, "wb") as fh:
        fh.write(b"short")
    problems = verify_against_manifest(str(src), manifest)
    assert any("size mismatch" in p or "total bytes" in p for p in problems)


# --- direct mode -------------------------------------------------------------
def test_direct_mode_writes_straight_to_final(tmp_path):
    storage = make_storage(tmp_path, mode=MODE_DIRECT)
    assert storage.resolved_mode == MODE_DIRECT
    path = storage.begin_episode(0)
    assert path.endswith(".partial")
    write_episode(path)
    storage.finish_episode(0, path)
    final = storage.final_episode_path(0)
    assert os.path.isdir(final)
    assert not os.path.exists(path), "the .partial directory must be renamed away"
    storage.close(timeout=5)


def test_direct_mode_episode_is_atomic(tmp_path):
    """While writing, the episode must not be visible under its final name."""
    storage = make_storage(tmp_path, mode=MODE_DIRECT)
    path = storage.begin_episode(0)
    write_episode(path)
    assert not os.path.isdir(storage.final_episode_path(0))
    storage.finish_episode(0, path)
    assert os.path.isdir(storage.final_episode_path(0))
    storage.close(timeout=5)


# --- staged mode -------------------------------------------------------------
def test_staged_transfer_completes_and_verifies(tmp_path, distinct_filesystems):
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip(f"staged unavailable here: {storage.mode_reason}")
    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)
    assert storage.wait_for_transfers(timeout=60)
    storage.close(timeout=60)

    record = storage.records[0]
    assert record.verified is True
    assert record.verification_problems == []
    assert os.path.isdir(storage.final_episode_path(0))
    assert record.staging_removed is True
    assert not os.path.exists(record.staging_path)


def test_staged_final_content_matches_source_byte_for_byte(tmp_path, distinct_filesystems):
    """Staging must not alter the data in any way."""
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    path = storage.begin_episode(0)
    write_episode(path)
    expected = {}
    for root, _, files in os.walk(path):
        for name in files:
            full = os.path.join(root, name)
            expected[os.path.relpath(full, path)] = open(full, "rb").read()
    storage.finish_episode(0, path)
    storage.wait_for_transfers(timeout=60)
    storage.close(timeout=60)

    final = storage.final_episode_path(0)
    for rel, blob in expected.items():
        assert open(os.path.join(final, rel), "rb").read() == blob, rel


def test_keep_staging_on_success_retains_source(tmp_path, distinct_filesystems):
    storage = make_storage(tmp_path, keep_staging_on_success=True)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)
    storage.wait_for_transfers(timeout=60)
    record = storage.records[0]
    assert record.staging_removed is False
    assert os.path.isdir(record.staging_path)
    storage.close(timeout=60)


def test_failed_transfer_preserves_staging_copy(tmp_path, distinct_filesystems):
    """If the final copy cannot be verified, the source must survive."""
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    # Hold the worker so the manifest can be corrupted before the copy starts;
    # otherwise the transfer may finish before the corruption lands.
    gate = threading.Event()
    original = storage._transfer
    storage._transfer = lambda record: (gate.wait(timeout=10), original(record))[1]

    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)
    ready = storage.records[0].staging_path

    # Claim a file that does not exist: verification of the copy must fail.
    with open(os.path.join(ready, MANIFEST_NAME), "r+", encoding="utf-8") as fh:
        manifest = json.load(fh)
        manifest["files"]["activations/stage_a/ghost.npy"] = 999
        manifest["file_count"] += 1
        fh.seek(0); json.dump(manifest, fh); fh.truncate()
    gate.set()

    deadline = time.time() + 30
    while time.time() < deadline and not storage.errors:
        time.sleep(0.1)

    assert storage.errors, "a verification failure must be recorded"
    assert os.path.isdir(ready), "staging copy must be preserved on failure"
    assert not os.path.isdir(storage.final_episode_path(0)), "no partial episode in final"
    storage.close(timeout=10)


# --- overwrite protection ----------------------------------------------------
def test_existing_final_episode_is_never_overwritten(tmp_path):
    storage = make_storage(tmp_path, mode=MODE_DIRECT)
    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)
    storage.close(timeout=5)

    storage2 = make_storage(tmp_path, mode=MODE_DIRECT)
    with pytest.raises(FileExistsError):
        storage2.begin_episode(0)
    storage2.close(timeout=5)


# --- backpressure ------------------------------------------------------------
def test_backpressure_blocks_when_queue_is_full(tmp_path, distinct_filesystems):
    """The collector must wait, not drop data, when the queue is saturated."""
    storage = make_storage(tmp_path, transfer_queue_size=1)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")

    # Stall the worker so the queue stays full.
    blocker = threading.Event()
    original = storage._transfer

    def slow_transfer(record):
        blocker.wait(timeout=10)
        return original(record)

    storage._transfer = slow_transfer

    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)

    released = threading.Event()

    def try_begin():
        storage.begin_episode(1)
        released.set()

    thread = threading.Thread(target=try_begin, daemon=True)
    thread.start()
    time.sleep(0.8)
    assert not released.is_set(), "begin_episode must block while the queue is full"

    blocker.set()
    thread.join(timeout=30)
    assert released.is_set(), "collection must resume once space frees up"
    assert storage.backpressure_seconds > 0
    storage.close(timeout=60)


def test_backpressure_times_out_safely(tmp_path, monkeypatch, distinct_filesystems):
    """A permanently stuck queue must raise, not hang or discard data."""
    import activation_storage

    monkeypatch.setattr(activation_storage, "BACKPRESSURE_TIMEOUT_S", 1.0)
    storage = make_storage(tmp_path, transfer_queue_size=1)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")

    storage._transfer = lambda record: time.sleep(60)
    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)

    with pytest.raises(RuntimeError, match="backpressure"):
        storage.begin_episode(1)
    assert os.path.isdir(storage.records[0].staging_path), "data must be preserved"


def test_low_free_space_forces_direct_mode(tmp_path):
    """auto must fall back to direct when staging cannot be used safely."""
    storage = make_storage(tmp_path, mode=MODE_AUTO, staging_min_free_gb=10 ** 6)
    assert storage.resolved_mode == MODE_DIRECT
    assert "free" in storage.mode_reason or "floor" in storage.mode_reason
    storage.close(timeout=5)


def test_same_filesystem_forces_direct_mode(tmp_path):
    """Staging on the same filesystem as the destination is pointless."""
    storage = ActivationStorage(
        run_id="same_fs", output_root=str(tmp_path / "FINROOT"),
        storage_mode=MODE_AUTO, staging_root=str(tmp_path / "STGROOT"),
        staging_min_free_gb=0.0001, log=lambda *_: None,
    )
    assert storage.resolved_mode == MODE_DIRECT
    assert "same filesystem" in storage.mode_reason
    storage.close(timeout=5)


def test_explicit_staged_request_still_falls_back_loudly(tmp_path):
    storage = make_storage(tmp_path, mode=MODE_STAGED, staging_min_free_gb=10 ** 6)
    assert storage.resolved_mode == MODE_DIRECT
    assert "fell back" in storage.mode_reason
    storage.close(timeout=5)


# --- recovery ----------------------------------------------------------------
def test_recovery_requeues_ready_episodes(tmp_path, distinct_filesystems):
    """A .ready episode left by a crashed run must be transferred on restart."""
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    path = storage.begin_episode(3)
    write_episode(path)
    ready = os.path.join(storage.staging_root, "episode_003.ready")
    manifest = build_manifest(path)
    with open(os.path.join(path, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    os.rename(path, ready)
    storage.close(timeout=5)          # simulate the crash: worker gone

    storage2 = make_storage(tmp_path)
    if storage2.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    result = storage2.recover()
    assert result["requeued"] == 1
    assert storage2.wait_for_transfers(timeout=60)
    storage2.close(timeout=60)
    assert os.path.isdir(storage2.final_episode_path(3))


def test_recovery_reports_partial_as_orphan_and_keeps_it(tmp_path, distinct_filesystems):
    """.partial means an interrupted write -- never auto-delete it."""
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    path = storage.begin_episode(5)
    write_episode(path)               # left as .partial: simulates a crash
    storage.close(timeout=5)

    storage2 = make_storage(tmp_path)
    result = storage2.recover()
    assert result["found"]["partial"] == 1
    assert len(result["orphans"]) == 1
    assert os.path.isdir(path), ".partial must be preserved for inspection"
    storage2.close(timeout=5)


def test_recovery_discards_stale_transfer_tmp(tmp_path, distinct_filesystems):
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    stale = os.path.join(storage.transfer_tmp_root, "episode_009")
    os.makedirs(stale, exist_ok=True)
    write_episode(stale)
    result = storage.recover()
    assert result["found"]["transfer_tmp"] == 1
    assert not os.path.exists(stale), "an incomplete copy must not be trusted"
    storage.close(timeout=5)


def test_recovery_does_not_retransfer_completed_episode(tmp_path, distinct_filesystems):
    storage = make_storage(tmp_path)
    if storage.resolved_mode != MODE_STAGED:
        pytest.skip("staged unavailable")
    final = storage.final_episode_path(2)
    os.makedirs(final, exist_ok=True)
    write_episode(final)
    ready = os.path.join(storage.staging_root, "episode_002.ready")
    os.makedirs(ready, exist_ok=True)
    write_episode(ready)

    result = storage.recover()
    assert result["requeued"] == 0
    assert any("already complete" in a for a in result["actions"])
    storage.close(timeout=5)


# --- metadata ----------------------------------------------------------------
def test_metadata_contains_required_fields(tmp_path):
    storage = make_storage(tmp_path, mode=MODE_DIRECT)
    path = storage.begin_episode(0)
    write_episode(path)
    storage.finish_episode(0, path)
    storage.close(timeout=5)

    meta = storage.metadata()
    for key in ("storage_mode_requested", "storage_mode_resolved", "storage_mode_reason",
                "output_root", "final_filesystem", "staging_max_gb", "staging_min_free_gb",
                "transfer_queue_size", "backpressure_seconds", "max_queue_depth", "transfers"):
        assert key in meta, key
    assert meta["transfers"][0]["raw_bytes"] > 0
    assert meta["transfers"][0]["file_count"] > 0
    json.dumps(meta)                   # must be serializable


def test_describe_filesystem_reports_rotational_flag():
    info = describe_filesystem("/")
    assert "free_gb" in info
    # On this server / is NVMe; the field must at least be present and boolean/None.
    assert info.get("rotational") in (True, False, None)


# --- analysis-tool compatibility --------------------------------------------
def test_integrity_report_skips_partial_and_dot_dirs(tmp_path):
    """Staging leftovers must never be picked up as analysable episodes."""
    from activation_integrity import build_integrity_report

    run = tmp_path / "run"
    episodes = run / "episodes"
    episodes.mkdir(parents=True)
    (episodes / "episode_000.partial").mkdir()
    (run / "episodes" / ".transfer_tmp").mkdir()
    report = build_integrity_report(str(run))
    assert report["num_episodes"] == 0, "in-flight directories must be ignored"
