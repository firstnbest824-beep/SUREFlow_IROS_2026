"""Storage backend for activation collection: direct or SSD-staged writes.

Why this exists
---------------
The final storage (`/home/user/4TB`, `/dev/sdb`) is a **rotational HDD**. Measured
with the real per-timestep write pattern it sustains far less than the
~89 MB/s the collector produces at 318 ms/timestep, so writing straight to it
makes the rollout I/O-bound. The root filesystem is NVMe but only has ~108 GB
free and must not be used as the final destination.

So: write each episode to a bounded SSD staging buffer, then move completed
episodes to the HDD from a background worker while the next episode is already
being collected.

Guarantees
----------
* An episode directory only appears under ``episodes/`` once every byte has been
  copied and verified. Readers (integrity checker, dashboard) therefore never
  see a partial episode.
* The staging copy is deleted only after the final copy has been verified
  file-by-file (relative path, size, and file count against the manifest).
* If a transfer fails the staging copy is kept and the error is recorded; the
  collector does not silently continue filling the disk.
* Backpressure blocks the collector rather than dropping data. Nothing is ever
  written to an over-full filesystem, and nothing is discarded.

Directory states (all inside the staging root)
----------------------------------------------
``<ep>.partial``  episode being written right now (or interrupted mid-write)
``<ep>.ready``    episode fully written, waiting for / undergoing transfer
``.transfer_tmp`` (under the final run dir) a copy in progress

Recovery on startup classifies leftovers into exactly those states. ``.ready``
is re-queued, ``.transfer_tmp`` is discarded and re-copied from ``.ready``, and
``.partial`` is reported as an orphan and never auto-deleted.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

MODE_DIRECT = "direct"
MODE_STAGED = "staged"
MODE_AUTO = "auto"

DEFAULT_OUTPUT_ROOT = "/home/user/4TB/hwkim/openvla_activation_collection"
DEFAULT_STAGING_ROOT = "/home/hwkim/.cache/openvla_activation_stage"
DEFAULT_STAGING_MAX_GB = 32.0
DEFAULT_STAGING_MIN_FREE_GB = 64.0
DEFAULT_TRANSFER_QUEUE_SIZE = 2

# How long the collector may sit in backpressure before we give up and stop
# safely instead of hanging forever.
BACKPRESSURE_TIMEOUT_S = 1800.0
_POLL_S = 0.25

MANIFEST_NAME = "transfer_manifest.json"


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def free_gb(path: str) -> float:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe).free / (1024 ** 3)


def device_of(path: str) -> Optional[str]:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return str(os.stat(probe).st_dev)
    except OSError:
        return None


def describe_filesystem(path: str) -> Dict[str, Any]:
    """Mount point, source device, fstype and rotational flag for ``path``."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    info: Dict[str, Any] = {"path": path, "resolved": probe}
    try:
        usage = shutil.disk_usage(probe)
        info.update(total_gb=usage.total / 1024 ** 3,
                    used_gb=usage.used / 1024 ** 3,
                    free_gb=usage.free / 1024 ** 3)
    except OSError:
        pass

    best: Optional[tuple] = None
    try:
        with open("/proc/self/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                source, mount, fstype = parts[0], parts[1].replace("\\040", " "), parts[2]
                real = os.path.realpath(probe)
                if real == mount or real.startswith(mount.rstrip("/") + "/"):
                    if best is None or len(mount) > len(best[1]):
                        best = (source, mount, fstype)
    except OSError:
        pass
    if best:
        info.update(source=best[0], mount_point=best[1], fstype=best[2])
        info["rotational"] = _is_rotational(best[0])
    return info


def _is_rotational(source: str) -> Optional[bool]:
    """True for spinning disks, False for SSD/NVMe, None if undeterminable."""
    name = os.path.basename(source)
    if not name.startswith(("sd", "nvme", "vd")):
        return None
    # strip partition suffix: sda2 -> sda, nvme0n1p2 -> nvme0n1
    base = name
    if base.startswith("nvme"):
        base = base.split("p")[0]
    else:
        base = base.rstrip("0123456789")
    try:
        with open(f"/sys/block/{base}/queue/rotational", "r", encoding="utf-8") as handle:
            return handle.read().strip() == "1"
    except OSError:
        return None


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def build_manifest(path: str) -> Dict[str, Any]:
    """Relative path -> size for every file, used to verify a transfer."""
    files: Dict[str, int] = {}
    for root, _, names in os.walk(path):
        for name in names:
            if name == MANIFEST_NAME:
                continue
            full = os.path.join(root, name)
            files[os.path.relpath(full, path)] = os.path.getsize(full)
    return {
        "file_count": len(files),
        "total_bytes": sum(files.values()),
        "files": files,
    }


def verify_against_manifest(path: str, manifest: Dict[str, Any]) -> List[str]:
    """Return a list of problems; empty means the copy matches exactly."""
    problems: List[str] = []
    actual = build_manifest(path)
    if actual["file_count"] != manifest["file_count"]:
        problems.append(
            f"file count {actual['file_count']} != expected {manifest['file_count']}"
        )
    if actual["total_bytes"] != manifest["total_bytes"]:
        problems.append(
            f"total bytes {actual['total_bytes']} != expected {manifest['total_bytes']}"
        )
    for rel, size in manifest["files"].items():
        got = actual["files"].get(rel)
        if got is None:
            problems.append(f"missing file: {rel}")
        elif got != size:
            problems.append(f"size mismatch {rel}: {got} != {size}")
        if len(problems) > 12:
            problems.append("... (truncated)")
            break
    return problems


def fsync_dir(path: str) -> None:
    """Make a rename durable by fsync'ing the containing directory."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


# -----------------------------------------------------------------------------
# Records
# -----------------------------------------------------------------------------
@dataclass
class TransferRecord:
    episode_id: int
    episode_name: str
    staging_path: Optional[str] = None
    final_path: Optional[str] = None
    raw_bytes: int = 0
    file_count: int = 0
    write_started_at: Optional[float] = None
    write_finished_at: Optional[float] = None
    queued_at: Optional[float] = None
    transfer_started_at: Optional[float] = None
    transfer_finished_at: Optional[float] = None
    queue_wait_s: Optional[float] = None
    transfer_seconds: Optional[float] = None
    transfer_mbps: Optional[float] = None
    verified: Optional[bool] = None
    verification_problems: List[str] = field(default_factory=list)
    staging_removed: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Storage manager
# -----------------------------------------------------------------------------
class ActivationStorage:
    """Decides direct vs staged, owns the transfer worker and backpressure."""

    def __init__(
        self,
        run_id: str,
        output_root: str,
        storage_mode: str = MODE_AUTO,
        staging_root: str = DEFAULT_STAGING_ROOT,
        staging_max_gb: float = DEFAULT_STAGING_MAX_GB,
        staging_min_free_gb: float = DEFAULT_STAGING_MIN_FREE_GB,
        transfer_queue_size: int = DEFAULT_TRANSFER_QUEUE_SIZE,
        keep_staging_on_success: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.run_id = run_id
        self.output_root = os.path.abspath(output_root)
        self.final_run_dir = os.path.join(self.output_root, run_id)
        self.requested_mode = storage_mode
        self.staging_max_gb = staging_max_gb
        self.staging_min_free_gb = staging_min_free_gb
        self.transfer_queue_size = max(1, transfer_queue_size)
        self.keep_staging_on_success = keep_staging_on_success
        self.log = log

        self.final_fs = describe_filesystem(self.output_root)
        self.staging_root = os.path.abspath(os.path.join(staging_root, run_id))
        self.staging_fs = describe_filesystem(staging_root)

        self.resolved_mode, self.mode_reason = self._resolve_mode()
        self.records: Dict[int, TransferRecord] = {}
        self.orphans: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.backpressure_seconds = 0.0
        self.max_queue_depth = 0

        self._queue: "queue.Queue[Optional[TransferRecord]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fatal: Optional[str] = None
        # An episode being copied right now is no longer in the queue but is
        # still occupying staging space, so it must count toward the depth limit.
        # Using qsize() alone would let the collector run one episode further
        # ahead than transfer_queue_size allows.
        self._in_flight = 0

        os.makedirs(self.final_run_dir, exist_ok=True)
        self.transfer_tmp_root = os.path.join(self.final_run_dir, ".transfer_tmp")
        self.episodes_dir = os.path.join(self.final_run_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

        if self.resolved_mode == MODE_STAGED:
            os.makedirs(self.staging_root, exist_ok=True)
            os.makedirs(self.transfer_tmp_root, exist_ok=True)
            self._start_worker()

    # -- mode resolution ------------------------------------------------------
    def _resolve_mode(self) -> tuple:
        """Pick direct/staged, falling back to direct with a recorded reason."""
        if self.requested_mode == MODE_DIRECT:
            return MODE_DIRECT, "explicitly requested"

        problems: List[str] = []
        staging_parent = os.path.dirname(self.staging_root.rstrip("/")) or "/"
        if self.staging_fs.get("rotational") is True:
            problems.append(
                f"staging filesystem is rotational ({self.staging_fs.get('source')}); "
                "staging on a spinning disk gives no benefit"
            )
        if device_of(staging_parent) == device_of(self.output_root):
            problems.append("staging and final output are on the same filesystem")
        staging_free = free_gb(staging_parent)
        if staging_free < self.staging_min_free_gb:
            problems.append(
                f"staging filesystem has {staging_free:.1f} GB free, below the "
                f"{self.staging_min_free_gb:.1f} GB floor"
            )
        try:
            os.makedirs(staging_parent, exist_ok=True)
            if not os.access(staging_parent, os.W_OK):
                problems.append(f"staging root not writable: {staging_parent}")
        except OSError as exc:
            problems.append(f"cannot create staging root: {exc}")

        if problems:
            reason = "; ".join(problems)
            if self.requested_mode == MODE_STAGED:
                # Explicit request, but unsafe -> still fall back, loudly.
                return MODE_DIRECT, f"staged requested but unsafe, fell back to direct: {reason}"
            return MODE_DIRECT, f"auto -> direct: {reason}"

        return MODE_STAGED, (
            f"auto -> staged: staging on {self.staging_fs.get('source')} "
            f"(rotational={self.staging_fs.get('rotational')}), final on "
            f"{self.final_fs.get('source')} (rotational={self.final_fs.get('rotational')})"
        )

    # -- episode lifecycle ----------------------------------------------------
    def episode_name(self, episode_id: int) -> str:
        return f"episode_{episode_id:03d}"

    def final_episode_path(self, episode_id: int) -> str:
        return os.path.join(self.episodes_dir, self.episode_name(episode_id))

    def begin_episode(self, episode_id: int) -> str:
        """Reserve space, wait out backpressure, return the write directory."""
        name = self.episode_name(episode_id)
        final_path = self.final_episode_path(episode_id)
        if os.path.isdir(final_path):
            raise FileExistsError(
                f"episode already exists in final storage, refusing to overwrite: {final_path}"
            )

        record = TransferRecord(episode_id=episode_id, episode_name=name)
        record.write_started_at = time.time()
        self.records[episode_id] = record

        if self.resolved_mode == MODE_DIRECT:
            self._require_free_space(self.output_root, "final")
            path = final_path + ".partial"
            os.makedirs(path, exist_ok=True)
            record.staging_path = None
            return path

        self._await_capacity()
        path = os.path.join(self.staging_root, name + ".partial")
        os.makedirs(path, exist_ok=True)
        record.staging_path = os.path.join(self.staging_root, name + ".ready")
        return path

    def finish_episode(self, episode_id: int, write_dir: str) -> None:
        """Seal the episode: manifest, .partial -> .ready, enqueue transfer."""
        record = self.records[episode_id]
        record.write_finished_at = time.time()

        manifest = build_manifest(write_dir)
        record.raw_bytes = manifest["total_bytes"]
        record.file_count = manifest["file_count"]
        with open(os.path.join(write_dir, MANIFEST_NAME), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        fsync_dir(write_dir)

        if self.resolved_mode == MODE_DIRECT:
            final_path = self.final_episode_path(episode_id)
            os.rename(write_dir, final_path)
            fsync_dir(self.episodes_dir)
            record.final_path = final_path
            record.verified = True
            self.log(f"  [storage] episode {episode_id} written directly to {final_path}")
            return

        ready = record.staging_path
        os.rename(write_dir, ready)
        fsync_dir(self.staging_root)
        record.queued_at = time.time()
        self._queue.put(record)
        self.max_queue_depth = max(self.max_queue_depth, self._pending_transfers())
        self.log(
            f"  [storage] episode {episode_id} staged ({record.raw_bytes/1e6:.0f} MB, "
            f"{record.file_count} files), queued for transfer (depth={self._queue.qsize()})"
        )

    # -- backpressure ---------------------------------------------------------
    def _staging_usage_gb(self) -> float:
        if not os.path.isdir(self.staging_root):
            return 0.0
        return dir_size_bytes(self.staging_root) / (1024 ** 3)

    def _require_free_space(self, path: str, label: str) -> None:
        available = free_gb(path)
        floor = self.staging_min_free_gb if label == "staging" else 5.0
        if available < floor:
            raise RuntimeError(
                f"{label} filesystem has only {available:.1f} GB free "
                f"(floor {floor:.1f} GB); refusing to write"
            )

    def _await_capacity(self) -> None:
        """Block until staging has room and the queue has drained enough."""
        started = time.time()
        warned = False
        while True:
            if self._fatal:
                raise RuntimeError(f"transfer worker failed: {self._fatal}")
            usage = self._staging_usage_gb()
            staging_free = free_gb(self.staging_root)
            depth = self._pending_transfers()

            over_budget = usage >= self.staging_max_gb
            low_disk = staging_free <= self.staging_min_free_gb
            queue_full = depth >= self.transfer_queue_size
            if not (over_budget or low_disk or queue_full):
                waited = time.time() - started
                if waited > 0.05:
                    self.backpressure_seconds += waited
                    self.log(f"  [storage] resumed after {waited:.1f}s of backpressure")
                return

            if not warned:
                self.log(
                    f"  [storage] backpressure: staging={usage:.1f}/{self.staging_max_gb} GB, "
                    f"free={staging_free:.1f} GB (floor {self.staging_min_free_gb}), "
                    f"queue={depth}/{self.transfer_queue_size} -- pausing collection"
                )
                warned = True

            if time.time() - started > BACKPRESSURE_TIMEOUT_S:
                self.backpressure_seconds += time.time() - started
                raise RuntimeError(
                    f"storage backpressure did not clear within {BACKPRESSURE_TIMEOUT_S:.0f}s "
                    f"(staging {usage:.1f} GB, free {staging_free:.1f} GB, queue {depth}). "
                    "Stopping safely; no data was discarded."
                )
            time.sleep(_POLL_S)

    # -- transfer worker ------------------------------------------------------
    def _start_worker(self) -> None:
        # A single worker on purpose: concurrent copies to one spinning disk
        # multiply seeks and reduce total throughput.
        self._worker = threading.Thread(target=self._worker_loop, name="transfer", daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            with self._lock:
                self._in_flight += 1
            try:
                self._transfer(item)
            except Exception as exc:  # keep the staging copy, record, keep going
                item.error = str(exc)
                self.errors.append(f"episode {item.episode_id}: {exc}")
                self._fatal = str(exc)
                self.log(f"  [storage][ERROR] transfer of episode {item.episode_id} failed: {exc}")
                self.log("  [storage] staging copy preserved; not deleting source")
            finally:
                with self._lock:
                    self._in_flight -= 1
                self._queue.task_done()

    def _pending_transfers(self) -> int:
        """Queued plus currently-copying episodes."""
        with self._lock:
            return self._queue.qsize() + self._in_flight

    def _transfer(self, record: TransferRecord) -> None:
        source = record.staging_path
        record.transfer_started_at = time.time()
        record.queue_wait_s = record.transfer_started_at - (record.queued_at or record.transfer_started_at)

        if not os.path.isdir(source):
            raise FileNotFoundError(f"staging directory vanished: {source}")

        with open(os.path.join(source, MANIFEST_NAME), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        final_path = self.final_episode_path(record.episode_id)
        if os.path.isdir(final_path):
            raise FileExistsError(f"final episode already exists: {final_path}")

        available = free_gb(self.output_root)
        needed = manifest["total_bytes"] / (1024 ** 3)
        if available < needed + 5.0:
            raise RuntimeError(
                f"final filesystem has {available:.1f} GB free, need {needed:.1f} GB + 5 GB margin"
            )

        tmp = os.path.join(self.transfer_tmp_root, record.episode_name)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(self.transfer_tmp_root, exist_ok=True)
        shutil.copytree(source, tmp)

        problems = verify_against_manifest(tmp, manifest)
        record.verification_problems = problems
        record.verified = not problems
        if problems:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError(f"verification failed: {problems}")

        # Atomic within the final filesystem: readers never see a partial episode.
        os.rename(tmp, final_path)
        fsync_dir(self.episodes_dir)

        record.transfer_finished_at = time.time()
        record.transfer_seconds = record.transfer_finished_at - record.transfer_started_at
        if record.transfer_seconds > 0:
            record.transfer_mbps = manifest["total_bytes"] / 1e6 / record.transfer_seconds
        record.final_path = final_path

        if not self.keep_staging_on_success:
            shutil.rmtree(source, ignore_errors=True)
            record.staging_removed = True

        self.log(
            f"  [storage] episode {record.episode_id} -> final "
            f"({manifest['total_bytes']/1e6:.0f} MB in {record.transfer_seconds:.1f}s, "
            f"{record.transfer_mbps or 0:.0f} MB/s, verified)"
        )

    # -- shutdown -------------------------------------------------------------
    def wait_for_transfers(self, timeout: Optional[float] = None) -> bool:
        if self.resolved_mode != MODE_STAGED or self._worker is None:
            return True
        deadline = time.time() + timeout if timeout else None
        while self._pending_transfers() > 0:
            if deadline and time.time() > deadline:
                return False
            if self._fatal:
                return False
            time.sleep(_POLL_S)
        self._queue.join()
        return not self._fatal

    def close(self, timeout: Optional[float] = 3600.0) -> bool:
        ok = self.wait_for_transfers(timeout)
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout=30)
        if self.resolved_mode == MODE_STAGED:
            # Only remove the staging root if nothing is left in it.
            try:
                if os.path.isdir(self.staging_root) and not os.listdir(self.staging_root):
                    os.rmdir(self.staging_root)
            except OSError:
                pass
            shutil.rmtree(self.transfer_tmp_root, ignore_errors=True)
        return ok

    # -- recovery -------------------------------------------------------------
    def scan_for_recovery(self) -> Dict[str, Any]:
        """Classify leftovers from an interrupted run. Never deletes .partial."""
        found = {"ready": [], "partial": [], "transfer_tmp": [], "complete": []}
        staging_parent = os.path.dirname(self.staging_root.rstrip("/"))
        if os.path.isdir(staging_parent):
            for run in sorted(os.listdir(staging_parent)):
                run_path = os.path.join(staging_parent, run)
                if not os.path.isdir(run_path):
                    continue
                for name in sorted(os.listdir(run_path)):
                    path = os.path.join(run_path, name)
                    if name.endswith(".ready"):
                        found["ready"].append(path)
                    elif name.endswith(".partial"):
                        found["partial"].append(path)
        if os.path.isdir(self.transfer_tmp_root):
            for name in sorted(os.listdir(self.transfer_tmp_root)):
                found["transfer_tmp"].append(os.path.join(self.transfer_tmp_root, name))
        if os.path.isdir(self.episodes_dir):
            for name in sorted(os.listdir(self.episodes_dir)):
                path = os.path.join(self.episodes_dir, name)
                if os.path.isdir(path) and not name.endswith(".partial"):
                    found["complete"].append(path)

        self.orphans = [
            {"path": p, "state": "partial",
             "note": "interrupted mid-write; kept for inspection, never auto-deleted"}
            for p in found["partial"]
        ]
        return found

    def recover(self) -> Dict[str, Any]:
        """Re-queue .ready episodes and discard stale .transfer_tmp copies."""
        found = self.scan_for_recovery()
        actions: List[str] = []

        for path in found["transfer_tmp"]:
            # A partial copy is never trusted; the .ready source is still intact.
            shutil.rmtree(path, ignore_errors=True)
            actions.append(f"discarded incomplete transfer copy: {path}")

        requeued = 0
        for path in found["ready"]:
            name = os.path.basename(path)[: -len(".ready")]
            try:
                episode_id = int(name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                actions.append(f"skipped unparseable staging dir: {path}")
                continue
            if os.path.isdir(self.final_episode_path(episode_id)):
                actions.append(
                    f"episode {episode_id} already complete in final storage; "
                    f"leaving staging copy untouched: {path}"
                )
                continue
            if self.resolved_mode != MODE_STAGED:
                actions.append(f"cannot re-queue {path} in {self.resolved_mode} mode")
                continue
            record = TransferRecord(episode_id=episode_id, episode_name=name,
                                    staging_path=path, queued_at=time.time())
            self.records.setdefault(episode_id, record)
            self._queue.put(record)
            requeued += 1
            actions.append(f"re-queued {path}")

        for orphan in self.orphans:
            actions.append(f"ORPHAN (kept): {orphan['path']}")

        return {"found": {k: len(v) for k, v in found.items()},
                "requeued": requeued, "actions": actions, "orphans": self.orphans}

    # -- metadata -------------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        transfers = [r.to_dict() for r in self.records.values()]
        done = [r for r in self.records.values() if r.transfer_mbps]
        return {
            "storage_mode_requested": self.requested_mode,
            "storage_mode_resolved": self.resolved_mode,
            "storage_mode_reason": self.mode_reason,
            "output_root": self.output_root,
            "final_run_dir": self.final_run_dir,
            "staging_root": self.staging_root if self.resolved_mode == MODE_STAGED else None,
            "final_filesystem": self.final_fs,
            "staging_filesystem": self.staging_fs if self.resolved_mode == MODE_STAGED else None,
            "staging_max_gb": self.staging_max_gb,
            "staging_min_free_gb": self.staging_min_free_gb,
            "transfer_queue_size": self.transfer_queue_size,
            "keep_staging_on_success": self.keep_staging_on_success,
            "backpressure_seconds": self.backpressure_seconds,
            "max_queue_depth": self.max_queue_depth,
            "mean_transfer_mbps": (sum(r.transfer_mbps for r in done) / len(done)) if done else None,
            "transfers": transfers,
            "errors": self.errors,
            "orphans": self.orphans,
        }
