"""Build a self-contained dashboard.html for one activation collection run.

The dashboard exists so a human can open one file and answer, without reading
any JSON: did the video, the action, the phase and the activations all land on
the same timestep, and does the data actually change over time?

Everything is embedded (images as data URIs, plots as inline PNG) so the file can
be copied anywhere. No network access at view time.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from activation_integrity import (  # noqa: E402
    FAIL,
    PASS,
    STAGE_SPECS,
    WARNING,
    load_episode,
)

STATUS_COLOR = {PASS: "#1a7f37", WARNING: "#9a6700", FAIL: "#cf222e"}
PHASE_COLOR = {"pre_grasp": "#dbeafe", "post_grasp": "#dcfce7", "uncertain": "#fef3c7"}


def _png_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image_data_uri(path: str, max_width: int = 220) -> str:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        if image.width > max_width:
            ratio = max_width / image.width
            image = image.resize((max_width, int(image.height * ratio)))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return ""


def _shade_phases(ax, rows: List[Dict[str, Any]]) -> None:
    """Background bands showing the task phase at each timestep."""
    if not rows:
        return
    start = 0
    current = rows[0].get("task_phase")
    for i, row in enumerate(rows + [None]):
        phase = row.get("task_phase") if row else None
        if phase != current:
            color = PHASE_COLOR.get(str(current))
            if color:
                ax.axvspan(start - 0.5, i - 0.5, color=color, alpha=0.7, zorder=0)
            start, current = i, phase


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def plot_activation_norms(rows: List[Dict[str, Any]], stages: List[str], plots_dir: str) -> str:
    fig, axes = plt.subplots(len(stages), 1, figsize=(11, 1.7 * len(stages)), sharex=True)
    if len(stages) == 1:
        axes = [axes]
    steps = [r["timestep"] for r in rows]
    for ax, stage in zip(axes, stages):
        norms = [
            ((r.get("activations") or {}).get(stage, {}).get("stats") or {}).get("l2_norm")
            for r in rows
        ]
        _shade_phases(ax, rows)
        ax.plot(steps, norms, marker="o", markersize=2.5, linewidth=1.2, color="#0969da")
        ax.set_ylabel(stage, fontsize=7.5, rotation=0, ha="right", va="center")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[-1].set_xlabel("timestep")
    fig.suptitle("Activation L2 norm per timestep (background = task phase)", fontsize=10)
    path = os.path.join(plots_dir, "activation_norms.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return _png_data_uri(fig)


def plot_activation_deltas(rows: List[Dict[str, Any]], stages: List[str], plots_dir: str) -> Tuple[str, Dict[str, float]]:
    """L2 norm of the difference between consecutive timesteps.

    A flat zero line means the same tensor is being written every step -- the
    single most important failure this dashboard has to make visible.
    """
    fig, ax = plt.subplots(figsize=(11, 3.4))
    _shade_phases(ax, rows)
    minima: Dict[str, float] = {}
    for stage in stages:
        deltas: List[float] = []
        previous: Optional[np.ndarray] = None
        for row in rows:
            entry = (row.get("activations") or {}).get(stage, {})
            path = entry.get("path")
            if not path or not os.path.isfile(path):
                deltas.append(np.nan)
                continue
            current = np.load(path, mmap_mode="r")
            current = np.asarray(current, dtype=np.float32)
            if previous is not None and previous.shape == current.shape:
                deltas.append(float(np.linalg.norm(current - previous)))
            else:
                deltas.append(np.nan)
            previous = current
        finite = [d for d in deltas if not np.isnan(d)]
        minima[stage] = float(min(finite)) if finite else float("nan")
        ax.plot([r["timestep"] for r in rows], deltas, marker="o", markersize=2.5,
                linewidth=1.2, label=stage)
    ax.set_xlabel("timestep")
    ax.set_ylabel("||a(t) - a(t-1)||")
    ax.set_title("Consecutive-timestep activation change (a flat zero line means frozen tensors)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.25, linewidth=0.5)
    path = os.path.join(plots_dir, "activation_deltas.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return _png_data_uri(fig), minima


def plot_actions(rows: List[Dict[str, Any]], plots_dir: str) -> str:
    labels = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    steps = [r["timestep"] for r in rows]
    actions = np.array([r["action_applied"] for r in rows])
    for ax in axes:
        _shade_phases(ax, rows)
    for i in range(6):
        axes[0].plot(steps, actions[:, i], marker="o", markersize=2, linewidth=1.1, label=labels[i])
    axes[0].set_ylabel("delta pose")
    axes[0].legend(fontsize=7, ncol=6)
    axes[0].grid(alpha=0.25, linewidth=0.5)
    axes[0].set_title("Applied action per timestep (background = task phase)")

    axes[1].step(steps, actions[:, 6], where="mid", color="#cf222e", linewidth=1.4)
    axes[1].set_ylabel("gripper")
    axes[1].set_xlabel("timestep")
    axes[1].grid(alpha=0.25, linewidth=0.5)

    path = os.path.join(plots_dir, "actions.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    return _png_data_uri(fig)


def token_norm_heatmap(path: str, max_tokens: int = 256) -> Optional[str]:
    """Per-token L2 norm as a compact strip -- comparable by eye across steps.

    Rendering all 1024-4096 channels as an image would be noise; the per-token
    norm keeps the spatial (patch) axis, which is what this study cares about.
    """
    if not path or not os.path.isfile(path):
        return None
    array = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if array.ndim != 3:
        return None
    norms = np.linalg.norm(array[0], axis=-1)
    tokens = norms[:max_tokens]
    # 256 patch tokens -> 16x16 grid restores the spatial layout.
    if tokens.size == 256:
        grid = tokens.reshape(16, 16)
        fig, ax = plt.subplots(figsize=(2.0, 2.0))
        im = ax.imshow(grid, cmap="viridis")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85).ax.tick_params(labelsize=6)
    else:
        fig, ax = plt.subplots(figsize=(3.4, 1.0))
        ax.plot(tokens, linewidth=0.8, color="#0969da")
        ax.set_xticks([]); ax.tick_params(labelsize=6)
    return _png_data_uri(fig)


# -----------------------------------------------------------------------------
# Representative timesteps
# -----------------------------------------------------------------------------
def select_representative_timesteps(rows: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Start, first motion, pre-grasp, gripper change, last."""
    if not rows:
        return []
    n = len(rows)
    picks: List[Tuple[str, int]] = [("start", 0)]

    actions = np.array([r["action_applied"] for r in rows])
    motion = np.linalg.norm(actions[:, :3], axis=1)
    moving = np.where(motion > max(1e-3, float(motion.mean()) * 0.5))[0]
    if moving.size:
        picks.append(("first motion", int(moving[0])))

    grippers = actions[:, 6]
    changes = np.where(np.abs(np.diff(grippers)) > 1e-6)[0]
    if changes.size:
        idx = int(changes[0])
        if idx > 0:
            picks.append(("just before gripper change", idx))
        picks.append(("gripper change", min(idx + 1, n - 1)))

    transitions = [i for i, r in enumerate(rows)
                   if i and r.get("task_phase") != rows[i - 1].get("task_phase")]
    for t in transitions[:1]:
        picks.append(("phase transition", t))

    picks.append(("last", n - 1))

    seen, unique = set(), []
    for label, idx in picks:
        if 0 <= idx < n and idx not in seen:
            seen.add(idx)
            unique.append((label, idx))
    return sorted(unique, key=lambda p: p[1])


# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------
def _esc(value: Any) -> str:
    return html.escape(str(value))


def _badge(status: str) -> str:
    return (f'<span style="background:{STATUS_COLOR.get(status, "#57606a")};color:#fff;'
            f'padding:2px 10px;border-radius:10px;font-weight:600;font-size:12px">{_esc(status)}</span>')


def build_dashboard(run_dir: str) -> str:
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    with open(os.path.join(run_dir, "collection_summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(os.path.join(run_dir, "integrity_report.json"), encoding="utf-8") as handle:
        report = json.load(handle)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    episode_dir = report["episodes"][0]["episode_dir"] if report["episodes"] else None
    episode = load_episode(episode_dir) if episode_dir else {"rows": [], "metadata": {}}
    rows, meta = episode["rows"], episode["metadata"]
    stages = meta.get("saved_stages") or []
    vis_dir = os.path.join(episode_dir, "visualizations") if episode_dir else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    overall = report["overall"]
    parts: List[str] = []
    parts.append(f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>Activation Collection — {_esc(config['run_id'])}</title>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',sans-serif;
      margin:0;padding:24px;background:#f6f8fa;color:#1f2328;line-height:1.5}}
 .wrap{{max-width:1180px;margin:0 auto}}
 .card{{background:#fff;border:1px solid #d1d9e0;border-radius:8px;padding:18px;margin-bottom:18px}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:26px 0 10px;
    border-bottom:2px solid #d1d9e0;padding-bottom:6px}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}}
 th,td{{border:1px solid #d1d9e0;padding:6px 9px;text-align:left;vertical-align:top}}
 th{{background:#f6f8fa;font-weight:600}}
 code{{background:#eff1f3;padding:1px 5px;border-radius:4px;font-size:12px}}
 img{{max-width:100%;border-radius:4px}}
 .kv{{display:grid;grid-template-columns:210px 1fr;gap:5px 14px;font-size:13px}}
 .kv div:nth-child(odd){{color:#57606a}}
 .hero{{font-size:30px;font-weight:700;color:{STATUS_COLOR.get(overall,'#57606a')}}}
 .frames{{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}}
 .frame{{flex:0 0 auto;text-align:center;font-size:11px}}
 .rep{{display:grid;grid-template-columns:230px 230px 1fr;gap:14px;align-items:start;
      border-top:1px solid #eaeef2;padding:14px 0}}
 .note{{background:#fff8c5;border:1px solid #d4a72c;border-radius:6px;padding:10px 12px;font-size:13px}}
</style></head><body><div class="wrap">""")

    # ---- header ----
    parts.append('<div class="card"><h1>OpenVLA Activation Collection — Pilot</h1>')
    parts.append(f'<div class="hero">{_esc(overall)}</div>')
    parts.append('<div class="kv">')
    for label, value in [
        ("run_id", config["run_id"]),
        ("task", f"{config['task_name']}"),
        ("instruction", config["task_instruction"]),
        ("seed", config["seed"]),
        ("episodes", summary["num_episodes"]),
        ("total timesteps", summary["total_timesteps"]),
        ("checkpoint", config["checkpoint_id"]),
        ("revision", config["revision"]),
        ("git commit", config["git_commit"]),
        ("storage dtype", config["save_dtype"]),
        ("total size", f"{summary['total_size_mb']:.1f} MB"),
        ("episode success", meta.get("task_success")),
        ("termination", meta.get("termination_reason")),
        ("activation source", meta.get("activation_source")),
        ("phase source", meta.get("phase_source")),
    ]:
        parts.append(f"<div>{_esc(label)}</div><div><code>{_esc(value)}</code></div>")
    parts.append("</div></div>")

    # ---- integrity ----
    parts.append('<div class="card"><h2>무결성 검사 (A–F)</h2><table>')
    parts.append("<tr><th>ID</th><th>검사</th><th>판정</th><th>상세</th></tr>")
    for episode_report in report["episodes"]:
        for check in episode_report["checks"]:
            parts.append(
                f"<tr><td><b>{_esc(check['check_id'])}</b></td><td>{_esc(check['name'])}</td>"
                f"<td>{_badge(check['status'])}</td><td>{_esc(check['detail'])}</td></tr>"
            )
    parts.append("</table></div>")

    # ---- hook table ----
    parts.append('<div class="card"><h2>Hook별 저장 현황</h2><table>')
    parts.append("<tr><th>stage</th><th>expected shape</th><th>actual shape</th>"
                 "<th>저장 timestep</th><th>calls/step</th><th>NaN</th><th>Inf</th>"
                 "<th>norm 범위</th><th>판정</th></tr>")
    for stage in stages:
        spec = STAGE_SPECS.get(stage)
        entries = [(r.get("activations") or {}).get(stage, {}) for r in rows]
        entries = [e for e in entries if e.get("saved")]
        shapes = sorted({tuple(e["shape"]) for e in entries if e.get("shape")})
        calls = sorted({e.get("call_count") for e in entries})
        nan_total = sum((e.get("stats") or {}).get("nan_count", 0) for e in entries)
        inf_total = sum((e.get("stats") or {}).get("inf_count", 0) for e in entries)
        norms = [(e.get("stats") or {}).get("l2_norm") for e in entries]
        norms = [n for n in norms if n is not None]
        expected = (f"(1, {spec.tokens or '*'}, {spec.hidden or '*'})" if spec else "?")
        ok = (len(shapes) == 1 and nan_total == 0 and inf_total == 0
              and len(entries) == len(rows) and len(set(np.round(norms, 8))) > 1)
        parts.append(
            f"<tr><td><code>{_esc(stage)}</code></td><td><code>{_esc(expected)}</code></td>"
            f"<td><code>{_esc(', '.join(str(list(s)) for s in shapes))}</code></td>"
            f"<td>{len(entries)} / {len(rows)}</td><td>{_esc(calls)}</td>"
            f"<td>{nan_total}</td><td>{inf_total}</td>"
            f"<td>{min(norms):.1f} – {max(norms):.1f}</td>"
            f"<td>{_badge(PASS if ok else WARNING)}</td></tr>"
        )
    parts.append("</table></div>")

    # ---- contact sheet ----
    parts.append('<div class="card"><h2>Episode contact sheet</h2>')
    parts.append('<p style="font-size:12.5px;color:#57606a">각 프레임은 해당 timestep에 '
                 '<b>모델이 실제로 본</b> 관측(obs_pre)이다. 아래 action·phase와 같은 timestep이다.</p>')
    step = max(1, len(rows) // 8)
    parts.append('<div class="frames">')
    for row in rows[::step]:
        obs = row.get("observations") or {}
        av = _image_data_uri(obs.get("agentview", ""), 190)
        eh = _image_data_uri(obs.get("eye_in_hand", ""), 190)
        act = row["action_applied"]
        parts.append(
            f'<div class="frame"><img src="{av}"><br><img src="{eh}"><br>'
            f'<b>t={row["timestep"]}</b><br>{_esc(row.get("task_phase"))}<br>'
            f'[{act[0]:+.2f},{act[1]:+.2f},{act[2]:+.2f}]<br>grip {act[6]:+.0f}</div>'
        )
    parts.append("</div></div>")

    # ---- plots ----
    if rows:
        parts.append('<div class="card"><h2>Activation L2 norm</h2>')
        parts.append(f'<img src="{plot_activation_norms(rows, stages, plots_dir)}"></div>')

        delta_uri, minima = plot_activation_deltas(rows, stages, plots_dir)
        parts.append('<div class="card"><h2>인접 timestep 변화량</h2>')
        frozen = [s for s, v in minima.items() if v == 0.0]
        if frozen:
            parts.append(f'<div class="note">⚠️ 변화량이 0인 stage: {_esc(frozen)} — 동일 tensor 반복 저장 의심</div>')
        parts.append(f'<img src="{delta_uri}"></div>')

        parts.append('<div class="card"><h2>Action</h2>')
        parts.append(f'<img src="{plot_actions(rows, plots_dir)}"></div>')

    # ---- representative timesteps ----
    parts.append('<div class="card"><h2>대표 timestep 상세</h2>')
    for label, idx in select_representative_timesteps(rows):
        row = rows[idx]
        obs = row.get("observations") or {}
        act = row["action_applied"]
        parts.append(f'<div class="rep"><div><b>{_esc(label)} — t={idx}</b><br>'
                     f'<img src="{_image_data_uri(obs.get("agentview", ""), 220)}"><br>'
                     f'<span style="font-size:11px">agentview</span></div>')
        parts.append(f'<div><img src="{_image_data_uri(obs.get("eye_in_hand", ""), 220)}"><br>'
                     f'<span style="font-size:11px">eye-in-hand</span></div>')
        parts.append("<div><table>")
        parts.append(f"<tr><th>phase</th><td>{_esc(row.get('task_phase'))} "
                     f"(relevant: {_esc(row.get('relevant_entity'))})</td></tr>")
        parts.append(f"<tr><th>action</th><td><code>[{', '.join(f'{v:+.3f}' for v in act)}]</code></td></tr>")
        parts.append(f"<tr><th>proprio</th><td><code>"
                     f"{', '.join(f'{v:+.3f}' for v in row.get('proprio_state', []))}</code></td></tr>")
        parts.append(f"<tr><th>obs step (pre→post)</th><td>{_esc(row.get('obs_step_index_pre'))} → "
                     f"{_esc(row.get('obs_step_index_post'))}</td></tr>")
        parts.append("</table><table><tr><th>stage</th><th>mean</th><th>std</th><th>L2</th><th>token norm</th></tr>")
        for stage in stages:
            entry = (row.get("activations") or {}).get(stage, {})
            stats = entry.get("stats") or {}
            heat = token_norm_heatmap(entry.get("path", "")) if stage in (
                "final_vision_dinov2", "final_vision_siglip", "projector_input", "projector_output"
            ) else None
            heat_html = f'<img src="{heat}" style="max-width:110px">' if heat else "—"
            parts.append(
                f"<tr><td><code>{_esc(stage)}</code></td>"
                f"<td>{stats.get('mean', float('nan')):.4f}</td>"
                f"<td>{stats.get('std', float('nan')):.4f}</td>"
                f"<td>{stats.get('l2_norm', float('nan')):.1f}</td>"
                f"<td>{heat_html}</td></tr>"
            )
        parts.append("</table></div></div>")
    parts.append("</div>")

    parts.append('<div class="card" style="font-size:12px;color:#57606a">'
                 'token norm heatmap은 256개 patch token을 16×16 격자로 되돌린 것이다 '
                 '(전 채널을 억지로 이미지화하지 않음). 배경색: '
                 '<span style="background:#dbeafe">pre_grasp</span> '
                 '<span style="background:#dcfce7">post_grasp</span> '
                 '<span style="background:#fef3c7">uncertain</span></div>')
    parts.append("</div></body></html>")

    output = os.path.join(run_dir, "dashboard.html")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    return output


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build dashboard.html for a collection run.")
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    print(build_dashboard(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
