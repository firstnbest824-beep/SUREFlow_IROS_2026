"""Call-order-preserving forward / forward-pre hooks for OpenVLA probing.

Every probe point here is the tensor the model *actually consumes*, verified
against the executed ``modeling_prismatic.py`` rather than assumed from module
names. Four problems with earlier versions of this file are fixed.

1. ``language_model.lm_head`` produces *vocabulary logits*, not the hidden state
   that drives action generation. Its output is therefore recorded under the
   functional stage ``lm_head_logits`` and never called an "action-head input".
   The representation the research plan actually needs -- the hidden state fed
   *into* ``lm_head`` -- is captured with a ``forward_pre`` hook and stored as
   ``pre_action_hidden``.

2. ``predict_action`` calls ``generate``, so every module inside the language
   model fires once per generated token. The old hook overwrote a single tensor
   and kept only the last call. Every hook here appends to a list, so the call
   order (prompt prefill first, then one call per autoregressive step) is
   preserved and reported. The number of calls is *measured*, never assumed to
   equal the 7-dimensional action.

3. The vision hooks pointed at the *last* transformer block of each featurizer
   (``blocks.23`` / ``blocks.26``). The model never uses those: each featurizer's
   ``forward`` is monkey-patched to return the **second-to-last** block's patch
   tokens. The last block still runs -- so the hook fired and produced
   plausible-looking data -- but its output is discarded. Those tensors differed
   from the consumed ones by rel_L2 1.61 (DINOv2) / 1.17 (SigLIP), cos 0.59 /
   0.66 on the real checkpoint. The hooks now sit on the featurizer modules, so
   they capture the prefix-stripped tensors that reach the projector.

4. ``projector.fc3`` and ``projector`` were both hooked, but for the fused
   backbone ``fc3`` is the final layer and its output *is* the projector's
   return value -- the same tensor object. Only ``projector`` is hooked now, and
   a ``forward_pre`` hook records the projector's input, which nothing captured
   before and which is the "before" endpoint for measuring projector-induced
   spatial-information loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# module path -> (functional stage, readout description)
#
# Vision: hook the two featurizer *modules*, not a block inside them.
# `PrismaticVisionBackbone.__init__` monkey-patches each featurizer's `forward`
# to `get_intermediate_layers(n={len(blocks) - 2})`, so the model consumes the
# **second-to-last** block (DINOv2 block 22 of 24, SigLIP block 25 of 27) with
# DINOv2's 5 prefix tokens (CLS + 4 registers) stripped and no final norm.
# Hooking `blocks.23` / `blocks.26` -- as this file used to -- captured tensors
# the model computes but then discards; on the real checkpoint those differ from
# the consumed ones by rel_L2 1.61 / 1.17 (cos 0.59 / 0.66), i.e. a different
# representation entirely. Hooking the module returns exactly the tensor handed
# to the projector: [1, 256, 1024] and [1, 256, 1152].
FORWARD_PROBE_TARGETS: Dict[str, Tuple[str, str]] = {
    "vision_backbone.featurizer": (
        "final_vision_dinov2",
        "DINOv2 features the projector consumes (penultimate block, prefix tokens stripped)",
    ),
    "vision_backbone.fused_featurizer": (
        "final_vision_siglip",
        "SigLIP features the projector consumes (penultimate block)",
    ),
    # `projector.fc3` is deliberately NOT hooked: for the fused backbone `fc3` is
    # the last layer and its output IS the projector's return value, so the two
    # hooks received the same tensor object (identical `data_ptr`, bitwise-equal
    # `.npy` files). Keeping both wasted 4.19 MB/timestep for zero information.
    "projector": ("projector_output", "full projector output"),
    "language_model.model.layers.0": ("llm_early", "first LLM layer hidden state"),
    "language_model.model.layers.15": ("llm_middle", "middle LLM layer hidden state"),
    "language_model.model.layers.31": ("llm_late", "late LLM layer hidden state"),
    # Vocabulary logits. NOT the action-head input representation.
    "language_model.lm_head": ("lm_head_logits", "vocabulary logits over the tokenizer"),
}

# module path -> (functional stage, readout description) captured with forward_pre
PRE_FORWARD_PROBE_TARGETS: Dict[str, Tuple[str, str]] = {
    # The concatenated vision features actually entering the projector:
    # torch.cat([dinov2_features, siglip_features], dim=2) -> [1, 256, 2176].
    # This is the "before" endpoint for measuring what the projector does to
    # spatial information; without it only the "after" side was recorded.
    "projector": (
        "projector_input",
        "concatenated DINOv2+SigLIP features entering the projector "
        "(channel-wise concat, dim=2)",
    ),
    "language_model.lm_head": (
        "pre_action_hidden",
        "hidden state entering lm_head, i.e. the representation immediately "
        "before action-token logits are produced",
    ),
}

# Stages whose modules run once per forward pass (no autoregressive repetition).
SINGLE_CALL_STAGES = {
    "final_vision_dinov2",
    "final_vision_siglip",
    "projector_input",
    "projector_output",
}


@dataclass
class CallRecord:
    """One invocation of a hooked module."""

    call_index: int
    shape: List[int]
    seq_len: Optional[int]
    hidden_dim: Optional[int]
    dtype: str
    device: str
    has_nan: bool
    has_inf: bool
    call_type: str  # "prompt_prefill" | "autoregressive_generation" | "single"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_index": self.call_index,
            "shape": self.shape,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "dtype": self.dtype,
            "device": self.device,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "call_type": self.call_type,
        }


@dataclass
class ProbeStream:
    """All calls captured for one (module_path, hook_type) pair."""

    module_path: str
    functional_stage: str
    readout_description: str
    hook_type: str  # "forward" | "forward_pre"
    tensors: List[torch.Tensor] = field(default_factory=list)
    records: List[CallRecord] = field(default_factory=list)
    non_tensor_calls: int = 0

    @property
    def call_count(self) -> int:
        return len(self.records) + self.non_tensor_calls

    def last_token_stack(self) -> Optional[np.ndarray]:
        """``[num_calls, hidden_dim]`` -- the last-token readout of every call."""
        if not self.tensors:
            return None
        rows = []
        for tensor in self.tensors:
            array = tensor.to(torch.float32).numpy()
            if array.ndim == 3:
                rows.append(array[0, -1, :])
            elif array.ndim == 2:
                rows.append(array[-1, :])
            else:
                return None
        return np.stack(rows, axis=0)

    def prefill_tensor(self) -> Optional[np.ndarray]:
        """Full-sequence tensor of the first call (contains the visual tokens)."""
        if not self.tensors:
            return None
        return self.tensors[0].to(torch.float32).numpy()


def _classify_call(tensor: torch.Tensor, functional_stage: str, call_index: int) -> str:
    if functional_stage in SINGLE_CALL_STAGES:
        return "single"
    seq_len = int(tensor.shape[1]) if tensor.ndim >= 3 else None
    if call_index == 0 and (seq_len is None or seq_len > 1):
        return "prompt_prefill"
    return "autoregressive_generation"


class ProbeHookManager:
    """Registers every probe hook and keeps per-call tensors in call order."""

    def __init__(
        self,
        vla: Any,
        forward_targets: Optional[Dict[str, Tuple[str, str]]] = None,
        pre_forward_targets: Optional[Dict[str, Tuple[str, str]]] = None,
        keep_on_cpu: bool = True,
    ) -> None:
        self.vla = vla
        self.forward_targets = dict(
            FORWARD_PROBE_TARGETS if forward_targets is None else forward_targets
        )
        self.pre_forward_targets = dict(
            PRE_FORWARD_PROBE_TARGETS if pre_forward_targets is None else pre_forward_targets
        )
        self.keep_on_cpu = keep_on_cpu

        self.streams: Dict[str, ProbeStream] = {}
        self.missing: List[str] = []
        self._handles: List[Any] = []
        self._register()

    # -- registration ---------------------------------------------------------
    def _get_module(self, path: str) -> Optional[Any]:
        module: Any = self.vla
        for part in path.split("."):
            if not hasattr(module, part):
                return None
            module = getattr(module, part)
        return module

    @staticmethod
    def stream_key(module_path: str, hook_type: str) -> str:
        return f"{module_path}::{hook_type}"

    def _register(self) -> None:
        for module_path, (stage, description) in self.forward_targets.items():
            module = self._get_module(module_path)
            if module is None:
                self.missing.append(module_path)
                continue
            key = self.stream_key(module_path, "forward")
            self.streams[key] = ProbeStream(module_path, stage, description, "forward")
            self._handles.append(module.register_forward_hook(self._forward_hook(key, stage)))

        for module_path, (stage, description) in self.pre_forward_targets.items():
            module = self._get_module(module_path)
            if module is None:
                if module_path not in self.missing:
                    self.missing.append(module_path)
                continue
            key = self.stream_key(module_path, "forward_pre")
            self.streams[key] = ProbeStream(module_path, stage, description, "forward_pre")
            self._handles.append(
                module.register_forward_pre_hook(self._forward_pre_hook(key, stage))
            )

    # -- hooks ----------------------------------------------------------------
    def _store(self, key: str, stage: str, tensor: Any) -> None:
        stream = self.streams[key]
        if isinstance(tensor, tuple):
            tensor = tensor[0] if tensor else None
        if not isinstance(tensor, torch.Tensor):
            stream.non_tensor_calls += 1
            return

        call_index = len(stream.records)
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        has_nan = bool(torch.isnan(detached).any().item())
        has_inf = bool((~finite & ~torch.isnan(detached)).any().item())

        stored = detached.to("cpu") if self.keep_on_cpu else detached.clone()
        stream.tensors.append(stored)
        stream.records.append(
            CallRecord(
                call_index=call_index,
                shape=[int(value) for value in detached.shape],
                seq_len=int(detached.shape[1]) if detached.ndim >= 3 else None,
                hidden_dim=int(detached.shape[-1]) if detached.ndim >= 1 else None,
                dtype=str(detached.dtype),
                device=str(detached.device),
                has_nan=has_nan,
                has_inf=has_inf,
                call_type=_classify_call(detached, stage, call_index),
            )
        )

    def _forward_hook(self, key: str, stage: str):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            self._store(key, stage, output)

        return hook

    def _forward_pre_hook(self, key: str, stage: str):
        def hook(module: Any, inputs: Any) -> None:
            self._store(key, stage, inputs)

        return hook

    # -- lifecycle ------------------------------------------------------------
    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        for stream in self.streams.values():
            stream.tensors.clear()
            stream.records.clear()
            stream.non_tensor_calls = 0

    # -- reporting ------------------------------------------------------------
    def call_counts(self) -> Dict[str, int]:
        return {key: stream.call_count for key, stream in self.streams.items()}

    def stream_by_stage(self, functional_stage: str) -> Optional[ProbeStream]:
        for stream in self.streams.values():
            if stream.functional_stage == functional_stage:
                return stream
        return None

    def nonfinite_stages(self) -> List[str]:
        bad: List[str] = []
        for stream in self.streams.values():
            if any(record.has_nan or record.has_inf for record in stream.records):
                bad.append(f"{stream.functional_stage} ({stream.module_path})")
        return bad
