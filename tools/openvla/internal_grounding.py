"""Parameter-free, hidden-state similarity maps for frozen OpenVLA.

Prediction functions in this module deliberately accept only prompt/token and
OpenVLA activation data.  They neither import simulator code nor accept masks,
object identities, depth, or environment handles.  A high score is a
zero-shot hidden-state similarity readout, *not* an attention map and not proof
that a causal visual token itself is conditioned on later instruction text.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


class InternalGroundingError(RuntimeError):
    """A required structural assertion for internal grounding did not hold."""


@dataclass(frozen=True)
class MultimodalTokenLayout:
    """Mapping from original prompt tokens to LLM prefill sequence positions."""

    input_token_count: int
    llm_sequence_length: int
    num_visual_tokens: int
    visual_start: int
    visual_end: int

    def llm_position_for_input_token(self, input_index: int) -> int:
        if input_index < 0 or input_index >= self.input_token_count:
            raise InternalGroundingError(f"input token index out of range: {input_index}")
        # OpenVLA's prefill layout is verified as BOS + visual tokens + remaining
        # original text tokens. Original input token 0 (BOS) stays at position 0.
        return input_index if input_index == 0 else input_index + self.num_visual_tokens

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TargetTokenSpan:
    phrase: str
    prompt_char_span: Tuple[int, int]
    input_token_indices: Tuple[int, ...]
    llm_token_positions: Tuple[int, ...]
    token_ids: Tuple[int, ...]
    decoded_tokens: Tuple[str, ...]
    resolution_method: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for key in ("prompt_char_span", "input_token_indices", "llm_token_positions", "token_ids", "decoded_tokens"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class GroundingMapResult:
    functional_stage: str
    readout_type: str
    target_phrase: str
    target_token_positions: Tuple[int, ...]
    num_visual_tokens: int
    patch_grid_shape: Tuple[int, int]
    image_shape: Tuple[int, int]
    patch_cell_size: Tuple[float, float]
    predicted_patch_index: int
    predicted_patch_row: int
    predicted_patch_col: int
    predicted_uv_model_input: Tuple[float, float]
    top_k_patch_indices: Tuple[int, ...]
    top_k_scores: Tuple[float, ...]
    scores: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "functional_stage": self.functional_stage,
            "readout_type": self.readout_type,
            "target_phrase": self.target_phrase,
            "target_token_positions": list(self.target_token_positions),
            "num_visual_tokens": self.num_visual_tokens,
            "patch_grid_shape": list(self.patch_grid_shape),
            "image_shape": list(self.image_shape),
            "patch_cell_size": list(self.patch_cell_size),
            "predicted_patch_index": self.predicted_patch_index,
            "predicted_patch_row": self.predicted_patch_row,
            "predicted_patch_col": self.predicted_patch_col,
            "predicted_uv_model_input": list(self.predicted_uv_model_input),
            "top_k_patch_indices": list(self.top_k_patch_indices),
            "top_k_scores": list(self.top_k_scores),
            "prediction_note": "patch-center quantized zero-shot hidden-state similarity map",
        }


def _as_input_ids(input_ids: Any) -> List[int]:
    array = np.asarray(input_ids.detach().cpu() if hasattr(input_ids, "detach") else input_ids)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise InternalGroundingError(f"expected batch size 1 input_ids, got {array.shape}")
        array = array[0]
    if array.ndim != 1:
        raise InternalGroundingError(f"expected one token sequence, got {array.shape}")
    return [int(value) for value in array.tolist()]


def _prefill_shape(hidden: Any) -> Tuple[int, int, int]:
    shape = tuple(int(value) for value in np.shape(hidden))
    if len(shape) != 3 or shape[0] != 1:
        raise InternalGroundingError(f"expected LLM prefill [1, sequence, hidden], got {shape}")
    return shape


def resolve_multimodal_layout(input_ids: Any, llm_prefill_hidden: Any, num_visual_tokens: int) -> MultimodalTokenLayout:
    """Resolve the verified ``BOS + visual + text`` prefill layout or fail."""
    ids = _as_input_ids(input_ids)
    _, sequence_length, _ = _prefill_shape(llm_prefill_hidden)
    num_visual_tokens = int(num_visual_tokens)
    if num_visual_tokens <= 0:
        raise InternalGroundingError(f"projector visual token count must be positive, got {num_visual_tokens}")
    expected = len(ids) + num_visual_tokens
    if sequence_length != expected:
        raise InternalGroundingError(
            "LLM prefill sequence does not match original tokens plus inserted visual tokens: "
            f"got {sequence_length}, expected {len(ids)} + {num_visual_tokens} = {expected}"
        )
    return MultimodalTokenLayout(
        input_token_count=len(ids), llm_sequence_length=sequence_length,
        num_visual_tokens=num_visual_tokens, visual_start=1, visual_end=1 + num_visual_tokens,
    )


def _find_unique_char_span(prompt: str, phrase: str) -> Tuple[int, int]:
    if not phrase.strip():
        raise InternalGroundingError("target phrase is empty")
    prompt_folded, phrase_folded = prompt.casefold(), phrase.casefold()
    starts: List[int] = []
    offset = 0
    while True:
        index = prompt_folded.find(phrase_folded, offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + 1
    if len(starts) != 1:
        raise InternalGroundingError(
            f"target phrase must occur exactly once in exact prompt; {phrase!r} occurred {len(starts)} times"
        )
    return starts[0], starts[0] + len(phrase)


def _encoding_value(encoding: Any, key: str) -> Any:
    try:
        return encoding[key]
    except (KeyError, TypeError):
        return getattr(encoding, key, None)


def _tokenize_ids(tokenizer: Any, text: str, **kwargs: Any) -> List[int]:
    encoded = tokenizer(text, **kwargs)
    return _as_input_ids(_encoding_value(encoded, "input_ids"))


def _subsequence_positions(haystack: Sequence[int], needle: Sequence[int]) -> List[Tuple[int, ...]]:
    if not needle:
        return []
    width = len(needle)
    return [tuple(range(index, index + width)) for index in range(len(haystack) - width + 1)
            if list(haystack[index:index + width]) == list(needle)]


def locate_target_token_span(
    tokenizer: Any, prompt: str, target_phrase: str, input_ids: Any, layout: MultimodalTokenLayout,
) -> TargetTokenSpan:
    """Resolve a unique target phrase span with tokenizer offsets or safe fallback."""
    ids = _as_input_ids(input_ids)
    char_span = _find_unique_char_span(prompt, target_phrase)
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        encoded_ids = _as_input_ids(_encoding_value(encoded, "input_ids"))
        offsets = _encoding_value(encoded, "offset_mapping")
        offsets_array = np.asarray(offsets)
        if offsets_array.ndim == 3:
            offsets_array = offsets_array[0]
        if encoded_ids != ids[:len(encoded_ids)] or offsets_array.shape != (len(encoded_ids), 2):
            raise InternalGroundingError("tokenizer offsets do not align with exact prepared input_ids")
        start, end = char_span
        positions = tuple(
            index for index, (token_start, token_end) in enumerate(offsets_array.tolist())
            if int(token_end) > start and int(token_start) < end
        )
        if not positions:
            raise InternalGroundingError("offset mapping produced no tokens for target phrase")
        method = "offset_mapping"
    except (NotImplementedError, TypeError, ValueError, KeyError, AttributeError, InternalGroundingError):
        # Some remote tokenizers do not expose offsets. Match both plain and
        # leading-space tokenizations, then require one unique input span.
        candidates = {
            tuple(_tokenize_ids(tokenizer, target_phrase, add_special_tokens=False)),
            tuple(_tokenize_ids(tokenizer, " " + target_phrase, add_special_tokens=False)),
        }
        matches = {match for candidate in candidates for match in _subsequence_positions(ids, candidate)}
        if len(matches) != 1:
            raise InternalGroundingError(
                "token-id fallback requires exactly one target span; "
                f"found {len(matches)} for {target_phrase!r}"
            )
        positions = next(iter(matches))
        method = "token_id_subsequence"
    llm_positions = tuple(layout.llm_position_for_input_token(index) for index in positions)
    token_ids = tuple(ids[index] for index in positions)
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        decoded = tuple(str(token) for token in tokenizer.convert_ids_to_tokens(list(token_ids)))
    else:
        decoded = tuple(str(token) for token in token_ids)
    return TargetTokenSpan(target_phrase, char_span, positions, llm_positions, token_ids, decoded, method)


def patch_index_to_uv(patch_index: int, num_visual_tokens: int, image_shape: Tuple[int, int] = (224, 224)) -> Tuple[float, float]:
    """Return the center of a square-grid visual patch in model-input pixels."""
    side = isqrt(int(num_visual_tokens))
    if side * side != int(num_visual_tokens):
        raise InternalGroundingError(f"visual token count {num_visual_tokens} is not a square patch grid")
    if patch_index < 0 or patch_index >= int(num_visual_tokens):
        raise InternalGroundingError(f"patch index out of range: {patch_index}")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    row, col = divmod(int(patch_index), side)
    return ((col + 0.5) * width / side, (row + 0.5) * height / side)


def compute_cosine_grounding_map(
    functional_stage: str,
    llm_prefill_hidden: Any,
    layout: MultimodalTokenLayout,
    target_span: TargetTokenSpan,
    readout_type: str = "target_last_token",
    image_shape: Tuple[int, int] = (224, 224),
    top_k: int = 5,
) -> GroundingMapResult:
    """Score visual patches against a target-text representation in LLM space."""
    hidden = np.asarray(llm_prefill_hidden.detach().cpu() if hasattr(llm_prefill_hidden, "detach") else llm_prefill_hidden,
                        dtype=np.float32)
    _, length, _ = _prefill_shape(hidden)
    if length != layout.llm_sequence_length:
        raise InternalGroundingError("prefill hidden sequence length disagrees with resolved token layout")
    visual = hidden[0, layout.visual_start:layout.visual_end]
    phrase_hidden = hidden[0, list(target_span.llm_token_positions)]
    if readout_type == "target_last_token":
        target = phrase_hidden[-1]
    elif readout_type == "target_token_mean":
        target = phrase_hidden.mean(axis=0)
    else:
        raise InternalGroundingError(f"unsupported readout type: {readout_type!r}")
    visual_norm = np.linalg.norm(visual, axis=1, keepdims=True)
    target_norm = float(np.linalg.norm(target))
    if not np.all(np.isfinite(visual_norm)) or not np.isfinite(target_norm) or target_norm == 0 or np.any(visual_norm == 0):
        raise InternalGroundingError("cannot cosine-normalize non-finite or zero hidden states")
    scores = (visual / visual_norm) @ (target / target_norm)
    side = isqrt(layout.num_visual_tokens)
    if side * side != layout.num_visual_tokens:
        raise InternalGroundingError(f"visual token count {layout.num_visual_tokens} is not square")
    predicted = int(np.argmax(scores))
    row, col = divmod(predicted, side)
    k = min(max(int(top_k), 1), layout.num_visual_tokens)
    top_indices = np.argsort(-scores, kind="stable")[:k]
    height, width = image_shape
    return GroundingMapResult(
        functional_stage=functional_stage, readout_type=readout_type, target_phrase=target_span.phrase,
        target_token_positions=target_span.llm_token_positions, num_visual_tokens=layout.num_visual_tokens,
        patch_grid_shape=(side, side), image_shape=(int(height), int(width)),
        patch_cell_size=(float(width) / side, float(height) / side), predicted_patch_index=predicted,
        predicted_patch_row=row, predicted_patch_col=col,
        predicted_uv_model_input=patch_index_to_uv(predicted, layout.num_visual_tokens, image_shape),
        top_k_patch_indices=tuple(int(value) for value in top_indices),
        top_k_scores=tuple(float(scores[value]) for value in top_indices), scores=np.asarray(scores, dtype=np.float32),
    )
