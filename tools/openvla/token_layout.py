"""Method-neutral OpenVLA token and image-patch layout helpers.

Transformer-relevance methods need to identify the visual-token slice inserted
into the language-model prefill sequence, map an instruction phrase to its text
tokens, and map a visual-token index back to a model-image patch.  This module
contains only that structural bookkeeping; it makes no attribution, cosine, or
simulator call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


class TokenLayoutError(RuntimeError):
    """The observed OpenVLA multimodal token layout is not understood safely."""


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
            raise TokenLayoutError("input token index out of range: {}".format(input_index))
        # OpenVLA prefill is BOS + visual tokens + remaining original text tokens.
        return input_index if input_index == 0 else input_index + self.num_visual_tokens

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class TargetTokenSpan:
    """A uniquely resolved instruction phrase in prompt and LLM coordinates."""

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


def _as_input_ids(input_ids: Any) -> List[int]:
    array = np.asarray(input_ids.detach().cpu() if hasattr(input_ids, "detach") else input_ids)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise TokenLayoutError("expected batch size 1 input_ids, got {}".format(array.shape))
        array = array[0]
    if array.ndim != 1:
        raise TokenLayoutError("expected one token sequence, got {}".format(array.shape))
    return [int(value) for value in array.tolist()]


def resolve_multimodal_layout(
    input_ids: Any, llm_prefill_hidden: Any, num_visual_tokens: int,
) -> MultimodalTokenLayout:
    """Resolve the measured ``BOS + visual + text`` layout or fail closed."""
    ids = _as_input_ids(input_ids)
    shape = tuple(int(value) for value in np.shape(llm_prefill_hidden))
    if len(shape) != 3 or shape[0] != 1:
        raise TokenLayoutError("expected LLM prefill [1, sequence, hidden], got {}".format(shape))
    num_visual_tokens = int(num_visual_tokens)
    if num_visual_tokens <= 0:
        raise TokenLayoutError("projector visual token count must be positive")
    expected = len(ids) + num_visual_tokens
    if shape[1] != expected:
        raise TokenLayoutError(
            "LLM prefill sequence does not match original tokens plus inserted visual tokens: "
            "got {}, expected {} + {} = {}".format(shape[1], len(ids), num_visual_tokens, expected)
        )
    return MultimodalTokenLayout(len(ids), shape[1], num_visual_tokens, 1, 1 + num_visual_tokens)


def _encoding_value(encoding: Any, key: str) -> Any:
    try:
        return encoding[key]
    except (KeyError, TypeError):
        return getattr(encoding, key, None)


def _find_unique_char_span(prompt: str, phrase: str) -> Tuple[int, int]:
    if not phrase.strip():
        raise TokenLayoutError("target phrase is empty")
    starts = []
    offset = 0
    while True:
        index = prompt.casefold().find(phrase.casefold(), offset)
        if index < 0:
            break
        starts.append(index)
        offset = index + 1
    if len(starts) != 1:
        raise TokenLayoutError("target phrase must occur exactly once; found {}".format(len(starts)))
    return starts[0], starts[0] + len(phrase)


def locate_target_token_span(
    tokenizer: Any, prompt: str, target_phrase: str, input_ids: Any, layout: MultimodalTokenLayout,
) -> TargetTokenSpan:
    """Resolve one instruction phrase by offsets, failing rather than guessing."""
    ids = _as_input_ids(input_ids)
    start, end = _find_unique_char_span(prompt, target_phrase)
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
        encoded_ids = _as_input_ids(_encoding_value(encoded, "input_ids"))
        offsets = np.asarray(_encoding_value(encoded, "offset_mapping"))
        if offsets.ndim == 3:
            offsets = offsets[0]
        if encoded_ids != ids[:len(encoded_ids)] or offsets.shape != (len(encoded_ids), 2):
            raise TokenLayoutError("tokenizer offsets do not align with prepared input_ids")
        positions = tuple(
            index for index, (token_start, token_end) in enumerate(offsets.tolist())
            if int(token_end) > start and int(token_start) < end
        )
        if not positions:
            raise TokenLayoutError("offset mapping produced no tokens for target phrase")
    except (NotImplementedError, TypeError, ValueError, KeyError, AttributeError, TokenLayoutError) as exc:
        raise TokenLayoutError("tokenizer offset mapping is required for unambiguous target-token tracking") from exc
    token_ids = tuple(ids[index] for index in positions)
    decoded = tuple(str(token) for token in tokenizer.convert_ids_to_tokens(list(token_ids))) if hasattr(tokenizer, "convert_ids_to_tokens") else tuple(str(token) for token in token_ids)
    return TargetTokenSpan(
        target_phrase, (start, end), positions,
        tuple(layout.llm_position_for_input_token(index) for index in positions),
        token_ids, decoded, "offset_mapping",
    )


def patch_index_to_uv(
    patch_index: int, num_visual_tokens: int, image_shape: Tuple[int, int] = (224, 224),
) -> Tuple[float, float]:
    """Return the center of a square-grid visual token's image patch."""
    side = isqrt(int(num_visual_tokens))
    if side * side != int(num_visual_tokens) or not 0 <= patch_index < int(num_visual_tokens):
        raise TokenLayoutError("invalid square-grid visual patch index")
    height, width = int(image_shape[0]), int(image_shape[1])
    row, col = divmod(int(patch_index), side)
    return ((col + 0.5) * width / side, (row + 0.5) * height / side)
