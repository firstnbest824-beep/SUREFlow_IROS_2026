"""Fail-closed extraction of the source object phrase from a manipulation instruction."""

from __future__ import annotations

import re


class InstructionTargetError(ValueError):
    """The instruction does not have the deliberately narrow supported grammar."""


def extract_source_phrase(instruction: str) -> str:
    """Return the source phrase in ``pick up X and place it in/on Y``.

    This is intentionally syntax-limited: a caller must not silently turn an
    unsupported instruction into a guessed target label.
    """
    match = re.match(
        r"^\s*(?:pick up|pick|grab)\s+(?:the\s+)?(.+?)\s+and\s+(?:place|put)\s+it\s+(?:in|on|into|onto)\s+.+\s*$",
        instruction,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise InstructionTargetError(
            "unsupported instruction for source extraction; expected "
            f"'pick up X and place it in/on Y': {instruction!r}"
        )
    return match.group(1).strip()
