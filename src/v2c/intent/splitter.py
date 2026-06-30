"""
Transcript splitter — breaks a multi-command voice utterance into segments.

When the user says:
    "add function first, next line, add function second, next line, next line"

Whisper returns it as a single string. This module splits it on every
"next line / new line / enter / line break" occurrence and returns a list of
(command_text, newlines_after) pairs so the pipeline can dispatch each
command independently and insert the right number of newlines between them.

Examples
--------
>>> split("add function first next line add function second")
[("add function first", 1), ("add function second", 0)]

>>> split("import os next line next line add class Foo")
[("import os", 2), ("add class Foo", 0)]

>>> split("hello world")
[("hello world", 0)]
"""

from __future__ import annotations

import re

# ── Newline marker pattern ────────────────────────────────────────────────────
# Matches "next line", "new line", "newline", "enter", "line break", "blank line"
# optionally preceded by punctuation/spaces, with an optional count word.
_NEWLINE_RE = re.compile(
    r"[,\.\s]*"                                           # optional leading punctuation
    r"(?:"
    r"(?P<count_word>two|three|four|five|twice|double|triple)"  # "three new lines"
    r"\s+)?"
    r"(?:new\s+line|next\s+line|newline|line\s+break|blank\s+line|enter)"
    r"(?:\s+(?P<count_word2>twice|two\s+times?|x\s*2))?"  # "new line twice"
    r"[,\.\s]*",
    re.I,
)

_WORD_TO_COUNT: dict[str, int] = {
    "twice": 2, "two": 2, "double": 2,
    "three": 3, "triple": 3,
    "four": 4,
    "five": 5,
}


def _parse_count(m: re.Match) -> int:
    """Extract the newline count from a regex match."""
    w1 = (m.group("count_word") or "").lower()
    w2 = (m.group("count_word2") or "").lower()
    word = w1 or w2
    return _WORD_TO_COUNT.get(word, 1)


def split(transcript: str) -> list[tuple[str, int]]:
    """
    Split *transcript* on newline-marker words and return a list of
    ``(command_text, newlines_after)`` pairs.

    - ``command_text`` is stripped; empty segments are dropped.
    - ``newlines_after`` is the number of newlines that should follow the
      command (0 for the last segment unless the utterance ends with a marker).
    - If there are no newline markers, returns ``[(transcript.strip(), 0)]``.

    Parameters
    ----------
    transcript : str
        The full ASR output to split.

    Returns
    -------
    list of (command_text, newlines_after) tuples
    """
    transcript = transcript.strip()
    if not transcript:
        return []

    segments: list[tuple[str, int]] = []
    pos = 0

    for m in _NEWLINE_RE.finditer(transcript):
        # Text before this marker
        segment = transcript[pos : m.start()].strip(" ,.")
        count = _parse_count(m)
        # Only record the segment if it has content, or if it's a bare newline
        # between two newline markers (i.e. consecutive "next line next line")
        if segment:
            segments.append((segment, count))
        elif segments:
            # Consecutive newline markers — add to the previous segment's count
            prev_text, prev_count = segments[-1]
            segments[-1] = (prev_text, prev_count + count)
        else:
            # Leading newline marker before any command — skip
            pass
        pos = m.end()

    # Remainder after the last marker
    remainder = transcript[pos:].strip(" ,.")
    if remainder:
        segments.append((remainder, 0))
    elif not segments:
        # No markers found at all
        segments.append((transcript.strip(), 0))

    return segments
