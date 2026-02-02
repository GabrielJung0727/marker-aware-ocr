from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class MarkResolution:
    final_mark: str
    all_marks: List[str]
    reason: str


def normalize_mark(label: str) -> str:
    label = label.upper()
    if label.startswith('MARKER_'):
        label = label.split('MARKER_', 1)[1]
    return label


def resolve_marks(marker_labels: Sequence[str], priority: Sequence[str]) -> MarkResolution:
    normalized = [normalize_mark(m) for m in marker_labels if m]
    if not normalized:
        return MarkResolution(final_mark='NONE', all_marks=[], reason='no_marker')

    priority_norm = [normalize_mark(p) for p in priority]
    for item in priority_norm:
        if item in normalized:
            return MarkResolution(
                final_mark=item,
                all_marks=normalized,
                reason=f'priority_match:{item}',
            )

    return MarkResolution(
        final_mark=normalized[-1],
        all_marks=normalized,
        reason='fallback_last_marker',
    )
