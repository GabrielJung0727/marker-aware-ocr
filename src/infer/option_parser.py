from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

from .mark_resolver import resolve_marks

Box = Tuple[float, float, float, float]


def box_center(box: Box) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def point_in_box(x: float, y: float, box: Box) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


@dataclass
class OptionResult:
    idx: int
    text: str
    mark: str
    markers: List[str]
    reason: str


def parse_option_blocks(
    option_boxes: Sequence[Box],
    option_texts: Sequence[str],
    marker_boxes: Sequence[Box],
    marker_labels: Sequence[str],
    *,
    priority: Sequence[str],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []

    for idx, option_box in enumerate(option_boxes, start=1):
        text = option_texts[idx - 1] if idx - 1 < len(option_texts) else ''
        assigned_markers: List[str] = []

        for marker_box, marker_label in zip(marker_boxes, marker_labels):
            cx, cy = box_center(marker_box)
            if point_in_box(cx, cy, option_box):
                assigned_markers.append(marker_label)

        resolved = resolve_marks(assigned_markers, priority)
        results.append(
            asdict(
                OptionResult(
                    idx=idx,
                    text=text,
                    mark=resolved.final_mark,
                    markers=resolved.all_marks,
                    reason=resolved.reason,
                )
            )
        )

    return results
