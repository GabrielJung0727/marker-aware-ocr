from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Box = Tuple[float, float, float, float]


@dataclass
class RegionItem:
    kind: str
    box: Box
    payload: dict


def line_cluster(boxes: Sequence[Box], line_thresh: float = 0.5) -> List[List[Box]]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    lines: List[List[Box]] = []
    current: List[Box] = []
    current_y = None
    current_h = None

    for box in sorted_boxes:
        y_center = (box[1] + box[3]) / 2.0
        height = max(1.0, box[3] - box[1])

        if current_y is None:
            current = [box]
            current_y = y_center
            current_h = height
            continue

        threshold = (current_h or height) * line_thresh
        if abs(y_center - current_y) <= threshold:
            current.append(box)
            current_y = (current_y + y_center) / 2.0
            current_h = (current_h + height) / 2.0
        else:
            lines.append(sorted(current, key=lambda x: x[0]))
            current = [box]
            current_y = y_center
            current_h = height

    if current:
        lines.append(sorted(current, key=lambda x: x[0]))
    return lines


def sort_boxes(boxes: Sequence[Box], line_thresh: float = 0.5) -> List[Box]:
    ordered: List[Box] = []
    for line in line_cluster(boxes, line_thresh=line_thresh):
        ordered.extend(line)
    return ordered


def sort_regions(regions: Iterable[RegionItem], line_thresh: float = 0.5) -> List[RegionItem]:
    region_list = list(regions)
    if not region_list:
        return []

    indexed = {item.box: [] for item in region_list}
    for item in region_list:
        indexed[item.box].append(item)

    ordered_boxes = sort_boxes([item.box for item in region_list], line_thresh=line_thresh)
    ordered_regions: List[RegionItem] = []
    for box in ordered_boxes:
        bucket = indexed.get(box, [])
        if not bucket:
            continue
        ordered_regions.append(bucket.pop(0))
    return ordered_regions
