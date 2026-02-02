from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .reading_order import RegionItem, sort_regions

Box = Tuple[float, float, float, float]


@dataclass
class FormulaSegment:
    token: str
    box: Box


@dataclass
class FormulaComposeResult:
    merged_text: str
    formula_map: Dict[str, Box]


def compose_with_formula_placeholders(
    text_segments: Sequence[Tuple[Box, str]],
    formula_boxes: Sequence[Box],
    *,
    placeholder_prefix: str = 'FORMULA',
    line_thresh: float = 0.5,
) -> FormulaComposeResult:
    items: List[RegionItem] = []

    for box, text in text_segments:
        items.append(RegionItem(kind='text', box=box, payload={'text': text}))

    for box in formula_boxes:
        items.append(RegionItem(kind='formula', box=box, payload={}))

    ordered = sort_regions(items, line_thresh=line_thresh)
    pieces: List[str] = []
    formula_map: Dict[str, Box] = {}
    formula_idx = 1

    for item in ordered:
        if item.kind == 'text':
            text = str(item.payload.get('text', '')).strip()
            if text:
                pieces.append(text)
        elif item.kind == 'formula':
            token = f'[{placeholder_prefix}_{formula_idx}]'
            formula_map[token] = item.box
            pieces.append(token)
            formula_idx += 1

    merged_text = '\n'.join(pieces)
    return FormulaComposeResult(merged_text=merged_text, formula_map=formula_map)
