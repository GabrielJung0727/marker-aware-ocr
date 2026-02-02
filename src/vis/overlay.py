from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from PIL import Image, ImageDraw

Box = Tuple[float, float, float, float]


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[Box],
    labels: Iterable[str] | None = None,
    color: str = 'red',
    width: int = 2,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    label_list = list(labels) if labels is not None else [''] * len(boxes)

    for box, label in zip(boxes, label_list):
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        if label:
            draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    return canvas
