from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import TesseractError, TesseractNotFoundError

Box = Tuple[float, float, float, float]


@dataclass
class CharDebugResult:
    gray_image: Image.Image
    binary_image: Image.Image
    contour_overlay: Image.Image
    contour_binary_overlay: Image.Image
    aligned_overlay: Image.Image
    char_gallery_image: Image.Image | None
    contour_boxes: List[Box]
    aligned_boxes: List[Box]
    tesseract_overlay: Image.Image | None
    tesseract_chars: List[dict]


def _ensure_odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def preprocess_for_contours(
    image: Image.Image,
    *,
    blur_kernel: int = 3,
    threshold_mode: str = 'adaptive',
    adaptive_block_size: int = 31,
    adaptive_c: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    k = _ensure_odd(blur_kernel)
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    mode = str(threshold_mode).lower()
    if mode == 'otsu':
        _thr, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        b = _ensure_odd(adaptive_block_size)
        binary_inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            b,
            int(adaptive_c),
        )

    return gray, binary_inv


def extract_contour_boxes(
    binary_inv: np.ndarray,
    *,
    min_area: int = 16,
    max_area_ratio: float = 0.03,
    min_w: int = 2,
    min_h: int = 6,
    max_boxes: int = 5000,
) -> List[Box]:
    contours, _hier = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary_inv.shape[:2]
    max_area = max(1.0, float(max_area_ratio) * float(w * h))

    boxes: List[Box] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area = float(cw * ch)
        if area < float(min_area) or area > max_area:
            continue
        if cw < int(min_w) or ch < int(min_h):
            continue
        boxes.append((float(x), float(y), float(x + cw), float(y + ch)))

    boxes.sort(key=lambda b: ((b[1] + b[3]) * 0.5, b[0]))
    if len(boxes) > int(max_boxes):
        boxes = boxes[: int(max_boxes)]
    return boxes


def align_boxes_reading_order(boxes: List[Box], line_thresh: float = 0.6) -> List[Box]:
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: (((b[1] + b[3]) * 0.5), b[0]))
    lines: List[List[Box]] = []
    current: List[Box] = []
    current_y = None
    current_h = None

    for box in sorted_boxes:
        y_center = (box[1] + box[3]) * 0.5
        height = max(1.0, box[3] - box[1])
        if current_y is None:
            current = [box]
            current_y = y_center
            current_h = height
            continue

        threshold = (current_h or height) * float(line_thresh)
        if abs(y_center - current_y) <= threshold:
            current.append(box)
            current_y = (current_y + y_center) * 0.5
            current_h = (current_h + height) * 0.5
        else:
            lines.append(sorted(current, key=lambda x: x[0]))
            current = [box]
            current_y = y_center
            current_h = height

    if current:
        lines.append(sorted(current, key=lambda x: x[0]))

    ordered: List[Box] = []
    for line in lines:
        ordered.extend(line)
    return ordered


def draw_boxes_overlay(
    image: Image.Image,
    boxes: List[Box],
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    labels: List[str] | None = None,
) -> Image.Image:
    canvas = np.array(image.convert('RGB')).copy()
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    if labels is None:
        labels = [''] * len(boxes)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (int(color[2]), int(color[1]), int(color[0])), 1)
        label = labels[idx] if idx < len(labels) else ''
        if label:
            cv2.putText(
                bgr,
                label,
                (x1, max(0, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (int(color[2]), int(color[1]), int(color[0])),
                1,
            )
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def build_char_gallery(
    image: Image.Image,
    boxes: List[Box],
    *,
    limit: int = 120,
    columns: int = 12,
    tile_width: int = 64,
    tile_height: int = 48,
) -> Image.Image | None:
    if not boxes:
        return None
    selected = boxes[: max(1, int(limit))]
    cols = max(1, int(columns))
    rows = int(np.ceil(len(selected) / cols))
    pad = 8
    canvas_w = cols * (tile_width + pad) + pad
    canvas_h = rows * (tile_height + pad + 16) + pad
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    src = np.array(image.convert('L'))

    for idx, box in enumerate(selected):
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = src[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue
        resized = cv2.resize(crop, (tile_width, tile_height), interpolation=cv2.INTER_LINEAR)
        r = idx // cols
        c = idx % cols
        ox = pad + c * (tile_width + pad)
        oy = pad + r * (tile_height + pad + 16)
        canvas[oy:oy + tile_height, ox:ox + tile_width] = resized
        cv2.putText(canvas, str(idx + 1), (ox, oy + tile_height + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 0, 1)

    return Image.fromarray(canvas)


def extract_tesseract_char_boxes(
    image: Image.Image,
    *,
    lang: str = 'eng',
    psm: int = 6,
    oem: int = 1,
) -> List[dict]:
    config = f'--psm {int(psm)} --oem {int(oem)}'
    try:
        raw = pytesseract.image_to_boxes(image, lang=lang, config=config)
    except (TesseractNotFoundError, TesseractError):
        return []

    width, height = image.size
    chars: List[dict] = []
    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        ch = parts[0]
        try:
            x1 = int(parts[1])
            y1 = int(parts[2])
            x2 = int(parts[3])
            y2 = int(parts[4])
        except ValueError:
            continue

        top = float(height - y2)
        bottom = float(height - y1)
        chars.append(
            {
                'char': ch,
                'box': [float(x1), top, float(x2), bottom],
            }
        )

    return chars


def draw_tesseract_char_overlay(image: Image.Image, chars: List[dict]) -> Image.Image:
    if not chars:
        return image
    canvas = np.array(image.convert('RGB')).copy()
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for item in chars:
        box = item.get('box', [])
        ch = str(item.get('char', ''))
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 255), 1)
        if ch:
            cv2.putText(bgr, ch, (x1, max(0, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def build_char_debug_result(image: Image.Image, cfg: dict) -> CharDebugResult:
    gray, binary_inv = preprocess_for_contours(
        image,
        blur_kernel=int(cfg.get('blur_kernel', 3)),
        threshold_mode=cfg.get('threshold_mode', 'adaptive'),
        adaptive_block_size=int(cfg.get('adaptive_block_size', 31)),
        adaptive_c=int(cfg.get('adaptive_c', 11)),
    )

    boxes = extract_contour_boxes(
        binary_inv,
        min_area=int(cfg.get('min_area', 16)),
        max_area_ratio=float(cfg.get('max_area_ratio', 0.03)),
        min_w=int(cfg.get('min_w', 2)),
        min_h=int(cfg.get('min_h', 6)),
        max_boxes=int(cfg.get('max_boxes', 5000)),
    )
    aligned = align_boxes_reading_order(boxes, line_thresh=float(cfg.get('line_thresh', 0.6)))

    label_limit = int(cfg.get('label_limit', 120))
    contour_labels = [str(i + 1) if i < label_limit else '' for i in range(len(boxes))]
    aligned_labels = [str(i + 1) if i < label_limit else '' for i in range(len(aligned))]
    contour_overlay = draw_boxes_overlay(image, boxes, color=(255, 0, 0), labels=contour_labels)
    aligned_overlay = draw_boxes_overlay(image, aligned, color=(0, 128, 255), labels=aligned_labels)
    gray_img = Image.fromarray(gray)
    binary_img = Image.fromarray(255 - binary_inv)
    binary_rgb = cv2.cvtColor(255 - binary_inv, cv2.COLOR_GRAY2RGB)
    contour_binary_overlay = draw_boxes_overlay(Image.fromarray(binary_rgb), boxes, color=(255, 0, 0), labels=contour_labels)
    gallery_image = build_char_gallery(
        image,
        aligned,
        limit=int(cfg.get('gallery_limit', 120)),
        columns=int(cfg.get('gallery_columns', 12)),
        tile_width=int(cfg.get('gallery_tile_width', 64)),
        tile_height=int(cfg.get('gallery_tile_height', 48)),
    )

    tesseract_enabled = bool(cfg.get('tesseract_char_boxes', False))
    if tesseract_enabled:
        chars = extract_tesseract_char_boxes(
            image,
            lang=cfg.get('tesseract_lang', 'eng'),
            psm=int(cfg.get('tesseract_psm', 6)),
            oem=int(cfg.get('tesseract_oem', 1)),
        )
        tess_overlay = draw_tesseract_char_overlay(image, chars)
    else:
        chars = []
        tess_overlay = None

    return CharDebugResult(
        gray_image=gray_img,
        binary_image=binary_img,
        contour_overlay=contour_overlay,
        contour_binary_overlay=contour_binary_overlay,
        aligned_overlay=aligned_overlay,
        char_gallery_image=gallery_image,
        contour_boxes=boxes,
        aligned_boxes=aligned,
        tesseract_overlay=tess_overlay,
        tesseract_chars=chars,
    )
