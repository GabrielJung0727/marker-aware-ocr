from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

Box = Tuple[float, float, float, float]


@dataclass
class PluginDetections:
    boxes: List[Box]
    errors: List[str]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _extract_polygons(obj: Any) -> List[List[Tuple[float, float]]]:
    polys: List[List[Tuple[float, float]]] = []

    def walk(node: Any) -> None:
        if isinstance(node, np.ndarray):
            node = node.tolist()

        if isinstance(node, (list, tuple)):
            # direct polygon candidate: [[x,y], [x,y], [x,y], [x,y]]
            if len(node) >= 4 and all(
                isinstance(p, (list, tuple)) and len(p) >= 2 and _is_number(p[0]) and _is_number(p[1])
                for p in node
            ):
                poly = [(float(p[0]), float(p[1])) for p in node]
                polys.append(poly)
                return
            for child in node:
                walk(child)

    walk(obj)
    return polys


def polygons_to_boxes(polygons: Iterable[List[Tuple[float, float]]], *, min_area: float = 20.0) -> List[Box]:
    boxes: List[Box] = []
    for poly in polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if (x2 - x1) * (y2 - y1) < float(min_area):
            continue
        boxes.append((x1, y1, x2, y2))
    return boxes


def clip_boxes(boxes: List[Box], width: int, height: int) -> List[Box]:
    clipped: List[Box] = []
    for x1, y1, x2, y2 in boxes:
        nx1 = max(0.0, min(float(width), x1))
        ny1 = max(0.0, min(float(height), y1))
        nx2 = max(0.0, min(float(width), x2))
        ny2 = max(0.0, min(float(height), y2))
        if nx2 > nx1 and ny2 > ny1:
            clipped.append((nx1, ny1, nx2, ny2))
    return clipped


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def nms_boxes(boxes: List[Box], iou_threshold: float = 0.5) -> List[Box]:
    if not boxes:
        return []
    # score by area descending
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]), reverse=True)
    kept: List[Box] = []
    for idx in order:
        candidate = boxes[idx]
        if any(iou(candidate, k) > float(iou_threshold) for k in kept):
            continue
        kept.append(candidate)
    return kept


def detect_with_paddle(image: Image.Image, cfg: dict) -> PluginDetections:
    errors: List[str] = []
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # noqa: BLE001
        return PluginDetections(boxes=[], errors=[f'paddle_import_error: {exc}'])

    try:
        import inspect

        lang = cfg.get('lang', 'korean')
        use_angle_cls = bool(cfg.get('use_angle_cls', True))
        init_params = set(inspect.signature(PaddleOCR.__init__).parameters.keys())

        init_kwargs: Dict[str, Any] = {
            'use_angle_cls': use_angle_cls,
            'lang': lang,
        }
        if 'show_log' in init_params:
            init_kwargs['show_log'] = bool(cfg.get('show_log', False))
        if 'use_gpu' in init_params:
            init_kwargs['use_gpu'] = bool(cfg.get('use_gpu', False))
        if 'device' in init_params and 'use_gpu' not in init_params:
            init_kwargs['device'] = 'gpu' if bool(cfg.get('use_gpu', False)) else 'cpu'

        detector = PaddleOCR(**init_kwargs)
        np_img = np.array(image.convert('RGB'))
        # API differs across PaddleOCR versions; retry with safest calls.
        try:
            result = detector.ocr(np_img, cls=use_angle_cls)
        except TypeError:
            result = detector.ocr(np_img)
        polygons = _extract_polygons(result)
        boxes = polygons_to_boxes(polygons, min_area=float(cfg.get('min_area', 20.0)))
        boxes = clip_boxes(boxes, image.width, image.height)
        boxes = nms_boxes(boxes, iou_threshold=float(cfg.get('iou_threshold', 0.4)))
        max_boxes = int(cfg.get('max_boxes', 400))
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        return PluginDetections(boxes=boxes, errors=errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f'paddle_detect_error: {exc}')
        return PluginDetections(boxes=[], errors=errors)


def detect_with_craft(image: Image.Image, cfg: dict) -> PluginDetections:
    errors: List[str] = []
    try:
        from craft_text_detector import Craft
    except Exception as exc:  # noqa: BLE001
        return PluginDetections(boxes=[], errors=[f'craft_import_error: {exc}'])

    craft = None
    try:
        craft = Craft(
            output_dir=None,
            crop_type='box',
            cuda=bool(cfg.get('cuda', False)),
        )
        np_img = np.array(image.convert('RGB'))
        pred = craft.detect_text(np_img)
        polygons = _extract_polygons(pred.get('boxes') if isinstance(pred, dict) else pred)
        boxes = polygons_to_boxes(polygons, min_area=float(cfg.get('min_area', 20.0)))
        boxes = clip_boxes(boxes, image.width, image.height)
        boxes = nms_boxes(boxes, iou_threshold=float(cfg.get('iou_threshold', 0.4)))
        max_boxes = int(cfg.get('max_boxes', 400))
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        return PluginDetections(boxes=boxes, errors=errors)
    except Exception as exc:  # noqa: BLE001
        errors.append(f'craft_detect_error: {exc}')
        return PluginDetections(boxes=[], errors=errors)
    finally:
        if craft is not None:
            try:
                craft.unload_craftnet_model()
                craft.unload_refinenet_model()
            except Exception:
                pass


def detect_text_boxes_with_plugins(image: Image.Image, cfg: dict) -> Dict[str, PluginDetections]:
    outputs: Dict[str, PluginDetections] = {}
    plugins = cfg.get('plugins', {})

    paddle_cfg = plugins.get('paddle', {})
    if bool(paddle_cfg.get('enabled', False)):
        outputs['paddle'] = detect_with_paddle(image, paddle_cfg)

    craft_cfg = plugins.get('craft', {})
    if bool(craft_cfg.get('enabled', False)):
        outputs['craft'] = detect_with_craft(image, craft_cfg)

    return outputs


def merge_boxes(groups: List[List[Box]], *, iou_threshold: float = 0.5, max_boxes: int = 400) -> List[Box]:
    merged: List[Box] = []
    for boxes in groups:
        merged.extend(boxes)
    merged = nms_boxes(merged, iou_threshold=iou_threshold)
    if len(merged) > int(max_boxes):
        merged = merged[: int(max_boxes)]
    return merged
