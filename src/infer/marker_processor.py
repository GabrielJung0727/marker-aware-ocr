from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

Box = Tuple[float, float, float, float]


@dataclass
class RedInkSeparationResult:
    text_image: Image.Image
    marker_mask_image: Image.Image
    marker_layer_image: Image.Image


def box_intersects(a: Box, b: Box) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def markers_in_text_box(text_box: Box, marker_boxes: Iterable[Box]) -> List[Box]:
    return [m for m in marker_boxes if box_intersects(text_box, m)]


def mask_or_inpaint_crop(
    crop_image: Image.Image,
    text_box: Box,
    marker_boxes: Iterable[Box],
    *,
    use_inpaint: bool = True,
    inpaint_radius: int = 3,
) -> Image.Image:
    """Apply marker masking to a text crop and optionally inpaint masked pixels."""
    markers = markers_in_text_box(text_box, marker_boxes)
    if not markers:
        return crop_image

    crop_rgb = np.array(crop_image)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(crop_bgr.shape[:2], dtype=np.uint8)

    tx1, ty1, tx2, ty2 = text_box
    crop_w = max(1, int(tx2 - tx1))
    crop_h = max(1, int(ty2 - ty1))

    for mx1, my1, mx2, my2 in markers:
        rx1 = max(0, int(mx1 - tx1))
        ry1 = max(0, int(my1 - ty1))
        rx2 = min(crop_w, int(mx2 - tx1))
        ry2 = min(crop_h, int(my2 - ty1))
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, thickness=-1)

    if use_inpaint:
        processed = cv2.inpaint(crop_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        processed = crop_bgr.copy()
        processed[mask > 0] = (255, 255, 255)

    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(processed_rgb)


def build_red_ink_mask(
    image_rgb: np.ndarray,
    *,
    saturation_min: int = 70,
    value_min: int = 35,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Red hue wraps around 0/180 in HSV.
    s_min = max(0, int(saturation_min))
    v_min = max(0, int(value_min))
    lower1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, s_min, v_min], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def remove_red_ink(
    image: Image.Image,
    *,
    use_inpaint: bool = True,
    inpaint_radius: int = 3,
    morph_kernel: int = 3,
    strategy: str = 'preserve_text',
    keep_dark_threshold: int = 150,
    saturation_min: int = 70,
    value_min: int = 35,
) -> RedInkSeparationResult:
    """Separate marker layer from text layer and return images for OCR/visualization."""
    rgb = np.array(image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = build_red_ink_mask(rgb, saturation_min=saturation_min, value_min=value_min)

    k = max(1, int(morph_kernel))
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    strategy = str(strategy).lower()
    if strategy == 'inpaint':
        if use_inpaint:
            processed_bgr = cv2.inpaint(bgr, mask, max(1, int(inpaint_radius)), cv2.INPAINT_TELEA)
        else:
            processed_bgr = bgr.copy()
            processed_bgr[mask > 0] = (255, 255, 255)
    elif strategy == 'preserve_text':
        # Keep text-like dark pixels even inside red region, remove only marker color cast.
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]
        gb_min = np.minimum(g, b)
        text_like = gb_min < int(keep_dark_threshold)

        processed_rgb = rgb.copy()
        processed_rgb[mask > 0] = (255, 255, 255)

        # For dark red-over-text pixels, recover as neutral gray from GB channels.
        replacement = np.stack([gb_min, gb_min, gb_min], axis=-1).astype(np.uint8)
        keep_idx = (mask > 0) & text_like
        processed_rgb[keep_idx] = replacement[keep_idx]
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
    else:
        # hard white-mask fallback
        processed_bgr = bgr.copy()
        processed_bgr[mask > 0] = (255, 255, 255)

    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    # visualize mask in red overlay style for trace viewer
    mask_vis = np.zeros_like(processed_rgb)
    mask_vis[..., 0] = mask
    marker_layer = np.full_like(processed_rgb, 255)
    marker_layer[mask > 0] = rgb[mask > 0]

    return RedInkSeparationResult(
        text_image=Image.fromarray(processed_rgb),
        marker_mask_image=Image.fromarray(mask_vis),
        marker_layer_image=Image.fromarray(marker_layer),
    )
