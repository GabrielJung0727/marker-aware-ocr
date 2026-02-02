from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PIL import Image
from ultralytics import YOLO
import yaml
import cv2
import numpy as np

from .char_analysis import build_char_debug_result
from .formula_processor import compose_with_formula_placeholders
from .marker_processor import mask_or_inpaint_crop, remove_red_ink
from .ocr_engine import OcrOptions, OcrResult, pick_best_result
from .ocr_llama import run_postprocess
from .ocr_router import OcrRoutePolicy, OcrRouter
from .ollama_client import OllamaClient, get_default_host
from .option_parser import parse_option_blocks
from .reading_order import sort_boxes
from .text_detectors import detect_text_boxes_with_plugins, merge_boxes
from src.vis.overlay import draw_boxes
from src.vis.tracer import TraceLogger

Box = Tuple[float, float, float, float]


@dataclass
class YoloPredictOptions:
    model: str
    conf: float = 0.25
    imgsz: int = 1024
    device: str | int | None = None
    classes: List[int] | None = None
    max_det: int = 300
    crop_padding: int = 6
    line_thresh: float = 0.5


@dataclass
class Detection:
    label: str
    box: Box
    conf: float
    source: str


@dataclass
class QuestionChunk:
    question_no: int
    text: str
    block_ids: List[str]
    confidence_min: float | None


@dataclass
class RawTextComposeResult:
    selected_text: str
    mode: str
    page_dump_text: str
    question_chunk_text: str
    uncaptured_block_ids: List[str]
    coverage_ratio: float


def normalize_device(device: str | int | None) -> str | int | None:
    if device is None:
        return None
    device_text = str(device).strip().lower()
    if device_text in {'cpu', 'mps', ''}:
        return device
    try:
        import torch
    except Exception:
        print('[WARN] torch를 불러오지 못해 device를 cpu로 강제합니다.')
        return 'cpu'
    if not torch.cuda.is_available():
        print('[WARN] CUDA를 사용할 수 없어 device를 cpu로 변경합니다.')
        return 'cpu'
    return device


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def expand_box(box: Box, pad: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(width, int(x2) + pad)
    y2 = min(height, int(y2) + pad)
    return x1, y1, x2, y2


def build_crops(image: Image.Image, boxes: Iterable[Box], pad: int) -> List[Image.Image]:
    width, height = image.size
    crops: List[Image.Image] = []
    for box in boxes:
        x1, y1, x2, y2 = expand_box(box, pad, width, height)
        crops.append(image.crop((x1, y1, x2, y2)))
    return crops


def preprocess_crop_for_ocr(crop: Image.Image, cfg: dict) -> Image.Image:
    mode = str(cfg.get('mode', 'none')).lower()
    scale = float(cfg.get('scale', 1.0))
    if mode == 'none' and abs(scale - 1.0) < 1e-6:
        return crop

    arr = np.array(crop.convert('RGB'))

    if mode == 'gray':
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif mode == 'adaptive':
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        arr = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

    out = Image.fromarray(arr)
    if scale > 1.0:
        w, h = out.size
        out = out.resize((int(w * scale), int(h * scale)))
    return out


def build_predict_options(cfg: dict) -> YoloPredictOptions:
    return YoloPredictOptions(
        model=cfg.get('model', 'yolov8n.pt'),
        conf=float(cfg.get('conf', 0.25)),
        imgsz=int(cfg.get('imgsz', 1024)),
        device=normalize_device(cfg.get('device')),
        classes=cfg.get('classes'),
        max_det=int(cfg.get('max_det', 300)),
        crop_padding=int(cfg.get('crop_padding', 6)),
        line_thresh=float(cfg.get('line_thresh', 0.5)),
    )


def detect_regions(image_path: str, opts: YoloPredictOptions, source: str = 'single') -> List[Detection]:
    model = YOLO(opts.model)
    results = model.predict(
        source=image_path,
        conf=opts.conf,
        imgsz=opts.imgsz,
        device=opts.device,
        classes=opts.classes,
        max_det=opts.max_det,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None or boxes.cls is None:
        return []

    names = result.names or {}
    detections: List[Detection] = []
    for box, cls_idx, conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
        label = names.get(int(cls_idx), f'class_{int(cls_idx)}')
        detections.append(
            Detection(
                label=label,
                box=tuple(map(float, box)),
                conf=float(conf),
                source=source,
            )
        )
    return detections


def group_boxes_by_label(detections: Iterable[Detection]) -> Dict[str, List[Box]]:
    grouped: Dict[str, List[Box]] = {}
    for det in detections:
        grouped.setdefault(det.label, []).append(det.box)
    return grouped


def extract_marker_boxes(grouped: Dict[str, List[Box]]) -> Tuple[List[Box], List[str]]:
    marker_boxes: List[Box] = []
    marker_labels: List[str] = []
    for label, boxes in grouped.items():
        if label == 'marker' or label.startswith('marker'):
            marker_boxes.extend(boxes)
            marker_labels.extend([label] * len(boxes))
    return marker_boxes, marker_labels


def select_text_boxes(grouped: Dict[str, List[Box]], detections: List[Detection], line_thresh: float) -> List[Box]:
    text_boxes = grouped.get('text_region', [])
    if not text_boxes:
        text_boxes = grouped.get('text', [])
    use_any_detection = any(
        d.label in {'text_region', 'text', 'formula_region', 'option_block', 'marker'}
        for d in detections
    )
    if not text_boxes and use_any_detection:
        text_boxes = [d.box for d in detections]
    return sort_boxes(text_boxes, line_thresh=line_thresh)


def choose_text_boxes_with_plugin_strategy(
    yolo_boxes: List[Box],
    plugin_boxes: List[Box],
    cfg: dict,
    *,
    line_thresh: float,
) -> Tuple[List[Box], str]:
    strategy = str(cfg.get('strategy', 'yolo_then_plugins')).lower()
    iou_threshold = float(cfg.get('merge_iou_threshold', 0.5))
    max_boxes = int(cfg.get('max_boxes', 400))

    if strategy == 'plugins_only':
        if plugin_boxes:
            return sort_boxes(plugin_boxes, line_thresh=line_thresh), 'plugins_only'
        return sort_boxes(yolo_boxes, line_thresh=line_thresh), 'plugins_only_fallback_yolo'

    if strategy == 'merge':
        merged = merge_boxes([yolo_boxes, plugin_boxes], iou_threshold=iou_threshold, max_boxes=max_boxes)
        return sort_boxes(merged, line_thresh=line_thresh), 'merge'

    if strategy == 'yolo_only':
        return sort_boxes(yolo_boxes, line_thresh=line_thresh), 'yolo_only'

    # default: yolo_then_plugins
    if yolo_boxes:
        return sort_boxes(yolo_boxes, line_thresh=line_thresh), 'yolo_then_plugins_yolo'
    return sort_boxes(plugin_boxes, line_thresh=line_thresh), 'yolo_then_plugins_plugin'


def build_fallback_text_boxes(image: Image.Image, cfg: dict) -> List[Box]:
    if not bool(cfg.get('enabled', True)):
        return []
    width, height = image.size
    mode = str(cfg.get('mode', 'columns')).lower()
    margin = int(cfg.get('margin', 8))

    if mode == 'full':
        return [(float(margin), float(margin), float(width - margin), float(height - margin))]

    columns = max(1, int(cfg.get('columns', 2)))
    rows = max(1, int(cfg.get('rows', 4)))
    boxes: List[Box] = []

    col_w = width / columns
    row_h = height / rows
    for r in range(rows):
        for c in range(columns):
            x1 = int(c * col_w) + margin
            y1 = int(r * row_h) + margin
            x2 = int((c + 1) * col_w) - margin
            y2 = int((r + 1) * row_h) - margin
            if x2 > x1 and y2 > y1:
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def build_llm_input_payload(
    image_path: str,
    text_segments: List[Tuple[Box, str]],
    ocr_records: List[dict],
    marker_boxes: List[Box],
    question_chunks: List[QuestionChunk],
) -> dict:
    def infer_role(text: str) -> str:
        t = text.strip().lower()
        if not t:
            return 'misc'
        if '보기 안에서' in text or '빈칸' in text or '바르게 배열' in text:
            return 'instruction'
        if '[' in text and ']' in text:
            return 'options'
        if '(' in text and ')' in text and t.count('(') >= 2:
            return 'options'
        if '?' in text or text.endswith('.'):
            return 'prompt'
        return 'misc'

    def infer_kind(rec: dict) -> str:
        engine = str(rec.get('engine', '')).lower()
        if 'trocr' in engine:
            return 'handwriting'
        return 'printed'

    blocks: List[dict] = []
    for idx, ((box, text), rec) in enumerate(zip(text_segments, ocr_records), start=1):
        if not str(text).strip():
            continue
        question_nos = extract_question_numbers(str(text))
        question_no = question_nos[0] if question_nos else None
        blocks.append(
            {
                'id': f'b{idx:03d}',
                'kind': infer_kind(rec),
                'role': infer_role(str(text)),
                'text': str(text).strip(),
                'bbox': [float(v) for v in box],
                'conf': float(rec.get('conf', 0.0)),
                'question_no': question_no,
                'question_nos': question_nos,
            }
        )

    for idx, box in enumerate(marker_boxes, start=1):
        blocks.append(
            {
                'id': f'm{idx:03d}',
                'kind': 'mark',
                'role': 'misc',
                'text': '',
                'bbox': [float(v) for v in box],
                'conf': None,
            }
        )

    expected_nos = sorted(
        {q.question_no for q in question_chunks}
        | {int(b['question_no']) for b in blocks if isinstance(b.get('question_no'), int)}
    )
    chunk_records = [
        {
            'question_no': q.question_no,
            'text': q.text,
            'block_ids': q.block_ids,
            'confidence_min': q.confidence_min,
        }
        for q in question_chunks
    ]

    return {
        'page_no': None,
        'source': Path(image_path).name,
        'blocks': blocks,
        'question_chunks': chunk_records,
        'expected_question_nos': expected_nos,
        'detected_question_nos_from_blocks': sorted(
            {int(no) for b in blocks for no in b.get('question_nos', []) if isinstance(no, int)}
        ),
    }


def extract_question_numbers(text: str) -> List[int]:
    numbers: List[int] = []
    seen: set[int] = set()
    for m in re.finditer(r'(?<!\d)(\d{1,3})[\.,](?=\s|$|[A-Za-z가-힣\(\[])', text):
        no = int(m.group(1))
        if no in seen:
            continue
        seen.add(no)
        numbers.append(no)
    return numbers


def split_text_by_question_numbers(text: str) -> List[Tuple[int, str]]:
    matches = list(re.finditer(r'(?<!\d)(\d{1,3})[\.,](?=\s|$|[A-Za-z가-힣\(\[])', text))
    if not matches:
        return []
    chunks: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        qno = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((qno, chunk_text))
    return chunks


def build_question_chunks(
    text_segments: List[Tuple[Box, str]],
    ocr_records: List[dict],
) -> List[QuestionChunk]:
    chunks: List[QuestionChunk] = []
    for idx, ((_box, text), rec) in enumerate(zip(text_segments, ocr_records), start=1):
        t = str(text).strip()
        if not t:
            continue
        block_id = f'b{idx:03d}'
        conf = float(rec.get('conf', 0.0))
        split_chunks = split_text_by_question_numbers(t)
        if split_chunks:
            for qno, chunk_text in split_chunks:
                chunks.append(
                    QuestionChunk(
                        question_no=qno,
                        text=chunk_text,
                        block_ids=[block_id],
                        confidence_min=conf,
                    )
                )
        else:
            # keep non-numbered segments out of question chunks
            continue
    # deduplicate/merge by question number while preserving order
    merged: Dict[int, QuestionChunk] = {}
    order: List[int] = []
    for c in chunks:
        if c.question_no not in merged:
            merged[c.question_no] = QuestionChunk(
                question_no=c.question_no,
                text=c.text,
                block_ids=list(c.block_ids),
                confidence_min=c.confidence_min,
            )
            order.append(c.question_no)
            continue
        m = merged[c.question_no]
        m.text = (m.text + '\n' + c.text).strip()
        for bid in c.block_ids:
            if bid not in m.block_ids:
                m.block_ids.append(bid)
        if m.confidence_min is None:
            m.confidence_min = c.confidence_min
        elif c.confidence_min is not None:
            m.confidence_min = min(m.confidence_min, c.confidence_min)
    return [merged[q] for q in sorted(order)]


def compose_raw_text(question_chunks: List[QuestionChunk], text_segments: List[Tuple[Box, str]]) -> RawTextComposeResult:
    page_dump_text = '\n'.join(str(text).strip() for _, text in text_segments if str(text).strip())
    question_chunk_text = '\n\n'.join(q.text for q in question_chunks if q.text.strip())

    captured_block_ids = {bid for q in question_chunks for bid in q.block_ids}
    uncaptured_block_ids: List[str] = []
    uncaptured_texts: List[str] = []
    for idx, (_box, text) in enumerate(text_segments, start=1):
        t = str(text).strip()
        if not t:
            continue
        block_id = f'b{idx:03d}'
        if block_id in captured_block_ids:
            continue
        uncaptured_block_ids.append(block_id)
        uncaptured_texts.append(t)

    page_chars = len(re.sub(r'\s+', '', page_dump_text))
    chunk_chars = len(re.sub(r'\s+', '', question_chunk_text))
    coverage_ratio = float(chunk_chars) / float(max(1, page_chars))

    if question_chunk_text and uncaptured_texts:
        selected_text = question_chunk_text + '\n\n' + '\n'.join(uncaptured_texts)
        mode = 'hybrid'
    elif question_chunk_text:
        selected_text = question_chunk_text
        mode = 'question_chunks'
    else:
        selected_text = page_dump_text
        mode = 'page_dump'

    return RawTextComposeResult(
        selected_text=selected_text,
        mode=mode,
        page_dump_text=page_dump_text,
        question_chunk_text=question_chunk_text,
        uncaptured_block_ids=uncaptured_block_ids,
        coverage_ratio=coverage_ratio,
    )


def analyze_ocr_text_quality(text: str, conf: float | None) -> dict:
    stripped = str(text or '').strip()
    tokens = re.findall(r'\S+', stripped)
    non_space = [ch for ch in stripped if not ch.isspace()]
    char_count = len(non_space)
    alnum_count = sum(ch.isalnum() for ch in non_space)
    hangul_count = len(re.findall(r'[가-힣]', stripped))
    latin_count = len(re.findall(r'[A-Za-z]', stripped))
    digit_count = len(re.findall(r'\d', stripped))
    word_like_count = len(re.findall(r'[A-Za-z]{2,}|[가-힣]{2,}', stripped))
    suspicious_symbols = sum(ch in '|[]{}<>@#$%^&*=+~`' for ch in non_space)

    alnum_ratio = float(alnum_count) / float(max(1, char_count))
    symbol_ratio = float(suspicious_symbols) / float(max(1, char_count))
    word_ratio = float(word_like_count) / float(max(1, len(tokens)))
    quality_score = max(0.0, min(1.0, (0.58 * alnum_ratio) + (0.22 * word_ratio) + (0.20 * (1.0 - symbol_ratio))))
    conf_penalty = 0.25 if (conf is not None and conf < 0.5) else 0.0
    noise_score = max(0.0, min(2.0, (1.0 - quality_score) + (0.60 * symbol_ratio) + conf_penalty))

    reasons: List[str] = []
    if quality_score < 0.45 and char_count >= 12:
        reasons.append('low_quality_score')
    if symbol_ratio > 0.18 and char_count >= 12:
        reasons.append('high_symbol_ratio')
    if word_ratio < 0.2 and len(tokens) >= 6:
        reasons.append('few_word_tokens')
    if conf is not None and conf < 0.5:
        reasons.append('low_confidence')

    return {
        'char_count': char_count,
        'token_count': len(tokens),
        'alnum_ratio': round(alnum_ratio, 4),
        'symbol_ratio': round(symbol_ratio, 4),
        'word_ratio': round(word_ratio, 4),
        'hangul_count': hangul_count,
        'latin_count': latin_count,
        'digit_count': digit_count,
        'quality_score': round(quality_score, 4),
        'noise_score': round(noise_score, 4),
        'suspect': bool(reasons),
        'suspect_reasons': reasons,
    }


def build_raw_text_debug(
    raw_text_result: RawTextComposeResult,
    text_segments: List[Tuple[Box, str]],
    ocr_records: List[dict],
    selected_text: str,
    *,
    selected_mode: str | None = None,
) -> dict:
    by_box: List[dict] = []
    suspect_boxes: List[dict] = []

    for idx, ((box, text), rec) in enumerate(zip(text_segments, ocr_records), start=1):
        conf = rec.get('conf')
        conf_value = float(conf) if isinstance(conf, (int, float)) else None
        quality = analyze_ocr_text_quality(text, conf_value)
        qnos = extract_question_numbers(str(text))

        x1, y1, x2, y2 = box
        box_w = max(0.0, float(x2) - float(x1))
        box_h = max(0.0, float(y2) - float(y1))
        area = box_w * box_h

        extra_reasons = list(quality.get('suspect_reasons', []))
        if len(qnos) > 1:
            extra_reasons.append('multiple_question_numbers_in_one_box')
        if area > 250000 and quality.get('char_count', 0) > 80 and not qnos:
            extra_reasons.append('large_box_without_question_no')
        if quality.get('char_count', 0) > 180:
            extra_reasons.append('oversized_ocr_block')

        noise_score = float(quality.get('noise_score') or 0.0)
        if 'multiple_question_numbers_in_one_box' in extra_reasons:
            noise_score += 0.2
        if 'large_box_without_question_no' in extra_reasons:
            noise_score += 0.2
        if 'oversized_ocr_block' in extra_reasons:
            noise_score += 0.15
        noise_score = round(min(2.0, noise_score), 4)
        quality['noise_score'] = noise_score
        quality['suspect_reasons'] = extra_reasons
        quality['suspect'] = bool(extra_reasons)

        entry = {
            'box_id': idx,
            'block_id': f'b{idx:03d}',
            'bbox': [float(v) for v in box],
            'box_area': round(area, 2),
            'engine': rec.get('engine'),
            'conf': conf_value,
            'question_numbers': qnos,
            'text': str(text),
            'text_preview': str(text).replace('\n', ' ')[:180],
            'quality': quality,
        }
        by_box.append(entry)
        if quality['suspect']:
            suspect_boxes.append(
                {
                    'box_id': idx,
                    'block_id': f'b{idx:03d}',
                    'noise_score': noise_score,
                    'quality_score': quality.get('quality_score'),
                    'suspect_reasons': extra_reasons,
                    'question_numbers': qnos,
                    'text_preview': entry['text_preview'],
                }
            )

    suspect_boxes.sort(key=lambda x: float(x.get('noise_score') or 0.0), reverse=True)

    return {
        'selected_mode': selected_mode or raw_text_result.mode,
        'coverage_ratio': round(raw_text_result.coverage_ratio, 4),
        'question_chunk_count': len([q for q in raw_text_result.question_chunk_text.split('\n\n') if q.strip()]),
        'box_count': len(by_box),
        'uncaptured_block_ids': raw_text_result.uncaptured_block_ids,
        'detected_question_numbers': sorted({int(n) for entry in by_box for n in entry.get('question_numbers', [])}),
        'selected_text': selected_text,
        'question_chunk_text': raw_text_result.question_chunk_text,
        'page_dump_text': raw_text_result.page_dump_text,
        'box_diagnostics': by_box,
        'suspect_boxes': suspect_boxes[:12],
    }


def inject_evidence_and_fill_missing_questions(corrected: str, question_chunks: List[QuestionChunk]) -> str:
    if not corrected or not question_chunks:
        return corrected
    try:
        data = json.loads(corrected)
    except json.JSONDecodeError:
        return corrected
    if not isinstance(data, dict):
        return corrected

    items = data.get('items')
    if not isinstance(items, list):
        return corrected

    qmap = {q.question_no: q for q in question_chunks}
    existing: Dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        no = item.get('no')
        if isinstance(no, int):
            existing[no] = item

    # Inject deterministic evidence to avoid LLM hallucinated ids/conf values.
    for no, item in existing.items():
        q = qmap.get(no)
        if not q:
            continue
        ev = item.get('evidence')
        if not isinstance(ev, dict):
            ev = {}
            item['evidence'] = ev
        ev['block_ids'] = q.block_ids
        ev['confidence_min'] = q.confidence_min

    # Ensure all detected question numbers are present in output.
    for no, q in qmap.items():
        if no in existing:
            continue
        items.append(
            {
                'no': no,
                'type': 'other',
                'instruction_ko': None,
                'prompt_en': q.text,
                'options': [],
                'tokens': [],
                'student_answer': None,
                'teacher_marks': {
                    'symbols': [],
                    'selected_option': None,
                    'is_correct': None,
                },
                'evidence': {
                    'block_ids': q.block_ids,
                    'confidence_min': q.confidence_min,
                },
                'notes': 'auto_added_missing_question_from_pipeline',
            }
        )

    def sort_key(item: dict) -> int:
        no = item.get('no')
        if isinstance(no, int):
            return no
        try:
            return int(no)
        except Exception:
            return 10**9

    items.sort(key=sort_key)
    data['items'] = items
    return json.dumps(data, ensure_ascii=False, indent=2)


def run_ocr_for_box(
    crop: Image.Image,
    region_label: str,
    router: OcrRouter,
    *,
    masked_crop: Image.Image | None = None,
    red_clean_crop: Image.Image | None = None,
    dual_ocr: bool = False,
) -> OcrResult:
    def infer_language_hint(text: str) -> str:
        if not text:
            return 'mixed'
        hangul = len(re.findall(r'[가-힣]', text))
        latin = len(re.findall(r'[A-Za-z]', text))
        if latin > max(3, hangul * 2):
            return 'eng'
        if hangul > max(3, latin * 2):
            return 'kor'
        return 'mixed'

    def text_weight(text: str) -> int:
        latin = len(re.findall(r'[A-Za-z]', text))
        hangul = len(re.findall(r'[가-힣]', text))
        digits = len(re.findall(r'[0-9]', text))
        return latin + hangul + digits

    engine_name = router.choose_engine_name(region_label)
    engine = router.get_engine(engine_name)

    candidate_pairs: List[Tuple[str, OcrResult]] = []

    try:
        primary = engine.read_image_result(crop)
        candidate_pairs.append(('original', primary))
    except Exception as exc:  # noqa: BLE001
        fallback_engine = router.get_engine(router.policy.low_conf_engine)
        fallback_result = fallback_engine.read_image_result(crop)
        fallback_result.evidence['router_error'] = str(exc)
        candidate_pairs.append(('fallback', fallback_result))

    if dual_ocr and masked_crop is not None:
        try:
            candidate_pairs.append(('marker_masked', engine.read_image_result(masked_crop)))
        except Exception:
            pass
    if dual_ocr and red_clean_crop is not None:
        try:
            candidate_pairs.append(('red_clean', engine.read_image_result(red_clean_crop)))
        except Exception:
            pass

    candidates = [res for _, res in candidate_pairs]
    chosen = pick_best_result(candidates)
    chosen_source = next((src for src, res in candidate_pairs if res is chosen), 'unknown')

    lang_hint = infer_language_hint(chosen.text)
    lang_engine_name = router.language_retry_engine(lang_hint, chosen.engine)
    if lang_engine_name:
        try:
            lang_engine = router.get_engine(lang_engine_name)
            lang_result = lang_engine.read_image_result(masked_crop or red_clean_crop or crop)
            lang_result.evidence['language_hint'] = lang_hint
            candidate_pairs.append((f'lang_retry:{lang_engine_name}', lang_result))
            candidates.append(lang_result)
            chosen = pick_best_result(candidates)
            chosen_source = next((src for src, res in candidate_pairs if res is chosen), chosen_source)
        except Exception:
            pass

    retry_engine_name = router.maybe_retry_low_conf(chosen.conf, chosen.engine)
    if retry_engine_name:
        retry_engine = router.get_engine(retry_engine_name)
        retry_target = masked_crop or red_clean_crop or crop
        retry_result = retry_engine.read_image_result(retry_target)
        candidate_pairs.append((f'low_conf_retry:{retry_engine_name}', retry_result))
        chosen = pick_best_result([chosen, retry_result])
        chosen_source = next((src for src, res in candidate_pairs if res is chosen), chosen_source)

    # Keep text/marker separation stable: if red-clean candidate drops too many readable chars,
    # prefer original OCR output to avoid erasing text under marker.
    original_result = next((res for src, res in candidate_pairs if src == 'original'), None)
    if (
        original_result is not None
        and chosen_source in {'red_clean', 'marker_masked'}
        and text_weight(chosen.text) < int(text_weight(original_result.text) * 0.75)
    ):
        chosen = original_result
        chosen_source = 'original_forced_by_text_preservation'

    chosen.evidence['source_variant'] = chosen_source

    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    image_path = cfg.get('image')
    if not image_path:
        raise ValueError('config.image is required')

    vis_cfg = cfg.get('trace', {})
    vis_cfg_path = cfg.get('vis_config')
    if vis_cfg_path:
        merged_vis = load_yaml(vis_cfg_path)
        merged_vis.update(vis_cfg)
        vis_cfg = merged_vis
    trace = TraceLogger(
        output_dir=vis_cfg.get('output_dir', 'artifacts/trace'),
        enabled=bool(vis_cfg.get('enabled', False)),
    )

    image = Image.open(image_path).convert('RGB')
    trace.emit('input/original', image=image, meta={'image': image_path})

    red_cfg = cfg.get('red_ink', {})
    red_enabled = bool(red_cfg.get('enabled', True))
    if red_enabled:
        red_sep = remove_red_ink(
            image,
            use_inpaint=bool(red_cfg.get('use_inpaint', True)),
            inpaint_radius=int(red_cfg.get('inpaint_radius', 3)),
            morph_kernel=int(red_cfg.get('morph_kernel', 3)),
            strategy=red_cfg.get('strategy', 'preserve_text'),
            keep_dark_threshold=int(red_cfg.get('keep_dark_threshold', 150)),
            saturation_min=int(red_cfg.get('saturation_min', 70)),
            value_min=int(red_cfg.get('value_min', 35)),
        )
        page_red_removed = red_sep.text_image
        red_mask_vis = red_sep.marker_mask_image
        trace.emit(
            'preprocess/red_ink_removed',
            image=page_red_removed,
            meta={
                'enabled': True,
                'strategy': red_cfg.get('strategy', 'preserve_text'),
                'use_inpaint': bool(red_cfg.get('use_inpaint', True)),
            },
        )
        trace.emit('preprocess/red_mask', image=red_mask_vis)
        trace.emit('preprocess/marker_layer', image=red_sep.marker_layer_image)
    else:
        page_red_removed = image

    char_cfg = cfg.get('char_visualization', {})
    char_enabled = bool(char_cfg.get('enabled', False))
    if char_enabled:
        char_source = str(char_cfg.get('source', 'red_removed')).lower()
        char_image = page_red_removed if char_source == 'red_removed' else image
        char_result = build_char_debug_result(char_image, char_cfg)
        trace.emit('preprocess/grayscale', image=char_result.gray_image)
        trace.emit('preprocess/binarized', image=char_result.binary_image)
        trace.emit(
            'contour/char_boxes',
            image=char_result.contour_overlay,
            json_obj={
                'count': len(char_result.contour_boxes),
                'boxes_sample': [list(b) for b in char_result.contour_boxes[:200]],
            },
        )
        trace.emit('contour/char_boxes_on_binary', image=char_result.contour_binary_overlay)
        trace.emit('contour/aligned_char_boxes_overlay', image=char_result.aligned_overlay)
        trace.emit(
            'contour/aligned_char_boxes',
            json_obj={
                'count': len(char_result.aligned_boxes),
                'boxes_sample': [list(b) for b in char_result.aligned_boxes[:200]],
            },
        )
        if char_result.char_gallery_image is not None:
            trace.emit('contour/char_gallery', image=char_result.char_gallery_image)
        if char_result.tesseract_overlay is not None:
            trace.emit('ocr/tesseract_char_boxes', image=char_result.tesseract_overlay)
            trace.emit(
                'ocr/tesseract_char_data',
                json_obj={
                    'count': len(char_result.tesseract_chars),
                    'chars_sample': char_result.tesseract_chars[:400],
                },
            )

    yolo_cfg = cfg.get('yolo', {})
    use_dual_model = bool(yolo_cfg.get('use_dual_model', False))

    if use_dual_model:
        doc_opts = build_predict_options(cfg.get('yolo_doc', {}))
        marker_opts = build_predict_options(cfg.get('yolo_marker', {}))
        detections = detect_regions(image_path, doc_opts, source='doc') + detect_regions(
            image_path, marker_opts, source='marker'
        )
        line_thresh = doc_opts.line_thresh
        crop_padding = doc_opts.crop_padding
    else:
        single_opts = build_predict_options(yolo_cfg)
        detections = detect_regions(image_path, single_opts, source='single')
        line_thresh = single_opts.line_thresh
        crop_padding = single_opts.crop_padding

    grouped = group_boxes_by_label(detections)
    marker_boxes, marker_labels = extract_marker_boxes(grouped)

    overlay = draw_boxes(
        image,
        [d.box for d in detections],
        [f'{d.label}:{d.conf:.2f}' for d in detections],
        color='lime',
    )
    trace.emit('detect/combined', image=overlay, meta={'detections': len(detections)})

    text_detection_cfg = cfg.get('text_detection', {})
    plugin_results = detect_text_boxes_with_plugins(page_red_removed, text_detection_cfg)
    plugin_boxes_all: List[Box] = []
    plugin_errors: Dict[str, List[str]] = {}
    for plugin_name, result in plugin_results.items():
        plugin_boxes_all.extend(result.boxes)
        if result.errors:
            plugin_errors[plugin_name] = result.errors
        if result.boxes:
            plugin_overlay = draw_boxes(
                page_red_removed,
                result.boxes,
                labels=[f'{plugin_name}_{i+1}' for i in range(len(result.boxes))],
                color='yellow' if plugin_name == 'paddle' else 'cyan',
            )
            trace.emit(
                f'detect/{plugin_name}_text_boxes',
                image=plugin_overlay,
                json_obj={'count': len(result.boxes), 'errors': result.errors},
            )
        elif result.errors:
            trace.emit(
                f'detect/{plugin_name}_text_boxes',
                json_obj={'count': 0, 'errors': result.errors},
            )

    ocr_cfg = cfg.get('ocr', {})
    ocr_cfg_path = cfg.get('ocr_config')
    if ocr_cfg_path:
        ocr_cfg = load_yaml(ocr_cfg_path)

    ocr_opts = OcrOptions(
        lang=ocr_cfg.get('lang', 'eng'),
        psm=int(ocr_cfg.get('psm', 6)),
        oem=int(ocr_cfg.get('oem', 1)),
        tesseract_cmd=ocr_cfg.get('tesseract_cmd'),
        easyocr_langs=ocr_cfg.get('easyocr_langs'),
        easyocr_gpu=bool(ocr_cfg.get('easyocr_gpu', False)),
        trocr_model=ocr_cfg.get('trocr_model', 'microsoft/trocr-base-handwritten'),
        trocr_device=ocr_cfg.get('trocr_device'),
        paddle_lang=ocr_cfg.get('paddle_lang', 'korean'),
        paddle_use_angle_cls=bool(ocr_cfg.get('paddle_use_angle_cls', True)),
        paddle_use_gpu=bool(ocr_cfg.get('paddle_use_gpu', False)),
    )
    routing_cfg = ocr_cfg.get('routing', {})
    router = OcrRouter(
        options=ocr_opts,
        policy=OcrRoutePolicy(
            default_engine=routing_cfg.get('default_engine', ocr_cfg.get('engine', 'ensemble')),
            printed_engine=routing_cfg.get('printed_engine', 'tesseract'),
            handwritten_engine=routing_cfg.get('handwritten_engine', 'trocr'),
            low_conf_engine=routing_cfg.get('low_conf_engine', 'ensemble'),
            low_conf_threshold=float(routing_cfg.get('low_conf_threshold', 0.45)),
            english_engine=routing_cfg.get('english_engine', 'tesseract_eng'),
            korean_engine=routing_cfg.get('korean_engine', 'tesseract_kor'),
            mixed_engine=routing_cfg.get('mixed_engine', 'tesseract_kor_eng'),
        ),
        ensemble=ocr_cfg.get('ensemble'),
    )

    marker_cfg = cfg.get('marker', {})
    marker_enabled = bool(marker_cfg.get('enabled', True))
    use_masking = bool(marker_cfg.get('use_masking', True))
    use_inpaint = bool(marker_cfg.get('use_inpaint', True))
    dual_ocr = bool(marker_cfg.get('dual_ocr', False))
    inpaint_radius = int(marker_cfg.get('inpaint_radius', 3))

    yolo_text_boxes = select_text_boxes(grouped, detections, line_thresh=line_thresh)
    text_boxes, text_box_source = choose_text_boxes_with_plugin_strategy(
        yolo_text_boxes,
        plugin_boxes_all,
        text_detection_cfg,
        line_thresh=line_thresh,
    )
    trace.emit(
        'detect/text_box_source',
        json_obj={
            'source': text_box_source,
            'yolo_count': len(yolo_text_boxes),
            'plugin_count': len(plugin_boxes_all),
            'plugin_errors': plugin_errors,
        },
    )
    if not text_boxes:
        fallback_cfg = cfg.get('fallback', {})
        text_boxes = build_fallback_text_boxes(image, fallback_cfg)
        if text_boxes:
            trace.emit(
                'detect/fallback_text_boxes',
                json_obj={'count': len(text_boxes), 'mode': fallback_cfg.get('mode', 'columns')},
            )
    text_crops = build_crops(image, text_boxes, crop_padding)
    red_crops = build_crops(page_red_removed, text_boxes, crop_padding) if red_enabled else [None] * len(text_crops)

    preprocess_cfg = cfg.get('ocr_preprocess', {})
    text_segments: List[Tuple[Box, str]] = []
    ocr_records: List[dict] = []

    for idx, (box, crop, red_crop) in enumerate(zip(text_boxes, text_crops, red_crops), start=1):
        masked_crop = None
        if marker_enabled and use_masking and marker_boxes:
            masked_crop = mask_or_inpaint_crop(
                crop,
                box,
                marker_boxes,
                use_inpaint=use_inpaint,
                inpaint_radius=inpaint_radius,
            )
        crop_for_ocr = preprocess_crop_for_ocr(crop, preprocess_cfg)
        masked_for_ocr = preprocess_crop_for_ocr(masked_crop, preprocess_cfg) if masked_crop else None
        red_for_ocr = preprocess_crop_for_ocr(red_crop, preprocess_cfg) if red_crop else None
        result = run_ocr_for_box(
            crop_for_ocr,
            region_label='text_region',
            router=router,
            masked_crop=masked_for_ocr,
            red_clean_crop=red_for_ocr,
            dual_ocr=dual_ocr,
        )
        text_segments.append((box, result.text.strip()))
        ocr_records.append(
            {
                'box_id': idx,
                'engine': result.engine,
                'conf': round(result.conf, 4),
                'latency_ms': round(result.latency_ms, 2),
                'text': result.text.strip(),
                'evidence': result.evidence,
            }
        )

    trace.emit('ocr/text_regions', json_obj=ocr_records, meta={'count': len(ocr_records)})

    question_chunks = build_question_chunks(text_segments, ocr_records)
    if question_chunks:
        trace.emit(
            'parse/question_chunks',
            json_obj=[
                {
                    'question_no': q.question_no,
                    'block_ids': q.block_ids,
                    'confidence_min': q.confidence_min,
                }
                for q in question_chunks
            ],
            meta={'count': len(question_chunks)},
        )

    formula_cfg = cfg.get('formula', {})
    formula_enabled = bool(formula_cfg.get('enabled', True))
    formula_boxes = sort_boxes(grouped.get('formula_region', []), line_thresh=line_thresh)
    raw_text_result = compose_raw_text(question_chunks, text_segments)

    if formula_enabled and formula_boxes:
        composed = compose_with_formula_placeholders(
            text_segments,
            formula_boxes,
            placeholder_prefix=formula_cfg.get('placeholder_prefix', 'FORMULA'),
            line_thresh=line_thresh,
        )
        raw_text = composed.merged_text
        raw_text_mode = f'formula_{raw_text_result.mode}'
        formula_map = {
            token: [float(v) for v in box]
            for token, box in composed.formula_map.items()
        }
        trace.emit('formula/placeholders', json_obj=formula_map, meta={'count': len(formula_map)})
    else:
        raw_text = raw_text_result.selected_text
        raw_text_mode = raw_text_result.mode
        formula_map = {}

    raw_text_debug = build_raw_text_debug(raw_text_result, text_segments, ocr_records, raw_text, selected_mode=raw_text_mode)
    trace.emit(
        'parse/raw_text_debug',
        json_obj={
            'selected_mode': raw_text_debug.get('selected_mode'),
            'coverage_ratio': raw_text_debug.get('coverage_ratio'),
            'uncaptured_block_ids': raw_text_debug.get('uncaptured_block_ids'),
            'detected_question_numbers': raw_text_debug.get('detected_question_numbers'),
            'suspect_boxes': raw_text_debug.get('suspect_boxes'),
        },
        meta={'box_count': raw_text_debug.get('box_count')},
    )

    option_cfg = cfg.get('option', {})
    option_enabled = bool(option_cfg.get('enabled', False))
    option_output = {'options': []}
    if option_enabled:
        option_boxes = sort_boxes(grouped.get('option_block', []), line_thresh=line_thresh)
        option_crops = build_crops(image, option_boxes, crop_padding)
        option_texts = [run_ocr_for_box(crop, 'option_block', router).text.strip() for crop in option_crops]
        option_output['options'] = parse_option_blocks(
            option_boxes,
            option_texts,
            marker_boxes,
            marker_labels,
            priority=option_cfg.get('final_mark_priority', ['X', 'O', 'V', 'CHECK', 'STAR']),
        )

    trace.emit('parse/options', json_obj=option_output, meta={'enabled': option_enabled})

    ollama_cfg = cfg.get('ollama', {})
    ollama_path = ollama_cfg.get('config', 'configs/ollama.yaml')
    llama_cfg = load_yaml(ollama_path)
    host = llama_cfg.get('host', get_default_host())

    if raw_text.strip():
        client = OllamaClient(host)
        llm_input_payload = build_llm_input_payload(
            image_path,
            text_segments,
            ocr_records,
            marker_boxes,
            question_chunks,
        )
        llama_result = run_postprocess(client, raw_text, llama_cfg, input_payload=llm_input_payload)
        corrected = str(llama_result.get('final_output', ''))
        corrected = inject_evidence_and_fill_missing_questions(corrected, question_chunks)

        if llama_result.get('two_stage'):
            trace.emit(
                'llama/fast_preview',
                text=str(llama_result.get('fast_output', '')),
                meta={'model': llama_result.get('fast_model')},
            )
        trace.emit(
            'llama/postprocess',
            text=corrected,
            meta={'model': llama_result.get('final_model')},
        )
    else:
        corrected = ''
        llama_result = {
            'two_stage': False,
            'fast_model': None,
            'fast_output': None,
            'final_model': None,
        }
        trace.emit('llama/skipped', text='OCR 텍스트가 비어 있어 LLM 후처리를 건너뜁니다.')

    final_output = {
        'raw_text': raw_text,
        'raw_text_debug': raw_text_debug,
        'corrected': corrected,
        'fast_preview': llama_result.get('fast_output'),
        'formula_map': formula_map,
        'options': option_output['options'],
        'ocr_records': ocr_records,
        'text_detection': {
            'source': text_box_source,
            'yolo_count': len(yolo_text_boxes),
            'plugin_count': len(plugin_boxes_all),
            'plugin_errors': plugin_errors,
            'selected_count': len(text_boxes),
        },
        'llama': {
            'two_stage': llama_result.get('two_stage'),
            'fast_model': llama_result.get('fast_model'),
            'final_model': llama_result.get('final_model'),
        },
        'question_chunks': [
            {
                'question_no': q.question_no,
                'block_ids': q.block_ids,
                'confidence_min': q.confidence_min,
            }
            for q in question_chunks
        ],
    }

    output_path = cfg.get('output_json')
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

    trace.emit('final/output', json_obj=final_output)
    trace.flush()

    print('=== OCR RAW ===')
    print(raw_text)
    print('\n=== LLAMA CORRECTED ===')
    print(corrected)

    if option_enabled:
        print('\n=== OPTION PARSED ===')
        print(json.dumps(option_output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
