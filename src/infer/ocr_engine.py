from __future__ import annotations

from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence
import os

from PIL import Image
import pytesseract
from pytesseract import TesseractNotFoundError, TesseractError


@dataclass
class OcrOptions:
    lang: str = 'eng'
    psm: int = 6
    oem: int = 1
    tesseract_cmd: str | None = None
    easyocr_langs: List[str] | None = None
    easyocr_gpu: bool = False
    trocr_model: str = 'microsoft/trocr-base-handwritten'
    trocr_device: str | int | None = None
    paddle_lang: str = 'korean'
    paddle_use_angle_cls: bool = True
    paddle_use_gpu: bool = False
    allow_missing_ensemble_engines: bool = True
    min_available_ensemble_engines: int = 1


@dataclass
class OcrResult:
    text: str
    conf: float
    engine: str
    latency_ms: float
    evidence: Dict[str, Any]


class BaseOcrEngine:
    name = 'base'

    def read_image(self, image: Image.Image) -> str:
        return self.read_image_result(image).text

    def read_image_result(self, image: Image.Image) -> OcrResult:
        started = perf_counter()
        text = self._infer_text(image)
        latency_ms = (perf_counter() - started) * 1000.0
        text_score = score_text(text)
        conf = score_to_confidence(text_score)
        return OcrResult(
            text=text.strip(),
            conf=conf,
            engine=self.name,
            latency_ms=latency_ms,
            evidence={'score': text_score},
        )

    def read_many(self, images: Sequence[Image.Image]) -> List[str]:
        return [self.read_image(img) for img in images]

    def read_many_result(self, images: Sequence[Image.Image]) -> List[OcrResult]:
        return [self.read_image_result(img) for img in images]

    def _infer_text(self, image: Image.Image) -> str:
        raise NotImplementedError


class TesseractOcrEngine(BaseOcrEngine):
    name = 'tesseract'

    def __init__(self, options: OcrOptions) -> None:
        self.options = options
        self._configure_tesseract_cmd()

    def _configure_tesseract_cmd(self) -> None:
        cmd = self.options.tesseract_cmd or os.getenv('TESSERACT_CMD')
        if not cmd and os.name == 'nt':
            candidates = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            for c in candidates:
                if Path(c).exists():
                    cmd = c
                    break
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd

    def _infer_text(self, image: Image.Image) -> str:
        config = f"--psm {self.options.psm} --oem {self.options.oem}"
        try:
            return pytesseract.image_to_string(image, lang=self.options.lang, config=config)
        except TesseractNotFoundError as exc:
            raise RuntimeError(
                'Tesseract OCR is not installed. '
                'Install Tesseract executable and add it to PATH.'
            ) from exc
        except TesseractError as exc:
            raise RuntimeError(
                f'Tesseract failed (lang={self.options.lang}). '
                'Check language packs (e.g. kor, eng) and Tesseract installation.'
            ) from exc

    def read_image_result(self, image: Image.Image) -> OcrResult:
        started = perf_counter()
        config = f"--psm {self.options.psm} --oem {self.options.oem}"
        conf_values: List[float] = []
        try:
            text = pytesseract.image_to_string(image, lang=self.options.lang, config=config)
            try:
                data = pytesseract.image_to_data(
                    image,
                    lang=self.options.lang,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                raw_conf = data.get('conf', []) if isinstance(data, dict) else []
                for c in raw_conf:
                    try:
                        value = float(c)
                    except (TypeError, ValueError):
                        continue
                    if value >= 0.0:
                        conf_values.append(value / 100.0)
            except Exception:
                # Keep OCR result even if detailed confidence extraction fails.
                conf_values = []
        except TesseractNotFoundError as exc:
            raise RuntimeError(
                'Tesseract OCR is not installed. '
                'Install Tesseract executable and add it to PATH.'
            ) from exc
        except TesseractError as exc:
            raise RuntimeError(
                f'Tesseract failed (lang={self.options.lang}). '
                'Check language packs (e.g. kor, eng) and Tesseract installation.'
            ) from exc

        latency_ms = (perf_counter() - started) * 1000.0
        text_score = score_text(text)
        heuristic_conf = score_to_confidence(text_score)
        native_conf = (sum(conf_values) / len(conf_values)) if conf_values else None
        conf = merge_confidence(heuristic_conf, native_conf)
        evidence: Dict[str, Any] = {
            'score': text_score,
            'heuristic_conf': round(heuristic_conf, 4),
            'native_conf': round(native_conf, 4) if native_conf is not None else None,
            'native_word_count': len(conf_values),
        }
        return OcrResult(
            text=text.strip(),
            conf=conf,
            engine=self.name,
            latency_ms=latency_ms,
            evidence=evidence,
        )


class EasyOcrEngine(BaseOcrEngine):
    name = 'easyocr'

    def __init__(self, options: OcrOptions) -> None:
        self.options = options
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError('easyocr is not installed. Check requirements.txt.') from exc
        langs = options.easyocr_langs or [options.lang]
        self.reader = easyocr.Reader(langs, gpu=options.easyocr_gpu)

    def _infer_text(self, image: Image.Image) -> str:
        import numpy as np

        img = np.array(image)
        results = self.reader.readtext(img, detail=0, paragraph=True)
        if not results:
            return ''
        if isinstance(results, list):
            return '\n'.join([str(r).strip() for r in results if str(r).strip()])
        return str(results).strip()

    def read_image_result(self, image: Image.Image) -> OcrResult:
        import numpy as np

        started = perf_counter()
        np_img = np.array(image.convert('RGB'))
        result = self.reader.readtext(np_img, detail=1, paragraph=False)

        lines: List[str] = []
        native_values: List[float] = []
        if isinstance(result, list):
            for item in result:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                txt = str(item[1]).strip()
                if txt:
                    lines.append(txt)
                if len(item) >= 3:
                    try:
                        conf = float(item[2])
                        if conf >= 0.0:
                            native_values.append(conf)
                    except (TypeError, ValueError):
                        continue

        if not lines:
            text = self._infer_text(image)
        else:
            text = '\n'.join(lines)

        latency_ms = (perf_counter() - started) * 1000.0
        text_score = score_text(text)
        heuristic_conf = score_to_confidence(text_score)
        native_conf = (sum(native_values) / len(native_values)) if native_values else None
        conf = merge_confidence(heuristic_conf, native_conf)
        return OcrResult(
            text=text.strip(),
            conf=conf,
            engine=self.name,
            latency_ms=latency_ms,
            evidence={
                'score': text_score,
                'heuristic_conf': round(heuristic_conf, 4),
                'native_conf': round(native_conf, 4) if native_conf is not None else None,
                'native_line_count': len(native_values),
            },
        )


class TrOcrEngine(BaseOcrEngine):
    name = 'trocr'

    def __init__(self, options: OcrOptions) -> None:
        self.options = options
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError('transformers is not installed. Check requirements.txt.') from exc

        self.processor = TrOCRProcessor.from_pretrained(options.trocr_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(options.trocr_model)
        if options.trocr_device is not None:
            self.model = self.model.to(options.trocr_device)

    def _infer_text(self, image: Image.Image) -> str:
        import torch

        pixel_values = self.processor(images=image, return_tensors='pt').pixel_values
        if self.options.trocr_device is not None:
            pixel_values = pixel_values.to(self.options.trocr_device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


class PaddleOcrEngine(BaseOcrEngine):
    name = 'paddleocr'

    def __init__(self, options: OcrOptions) -> None:
        self.options = options
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError('paddleocr is not installed. Use requirements-advanced-ocr.txt.') from exc

        import inspect

        init_params = set(inspect.signature(PaddleOCR.__init__).parameters.keys())
        init_kwargs: Dict[str, Any] = {
            'use_angle_cls': options.paddle_use_angle_cls,
            'lang': options.paddle_lang,
        }
        if 'show_log' in init_params:
            init_kwargs['show_log'] = False
        if 'use_gpu' in init_params:
            init_kwargs['use_gpu'] = options.paddle_use_gpu
        if 'device' in init_params and 'use_gpu' not in init_params:
            init_kwargs['device'] = 'gpu' if options.paddle_use_gpu else 'cpu'

        self.reader = PaddleOCR(**init_kwargs)

    def _infer_text(self, image: Image.Image) -> str:
        import numpy as np

        np_img = np.array(image.convert('RGB'))
        try:
            result = self.reader.ocr(np_img, cls=self.options.paddle_use_angle_cls)
        except TypeError:
            result = self.reader.ocr(np_img)
        lines: List[str] = []
        if not isinstance(result, list):
            return ''
        for item in result:
            if not isinstance(item, list):
                continue
            for line in item:
                # line: [box, (text, conf)] for PaddleOCR rec path
                if (
                    isinstance(line, (list, tuple))
                    and len(line) >= 2
                    and isinstance(line[1], (list, tuple))
                    and len(line[1]) >= 1
                ):
                    txt = str(line[1][0]).strip()
                    if txt:
                        lines.append(txt)
        return '\n'.join(lines)

    def read_image_result(self, image: Image.Image) -> OcrResult:
        import numpy as np

        started = perf_counter()
        np_img = np.array(image.convert('RGB'))
        try:
            result = self.reader.ocr(np_img, cls=self.options.paddle_use_angle_cls)
        except TypeError:
            result = self.reader.ocr(np_img)

        lines: List[str] = []
        native_values: List[float] = []
        if isinstance(result, list):
            for item in result:
                if not isinstance(item, list):
                    continue
                for line in item:
                    if (
                        isinstance(line, (list, tuple))
                        and len(line) >= 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) >= 1
                    ):
                        txt = str(line[1][0]).strip()
                        if txt:
                            lines.append(txt)
                        if len(line[1]) >= 2:
                            try:
                                conf = float(line[1][1])
                                if conf >= 0.0:
                                    native_values.append(conf)
                            except (TypeError, ValueError):
                                continue

        text = '\n'.join(lines)
        latency_ms = (perf_counter() - started) * 1000.0
        text_score = score_text(text)
        heuristic_conf = score_to_confidence(text_score)
        native_conf = (sum(native_values) / len(native_values)) if native_values else None
        conf = merge_confidence(heuristic_conf, native_conf)
        return OcrResult(
            text=text.strip(),
            conf=conf,
            engine=self.name,
            latency_ms=latency_ms,
            evidence={
                'score': text_score,
                'heuristic_conf': round(heuristic_conf, 4),
                'native_conf': round(native_conf, 4) if native_conf is not None else None,
                'native_line_count': len(native_values),
            },
        )


class EnsembleOcrEngine(BaseOcrEngine):
    name = 'ensemble'

    def __init__(self, engines: List[BaseOcrEngine]) -> None:
        if not engines:
            raise ValueError('Ensemble requires at least one OCR engine')
        self.engines = engines

    def read_image_result(self, image: Image.Image) -> OcrResult:
        started = perf_counter()
        candidates: List[OcrResult] = []
        errors: List[dict] = []
        for engine in self.engines:
            try:
                candidates.append(engine.read_image_result(image))
            except Exception as exc:  # noqa: BLE001
                errors.append({'engine': engine.name, 'error': str(exc)})
        if not candidates:
            latency_ms = (perf_counter() - started) * 1000.0
            return OcrResult(
                text='',
                conf=0.0,
                engine=self.name,
                latency_ms=latency_ms,
                evidence={'errors': errors},
            )
        chosen = pick_best_result(candidates)
        latency_ms = (perf_counter() - started) * 1000.0
        evidence = {
            'selected': chosen.engine,
            'candidates': [
                {
                    'engine': c.engine,
                    'conf': round(c.conf, 4),
                    'latency_ms': round(c.latency_ms, 2),
                    'score': round(score_text(c.text), 4),
                }
                for c in candidates
            ],
            'errors': errors,
        }
        return OcrResult(
            text=chosen.text,
            conf=chosen.conf,
            engine=self.name,
            latency_ms=latency_ms,
            evidence=evidence,
        )

    def _infer_text(self, image: Image.Image) -> str:
        # Unused because read_image_result is specialized.
        return ''


def score_text(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(ch.isalnum() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    penalty = sum(ch in '@#$%^&*<>' for ch in text)
    length = len(text)
    return (letters + 0.3 * spaces) - 0.5 * penalty + 0.01 * length


def consensus_score(text: str, candidates: Sequence[str]) -> float:
    if not text:
        return 0.0
    agree = 0.0
    for other in candidates:
        if other == text:
            continue
        agree += SequenceMatcher(None, text, other).ratio()
    return agree


def score_to_confidence(score: float) -> float:
    # Simple bounded confidence proxy from heuristic text score.
    conf = (score + 5.0) / 60.0
    return max(0.01, min(0.99, conf))


def merge_confidence(heuristic_conf: float, native_conf: float | None) -> float:
    if native_conf is None:
        return max(0.01, min(0.99, heuristic_conf))
    bounded_native = max(0.0, min(1.0, native_conf))
    # Prefer engine-native confidence while retaining lightweight text-quality prior.
    merged = (0.65 * bounded_native) + (0.35 * heuristic_conf)
    return max(0.01, min(0.99, merged))


def pick_best_text(candidates: Sequence[str]) -> str:
    if not candidates:
        return ''
    return max(candidates, key=lambda t: score_text(t) + consensus_score(t, candidates))


def pick_best_result(candidates: Sequence[OcrResult]) -> OcrResult:
    if not candidates:
        return OcrResult(text='', conf=0.0, engine='none', latency_ms=0.0, evidence={})
    texts = [c.text for c in candidates]
    return max(
        candidates,
        key=lambda c: score_text(c.text) + consensus_score(c.text, texts) + c.conf,
    )


def build_ocr_engine(engine: str, options: OcrOptions, ensemble: List[str] | None = None) -> BaseOcrEngine:
    engine = engine.lower()
    if engine.startswith('tesseract_'):
        # examples: tesseract_eng, tesseract_kor, tesseract_kor_eng
        lang_alias = engine.replace('tesseract_', '', 1).replace('_', '+')
        return TesseractOcrEngine(replace(options, lang=lang_alias))
    if engine == 'tesseract':
        return TesseractOcrEngine(options)
    if engine == 'easyocr':
        return EasyOcrEngine(options)
    if engine == 'trocr':
        return TrOcrEngine(options)
    if engine in {'paddleocr', 'paddle'}:
        return PaddleOcrEngine(options)
    if engine == 'ensemble':
        engines: List[BaseOcrEngine] = []
        errors: List[str] = []
        for name in (ensemble or ['tesseract', 'easyocr']):
            try:
                engines.append(build_ocr_engine(name, options))
            except Exception as exc:  # noqa: BLE001
                if not bool(options.allow_missing_ensemble_engines):
                    raise
                errors.append(f'{name}: {exc}')
        min_required = max(1, int(options.min_available_ensemble_engines))
        if len(engines) < min_required:
            detail = '; '.join(errors) if errors else 'no engines available'
            raise RuntimeError(
                f'Ensemble requires at least {min_required} available engines; got {len(engines)} ({detail})'
            )
        return EnsembleOcrEngine(engines)
    raise ValueError(f'Unknown OCR engine: {engine}')
