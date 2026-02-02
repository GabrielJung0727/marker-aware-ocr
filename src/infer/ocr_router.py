from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .ocr_engine import BaseOcrEngine, OcrOptions, build_ocr_engine


@dataclass
class OcrRoutePolicy:
    default_engine: str = 'ensemble'
    printed_engine: str = 'tesseract'
    handwritten_engine: str = 'trocr'
    low_conf_engine: str = 'ensemble'
    low_conf_threshold: float = 0.45
    english_engine: str = 'tesseract_eng'
    korean_engine: str = 'tesseract_kor'
    mixed_engine: str = 'tesseract_kor_eng'


class OcrRouter:
    def __init__(self, options: OcrOptions, policy: OcrRoutePolicy, ensemble: list[str] | None = None) -> None:
        self.options = options
        self.policy = policy
        self.ensemble = ensemble
        self._cache: Dict[str, BaseOcrEngine] = {}

    def get_engine(self, name: str) -> BaseOcrEngine:
        key = name.lower()
        if key not in self._cache:
            self._cache[key] = build_ocr_engine(key, self.options, ensemble=self.ensemble)
        return self._cache[key]

    def choose_engine_name(self, region_label: str, handwriting_hint: bool = False) -> str:
        label = region_label.lower()
        if handwriting_hint or 'hand' in label or 'answer' in label:
            return self.policy.handwritten_engine
        if 'printed' in label or 'text' in label or 'question' in label:
            return self.policy.printed_engine
        return self.policy.default_engine

    def maybe_retry_low_conf(self, conf: float, current_engine: str) -> Optional[str]:
        if conf >= self.policy.low_conf_threshold:
            return None
        low_conf_engine = self.policy.low_conf_engine.lower()
        if low_conf_engine == current_engine.lower():
            return None
        return low_conf_engine

    def language_retry_engine(self, language_hint: str, current_engine: str) -> Optional[str]:
        hint = (language_hint or '').lower()
        if hint == 'eng':
            target = self.policy.english_engine
        elif hint == 'kor':
            target = self.policy.korean_engine
        elif hint == 'mixed':
            target = self.policy.mixed_engine
        else:
            return None
        if target.lower() == current_engine.lower():
            return None
        return target
