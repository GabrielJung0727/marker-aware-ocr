from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


@dataclass
class TraceEvent:
    stage_name: str
    timestamp: str
    latency_ms: float | None = None
    meta: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)


class TraceLogger:
    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.events: List[TraceEvent] = []
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'artifacts').mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        stage_name: str,
        *,
        meta: Dict[str, Any] | None = None,
        text: str | None = None,
        json_obj: Any | None = None,
        image: Image.Image | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if not self.enabled:
            return

        artifact_map: Dict[str, str] = {}
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
        safe_stage = stage_name.replace('/', '_').replace(' ', '_')

        if text is not None:
            text_path = self.output_dir / 'artifacts' / f'{stamp}_{safe_stage}.txt'
            text_path.write_text(text, encoding='utf-8')
            artifact_map['text'] = str(text_path)

        if json_obj is not None:
            json_path = self.output_dir / 'artifacts' / f'{stamp}_{safe_stage}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_obj, f, ensure_ascii=False, indent=2)
            artifact_map['json'] = str(json_path)

        if image is not None:
            image_path = self.output_dir / 'artifacts' / f'{stamp}_{safe_stage}.png'
            image.save(image_path)
            artifact_map['image'] = str(image_path)

        event = TraceEvent(
            stage_name=stage_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency_ms,
            meta=meta or {},
            artifacts=artifact_map,
        )
        self.events.append(event)

    def flush(self) -> None:
        if not self.enabled:
            return
        events_path = self.output_dir / 'events.jsonl'
        with open(events_path, 'w', encoding='utf-8') as f:
            for event in self.events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + '\n')
