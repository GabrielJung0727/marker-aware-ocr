# YOLO OCR Project

YOLO detection + OCR ensemble + Ollama postprocess pipeline for handwritten/marker-heavy documents.

## Core Flow
1. Detect regions with YOLO (single model or dual model: doc + marker).
2. Apply red-ink cleanup + contour character visualization + marker masking/inpaint.
3. Insert formula placeholders (`[FORMULA_n]`) and parse options/marks.
4. Postprocess with local Ollama (2-stage: fast preview -> final confirm) and export structured JSON.
5. Emit trace events for step-by-step visualization.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Optional advanced detectors: `pip install -r requirements-advanced-ocr.txt`
   - Python 3.11+에서는 CRAFT가 자동으로 제외되고 PaddleOCR 중심으로 동작합니다.
3. Start Ollama: `ollama serve`
4. Update config files:
   - `configs/pipeline.yaml`
   - `configs/ocr.yaml`
   - `configs/ollama.yaml`
5. Run pipeline: `python -m src.infer.pipeline --config configs/pipeline.yaml`

## Scripts
- Train baseline: `scripts/train.ps1`
- Train doc model: `scripts/train_doc.ps1`
- Train marker model: `scripts/train_marker.ps1`
- Run full pipeline: `scripts/pipeline.ps1`
- Llama-only cleanup: `scripts/llama.ps1`
- Basic evaluation: `scripts/eval.ps1`
- Trace viewer: `scripts/viewer.ps1`

## Main Docs
- `docs/dev.md`: sprint plan + execution status
- `docs/dataset_spec.md`: labeling/class policy
- `docs/mark_policy.md`: duplicate mark resolution policy
- `docs/architecture.md`: architecture diagram
- `docs/visualization_spec.md`: trace/viewer spec
- `docs/runbook.md`: setup/run/troubleshooting
