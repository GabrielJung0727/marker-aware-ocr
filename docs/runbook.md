# Runbook

## 1) Environment
1. Create/activate virtual environment.
2. Install deps: `pip install -r requirements.txt`.
3. (Optional) Install advanced detectors: `pip install -r requirements-advanced-ocr.txt`
4. Install Tesseract and add to PATH if you use `tesseract` route.
5. Start Ollama: `ollama serve`.
6. (Optional) Set dataset root: `$env:DATA_ROOT='F:\data'`.

## 2) Training
- Baseline single model:
  - `python -m src.train.train --data configs/data.yaml --config configs/yolo_ocr.yaml`
- Doc model config template: `configs/yolo_doc.yaml`
- Marker model config template: `configs/yolo_marker.yaml`
- Script shortcuts:
  - `scripts/train_doc.ps1`
  - `scripts/train_marker.ps1`

## 3) Full Pipeline
1. Update `configs/pipeline.yaml` image/model paths.
   - For teacher red pen pages: keep `red_ink.enabled=true`.
   - For generic YOLO fallback: tune `fallback.rows` (e.g. 8~12).
   - For contour/character visualization: set `char_visualization.enabled=true`.
2. Run:
   - `python -m src.infer.pipeline --config configs/pipeline.yaml`
3. Output json (if enabled): `data/processed/pipeline_output.json`

## 4) OCR Routing and Engine
- Main config: `configs/ocr.yaml`
- Key fields:
  - `engine`
  - `ensemble`
  - `routing.default_engine`
  - `routing.printed_engine`
  - `routing.handwritten_engine`
  - `routing.low_conf_threshold`
  - `routing.english_engine`, `routing.korean_engine`, `routing.mixed_engine`

## 4-1) PaddleOCR/CRAFT Text Detector Plugins
- Config section: `configs/pipeline.yaml > text_detection`
- Stage examples in trace:
  - `detect/paddle_text_boxes`
  - `detect/craft_text_boxes`
  - `detect/text_box_source`
- Compatibility note:
  - `craft-text-detector` is unstable on Python 3.11+ due to legacy OpenCV pin.
  - On Python 3.11+, use PaddleOCR plugin first (`plugins.paddle.enabled=true`, `plugins.craft.enabled=false`).

## 5) Prompt Template and Llama Output
- Main config: `configs/ollama.yaml`
- Templates: `configs/prompts/*.yaml`
- JSON strict output:
  - set `output_mode: json`
- Two-stage mode (recommended):
  - `stages.fast.model: llama3.2:3b` (quick preview)
  - `stages.final.model: qwen2.5:7b-instruct` (final confirm)

## 6) Trace Visualization
1. Enable trace in `configs/pipeline.yaml`:
   - `trace.enabled: true`
2. Run pipeline once.
3. Run viewer:
   - `python -m streamlit run src/vis/viewer_streamlit.py`
   - Do not run `python src/vis/viewer_streamlit.py` directly.
4. Default trace directory: `artifacts/trace`

## 7) Evaluation
- Basic report from pipeline output:
  - `python -m src.eval.eval_pipeline --pred data/processed/pipeline_output.json --out data/processed/eval_report.json`

## 8) Troubleshooting
- Empty OCR text: verify YOLO class names include `text_region`.
- If boxes are too large/mixed, increase `fallback.rows` and rerun.
- If red marks damage words, verify `red_ink.enabled=true` and inspect `preprocess/red_ink_removed` in viewer.
- If text is erased under red pen, use `red_ink.strategy=preserve_text` and tune `red_ink.keep_dark_threshold`.
- If contour boxes are too noisy, raise `char_visualization.min_area` or lower `max_area_ratio`.
- Option parse empty: verify class name `option_block` exists.
- Marker occlusion ineffective: set `marker.use_masking=true`, `marker.use_inpaint=true`.
- Slow first run: EasyOCR/TrOCR may download model weights.
