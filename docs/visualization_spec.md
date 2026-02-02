# Visualization Spec

## Goal
Provide object-detection-like visibility for each OCR stage with replayable traces.

## Required Views
1. Original image
2. Red-ink removed image
3. Marker layer split (red-only)
4. Grayscale snapshot
5. Binarized snapshot
6. Contour character boxes (OpenCV)
7. Contour boxes on binarized image
8. Aligned character boxes (x-order by line)
9. Character crop gallery
10. YOLO overlays (doc + marker)
11. Crop gallery (text/formula/option)
12. OCR engine comparison (text/conf/latency/evidence)
13. Reading order numbering
14. Llama input vs output
15. Final JSON

## Trace Event Schema
- `stage_name`: pipeline stage id
- `timestamp`: UTC ISO timestamp
- `latency_ms`: optional stage latency
- `meta`: structured metadata
- `artifacts`: paths to image/text/json

## Event Naming Convention
- `input/original`
- `preprocess/red_ink_removed`
- `preprocess/red_mask`
- `preprocess/marker_layer`
- `preprocess/grayscale`
- `preprocess/binarized`
- `contour/char_boxes`
- `contour/char_boxes_on_binary`
- `contour/aligned_char_boxes_overlay`
- `contour/aligned_char_boxes`
- `contour/char_gallery`
- `detect/combined`
- `detect/paddle_text_boxes`
- `detect/craft_text_boxes`
- `detect/text_box_source`
- `ocr/text_regions`
- `ocr/tesseract_char_boxes`
- `ocr/tesseract_char_data`
- `formula/placeholders`
- `parse/options`
- `llama/postprocess`
- `final/output`

## Viewer
- Entry: `python -m streamlit run src/vis/viewer_streamlit.py`
- Default trace directory: `artifacts/trace`
