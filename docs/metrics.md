# Metrics Plan

## Detection
- mAP50 / mAP50-95 per class
- Recall for `text_region`, `formula_region`, `marker`, `option_block`

## OCR
- CER / WER (text)
- Edit distance (formula placeholder stream)
- Engine latency (ms) per crop

## Parsing
- Option parse success rate
- Final mark agreement rate
- JSON schema compliance rate
