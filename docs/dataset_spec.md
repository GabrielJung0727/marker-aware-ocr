# Dataset Spec

## Class Design (MVP)
- `text_region`: OCR target text area (question/passage/options)
- `formula_region`: mathematical formula area
- `marker`: O/X/V/check/star and similar symbols
- `option_block`: option container for multi-choice parsing

## Extended Classes (Next)
- `question_no`, `question_block`, `passage_block`, `answer_blank`
- `table_region`, `figure_region`

## Marker Labeling Rules
- Annotate each visible marker separately, including overlap cases.
- If marker traces are separable, keep multiple boxes.
- Preserve noisy/correction traces for training robustness.

## Duplicate Mark Policy
- Multiple markers in one option are allowed.
- Final mark is resolved by priority (default: `X > O > V > CHECK > STAR`).
- Keep all raw markers for audit.

## Split Policy
- Split by document id to avoid leakage.
- Recommended ratio: train/val/test = 8/1/1.
- Validation must include camera noise, skew, glare, blur, occlusion.

## QA Checklist
- Option blocks fully covered.
- Marker boxes are tight and complete.
- Formula boxes are separated from normal text.
- Text regions do not clip important characters.
