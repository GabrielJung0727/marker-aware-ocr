# 개발 할 일 정리 (YOLO + OCR + Llama) — 개편/병합본 (2026-02-02)

본 문서는 기존 `dev.md` + 현재 구현 상태를 병합한 실행 보드다.

- 표기: `(O)=완료`, `(△)=부분완료`, `( )=미완료`

---

## 0) 이번 스프린트 산출물

- 마커 중복/가림 대응
- 필기체 OCR 정확도 개선
- YOLO 검출 구조 고도화
- Llama 구조화 후처리
- 단계별 Trace 시각화

---

## 1) 데이터/라벨링

### 1-1. 교육 도메인 태깅 규칙 확정 (△)

- 태그/정책 문서 초안 반영 완료
- 실제 라벨링 샘플셋 기준 검수 라운드 필요

### 1-2. YOLO 라벨 스키마 확정 (O)

- `text_region`, `formula_region`, `marker`, `option_block` 기준 확정
- 반영: `docs/dataset_spec.md`, `configs/data.yaml`

### 1-3. 데이터셋 split/평가 샘플링 확정 (△)

- 문서 단위 split 정책 문서화 완료
- 실제 split 파일 생성/검증은 남음

---

## 2) YOLO 업그레이드

### 2-1. 2모델 전략(문서/마커) 적용 (O)

- `yolo_doc` + `yolo_marker` dual 추론 경로 구현
- 파일: `configs/yolo_doc.yaml`, `configs/yolo_marker.yaml`, `src/infer/pipeline.py`

### 2-2. 촬영/필기/가림 증강 정책 반영 ( )

- 구현 전 (학습 파이프라인 증강 파라미터 설계 필요)

### 2-3. 평가/리포트 자동화 (△)

- 기본 평가 스크립트 추가: `scripts/eval.ps1`, `src/eval/eval_pipeline.py`
- CER/WER/mAP 본평가 스크립트는 후속

---

## 3) 파이프라인 로직

### 3-1. 수식 placeholder 처리 (O)

- 파일: `src/infer/formula_processor.py`
- 파이프라인에서 `[FORMULA_n]` 삽입 지원

### 3-2. 마커 가림(occlusion) 처리 (O)

- 마스킹 + 인페인팅 + dual OCR 비교 지원
- 파일: `src/infer/marker_processor.py`, `src/infer/pipeline.py`

### 3-3. 중복 마크 판정 로직 (O)

- 우선순위 기반 최종 mark 결정
- 파일: `src/infer/mark_resolver.py`, `src/infer/option_parser.py`

### 3-4. 읽기 순서 정렬 (O)

- line clustering 기반 정렬 모듈화
- 파일: `src/infer/reading_order.py`

---

## 4) OCR 엔진 고도화

### 4-1. Efficient Recognition 로그 구조 (O)

- 엔진/latency/conf/evidence 기록 구조 반영
- 파일: `src/infer/ocr_engine.py`, `src/infer/pipeline.py`

### 4-2. OCR Router(printed/handwritten/lowconf) (O)

- 파일: `src/infer/ocr_router.py`, `configs/ocr.yaml`

---

## 5) Llama 후처리

### 5-1. 템플릿 분리 (O)

- 파일: `configs/prompts/exam_option.yaml`, `configs/prompts/handwriting_note.yaml`

### 5-2. JSON schema 강제 + 실패 복구 (O)

- 파일: `src/infer/ocr_llama.py`, `src/infer/ollama_client.py`
- 2단 LLM 반영: `llama3.2:3b`(fast preview) -> `qwen2.5:7b-instruct`(final confirm)

---

## 6) 실시간 시각화

### 6-1. Trace 이벤트/아티팩트 저장 (O)

- 파일: `src/vis/tracer.py`, `configs/vis.yaml`

### 6-2. Viewer/Overlay MVP (O)

- 파일: `src/vis/overlay.py`, `src/vis/viewer_streamlit.py`, `scripts/viewer.ps1`

---

## 7) 문서/다이어그램

### 7-1. 아키텍처 문서화 (O)

- `docs/architecture.md`, `docs/architecture.png` 생성 완료

### 7-2. 운영 가이드 보강 (O)

- `docs/runbook.md`, `docs/visualization_spec.md` 반영

---

# TODO (병합 후 최신 체크리스트)

## 데이터/설정

- [x] `docs/dataset_spec.md` 업데이트
- [x] `configs/ocr.yaml` 분리/확장
- [x] `configs/data.yaml` 실제 split/경로 확정본 반영

## 모델/파이프라인

- [x] `src/infer/formula_processor.py`
- [x] `src/infer/marker_processor.py` (mask/inpaint + dual OCR)
- [x] `src/infer/mark_resolver.py`
- [x] `src/infer/reading_order.py`
- [x] dual model 추론 경로 (`yolo_doc`, `yolo_marker`)
- [x] `train_doc.py`, `train_marker.py` 분리 학습 엔트리

## OCR/LLM

- [x] `src/infer/ocr_router.py`
- [x] `src/infer/ocr_engine.py` scoring/log 확장
- [x] Llama JSON 강제 + retry

## 시각화

- [x] `src/vis/tracer.py`
- [x] `src/vis/overlay.py`
- [x] `src/vis/viewer_streamlit.py`
- [x] `docs/visualization_spec.md`

## 문서

- [x] `docs/architecture.md`
- [x] `docs/architecture.png`
- [x] `docs/runbook.md`

---
