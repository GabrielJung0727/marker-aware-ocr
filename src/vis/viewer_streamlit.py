from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


STAGE_KO_LABELS = {
    'input/original': '원본 입력',
    'preprocess/red_ink_removed': '빨간 채점 제거(복원)',
    'preprocess/red_mask': '빨간 채점 마스크',
    'preprocess/marker_layer': '마커 레이어 분리',
    'preprocess/grayscale': '그레이스케일',
    'preprocess/binarized': '이진화',
    'detect/combined': '검출 결과 통합',
    'detect/paddle_text_boxes': 'PaddleOCR 텍스트 박스',
    'detect/craft_text_boxes': 'CRAFT 텍스트 박스',
    'detect/text_box_source': '텍스트 박스 소스 선택',
    'detect/fallback_text_boxes': '대체 텍스트 박스 생성',
    'contour/char_boxes': '윤곽 기반 문자 박스',
    'contour/char_boxes_on_binary': '이진화 이미지 위 문자 박스',
    'contour/aligned_char_boxes_overlay': '정렬된 문자 박스 오버레이',
    'contour/aligned_char_boxes': '문자 박스 정렬 결과',
    'contour/char_gallery': '문자 크롭 갤러리',
    'ocr/text_regions': '텍스트 OCR 결과',
    'ocr/tesseract_char_boxes': '테서랙트 문자 박스 시각화',
    'ocr/tesseract_char_data': '테서랙트 문자 데이터',
    'parse/question_chunks': '문항 번호 분할',
    'parse/raw_text_debug': 'RAW 텍스트 품질/주범 박스 분석',
    'formula/placeholders': '수식 플레이스홀더 생성',
    'parse/options': '보기/마커 파싱',
    'llama/fast_preview': 'COR 1단계(빠른 교정 초안)',
    'llama/postprocess': 'COR 2단계(최종 확정 교정)',
    'final/output': '최종 출력',
}


def load_events(trace_dir: Path):
    events_path = trace_dir / 'events.jsonl'
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def latest_event_by_stage(events: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for event in events:
        stage = str(event.get('stage_name', ''))
        if not stage:
            continue
        latest[stage] = event
    return latest


def render_artifacts(artifacts: dict):
    image_path = artifacts.get('image')
    text_path = artifacts.get('text')
    json_path = artifacts.get('json')

    if image_path and Path(image_path).exists():
        st.image(str(image_path), width='stretch')
    if text_path and Path(text_path).exists():
        st.code(Path(text_path).read_text(encoding='utf-8'))
    if json_path and Path(json_path).exists():
        st.json(json.loads(Path(json_path).read_text(encoding='utf-8')))


def get_stage_display_name(stage_name: str) -> str:
    if stage_name in STAGE_KO_LABELS:
        return f"{STAGE_KO_LABELS[stage_name]} ({stage_name})"
    return f"사용자 단계 ({stage_name})"


def render_usage_guide() -> None:
    st.markdown('### OCR 테스트 방법')
    st.markdown(
        '1. `configs/pipeline.yaml`에서 `image`를 테스트 이미지로 변경합니다.\n'
        '2. 아래 명령으로 OCR 파이프라인을 실행해 trace를 생성합니다.\n'
        '3. 이 뷰어에서 `artifacts/trace`를 열어 단계별 결과를 확인합니다.'
    )
    st.code(
        'python -m src.infer.pipeline --config configs/pipeline.yaml\n'
        'streamlit run src/vis/viewer_streamlit.py',
        language='bash',
    )

    st.markdown('### COR(교정) 단계')
    st.markdown(
        '- **COR 1단계(빠른 교정):** `llama3.2:3b`로 초안 생성 (`llama/fast_preview`)\n'
        '- **COR 2단계(최종 확정):** `qwen2.5:7b-instruct`로 최종 교정 (`llama/postprocess`)'
    )


def render_overview(events: list[dict]) -> None:
    key_stages = [
        'input/original',
        'preprocess/red_ink_removed',
        'preprocess/marker_layer',
        'preprocess/grayscale',
        'preprocess/binarized',
        'contour/char_boxes',
        'contour/char_boxes_on_binary',
        'contour/aligned_char_boxes_overlay',
        'contour/char_gallery',
        'detect/paddle_text_boxes',
        'detect/craft_text_boxes',
        'detect/combined',
        'ocr/text_regions',
        'parse/raw_text_debug',
        'llama/fast_preview',
        'llama/postprocess',
        'final/output',
    ]
    latest = latest_event_by_stage(events)
    st.markdown('### 단계 요약 보기')
    for stage in key_stages:
        event = latest.get(stage)
        if not event:
            continue
        st.markdown(f"**{get_stage_display_name(stage)}**")
        render_artifacts(event.get('artifacts', {}))
        st.caption(f"시간: {event.get('timestamp')}, 지연(ms): {event.get('latency_ms')}")
        st.divider()


def main() -> None:
    if get_script_run_ctx() is None:
        print('이 뷰어는 Streamlit으로 실행해야 합니다.')
        print('실행: streamlit run src/vis/viewer_streamlit.py')
        return

    st.set_page_config(page_title='OCR 트레이스 뷰어', layout='wide')
    st.title('OCR 파이프라인 트레이스 뷰어')
    render_usage_guide()
    st.divider()

    default_dir = Path('artifacts/trace')
    trace_input = st.text_input('트레이스 폴더 경로', value=str(default_dir))
    trace_dir = Path(trace_input)

    events = load_events(trace_dir)
    if not events:
        st.warning('이벤트가 없습니다. 파이프라인에서 trace를 활성화하고 다시 실행하세요.')
        return

    show_overview = st.toggle('핵심 단계 한 번에 보기', value=True)
    if show_overview:
        render_overview(events)

    stage_names = [f"{idx:03d} | {get_stage_display_name(e['stage_name'])}" for idx, e in enumerate(events)]
    selected = st.selectbox('이벤트 선택', stage_names)
    event_idx = int(selected.split('|', 1)[0].strip())
    event = events[event_idx]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(get_stage_display_name(event['stage_name']))
        render_artifacts(event.get('artifacts', {}))

    with col2:
        st.subheader('메타 정보')
        st.json(event.get('meta', {}))
        st.write(f"시간: {event.get('timestamp')}")
        st.write(f"지연(ms): {event.get('latency_ms')}")


if __name__ == '__main__':
    main()
