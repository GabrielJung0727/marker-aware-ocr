import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .ollama_client import OllamaClient, get_default_host

WORKBOOK_SYSTEM_PROMPT = """You are an OCR post-processor for Korean school English workbook photos.
Your job is to convert OCR blocks into clean, structured JSON for downstream parsing.

Rules:
- Output MUST be valid JSON that matches the schema given by the user. Output JSON ONLY.
- Do NOT hallucinate or invent missing text. If unsure, use null or empty string and write a short reason in notes.
- Preserve original language and punctuation. Do not translate unless requested.
- Keep reading order using the given blocks order. Do not reorder across questions.
- Separate printed text vs handwriting vs teacher marks if indicated in input.
- Keep blanks as "___".
- If circles/check/X marks are detected, record them in teacher_marks without guessing correctness.
- For Korean text, preserve valid Hangul syllables. If OCR splits jamo (e.g., ㄱ ㅏ), recombine only when deterministic.
"""

WORKBOOK_SCHEMA_TEXT = """{
  "doc_type": "english_workbook_page",
  "page_meta": {
    "source": "photo",
    "page_no": integer|null
  },
  "items": [
    {
      "no": integer,
      "type": "choose_correct" | "fill_blank" | "reorder_words" | "other",
      "instruction_ko": string|null,
      "prompt_en": string|null,
      "options": [string],
      "tokens": [string],
      "student_answer": string|null,
      "teacher_marks": {
        "symbols": [ "circle" | "check" | "x" | "v" | "star" | "underline" | "strike" | "highlight" ],
        "selected_option": string|null,
        "is_correct": true|false|null
      },
      "evidence": {
        "block_ids": [string],
        "confidence_min": number|null
      },
      "notes": string|null
    }
  ],
  "notes": string|null
}"""


def load_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_template(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    return load_yaml(path)


def merge_cfg(base: dict, override: dict | None) -> dict:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def merged_prompt_cfg(cfg: dict | None) -> dict:
    cfg = cfg or {}
    template_cfg = load_template(cfg.get('prompt_template'))
    merged = dict(template_cfg)
    merged.update(cfg)
    return merged


def build_basic_prompt(text: str, cfg: dict) -> str:
    domain = cfg.get('domain', 'general OCR')
    style = cfg.get('style', 'concise')
    constraints = cfg.get(
        'constraints',
        [
            'Preserve original meaning',
            'Do not add new facts',
            'Fix only OCR mistakes and spacing',
        ],
    )
    extra_instructions = cfg.get('extra_instructions', '')
    output_mode = cfg.get('output_mode', 'text')

    constraint_block = '\n'.join(f'- {c}' for c in constraints if c)
    prompt = (
        'You are an OCR post-processing specialist.\n'
        f'Domain: {domain}\n'
        f'Style: {style}\n'
        'Rules:\n'
        f'{constraint_block}\n'
    )
    if output_mode == 'json':
        prompt += (
            '\nReturn JSON only with this schema:\n'
            '{"corrected_text":"string","notes":["string"]}\n'
        )
    if extra_instructions:
        prompt += f'\nExtra instructions:\n{extra_instructions}\n'

    prompt += f'\n[OCR]\n{text}\n\n[OUTPUT]\n'
    return prompt


def build_input_payload_from_text(text: str) -> dict:
    blocks: List[dict] = []
    for idx, line in enumerate([ln for ln in text.splitlines() if ln.strip()], start=1):
        blocks.append(
            {
                'id': f'b{idx:03d}',
                'kind': 'printed',
                'role': 'misc',
                'text': line,
                'bbox': None,
                'conf': None,
            }
        )
    return {
        'page_no': None,
        'blocks': blocks,
    }


def build_workbook_user_prompt(input_payload: dict, cfg: dict, fast_output: str | None = None) -> str:
    schema_text = cfg.get('target_schema_text', WORKBOOK_SCHEMA_TEXT)
    classification_rules = cfg.get(
        'classification_rules',
        [
            'choose_correct: sentence with multiple options where one may be circled/checked',
            'fill_blank: sentence with blanks and student-filled words',
            'reorder_words: token boxes that must be rearranged',
            'other: anything else',
        ],
    )
    extraction_rules = cfg.get(
        'extraction_rules',
        [
            'Group blocks by question number patterns like 107., 108., 113.',
            'instruction_ko: Korean instruction line if present',
            'prompt_en: English sentence with blanks as ___',
            'options: parenthesis/bracket options in order',
            'tokens: word tiles for reorder_words',
            'student_answer: handwriting near each item',
            'teacher_marks: record symbols; do not guess correctness',
            'evidence.block_ids: include block ids used for each item',
            'evidence.confidence_min: minimum conf among used blocks if available',
            'For Korean strings, keep Hangul syllables normalized; if uncertain jamo composition exists, keep raw text and add notes',
        ],
    )

    rule_block = '\n'.join(f'- {rule}' for rule in classification_rules)
    extract_block = '\n'.join(f'- {rule}' for rule in extraction_rules)

    expected_nos = input_payload.get('expected_question_nos', []) or []
    expected_text = ', '.join(str(n) for n in expected_nos)

    prompt = (
        'Convert the following OCR result into JSON.\n\n'
        '### Target JSON Schema (strict)\n'
        f'{schema_text}\n\n'
        '### Classification rules\n'
        f'{rule_block}\n\n'
        '### How to extract\n'
        f'{extract_block}\n\n'
    )
    if expected_nos:
        prompt += (
            '### Mandatory coverage\n'
            f'- You MUST include every question number in items.no: [{expected_text}]\n'
            '- If a question is unclear, still include it with type="other" and a short notes field.\n\n'
        )
    if fast_output:
        prompt += (
            '### Fast model draft (reference only)\n'
            f'{fast_output}\n\n'
            'Use the draft only as a hint. Fix errors and output final JSON only.\n\n'
        )

    prompt += (
        '### Input (ordered OCR blocks)\n'
        f"{json.dumps(input_payload, ensure_ascii=False, indent=2)}\n"
    )
    return prompt


def build_messages(
    text: str,
    cfg: dict,
    *,
    input_payload: dict | None = None,
    fast_output: str | None = None,
) -> Tuple[List[Dict[str, str]], dict]:
    merged = merged_prompt_cfg(cfg)
    prompt_mode = str(merged.get('prompt_mode', 'basic')).lower()

    if prompt_mode == 'workbook_json':
        payload = input_payload or build_input_payload_from_text(text)
        system_prompt = merged.get('system_prompt', WORKBOOK_SYSTEM_PROMPT)
        user_prompt = build_workbook_user_prompt(payload, merged, fast_output=fast_output)
        return (
            [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            merged,
        )

    prompt = build_basic_prompt(text, merged)
    if fast_output:
        prompt = (
            f'{prompt}\n'
            'Fast model draft:\n'
            f'{fast_output}\n\n'
            'Use draft only as hint and output final answer.\n'
        )

    system_prompt = merged.get('system_prompt')
    if system_prompt:
        return ([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}], merged)
    return ([{'role': 'user', 'content': prompt}], merged)


def run_generate(
    client: OllamaClient,
    cfg: dict,
    *,
    prompt: str | None = None,
    messages: List[Dict[str, str]] | None = None,
) -> str:
    model = cfg.get('model', 'llama3.1:8b')
    temperature = float(cfg.get('temperature', 0.2))
    max_tokens = int(cfg.get('max_tokens', 256))
    output_mode = cfg.get('output_mode', 'text')
    output_format = 'json' if output_mode == 'json' else None
    use_chat = bool(cfg.get('use_chat', True))

    if messages is not None:
        return client.chat(
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
        )

    if prompt is None:
        prompt = ''

    if use_chat:
        chat_messages = [{'role': 'user', 'content': prompt}]
        return client.chat(
            model,
            chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            output_format=output_format,
        )

    return client.generate(
        model,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        output_format=output_format,
    )


def parse_json_or_retry(raw: str, client: OllamaClient, cfg: dict) -> str:
    try:
        obj = json.loads(raw)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        schema_hint = cfg.get(
            'target_schema_text',
            '{"corrected_text":"string","notes":["string"]}',
        )
        recovery_prompt = (
            'Rewrite the response as strict JSON only. '
            f'Schema: {schema_hint}.\n\n'
            f'Previous response:\n{raw}'
        )
        fixed = run_generate(client, cfg, prompt=recovery_prompt)
        try:
            obj = json.loads(fixed)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            if str(cfg.get('prompt_mode', '')).lower() == 'workbook_json':
                fallback = {
                    'doc_type': 'english_workbook_page',
                    'page_meta': {'source': 'photo', 'page_no': None},
                    'items': [],
                    'notes': f'json_parse_failed: {fixed[:200]}',
                }
            else:
                fallback = {
                    'corrected_text': '',
                    'notes': ['json_parse_failed', fixed[:240]],
                }
            return json.dumps(fallback, ensure_ascii=False, indent=2)


def parse_item_numbers(json_text: str) -> set[int]:
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        return set()
    if not isinstance(obj, dict):
        return set()
    items = obj.get('items', [])
    if not isinstance(items, list):
        return set()
    numbers: set[int] = set()
    for item in items:
        if isinstance(item, dict) and isinstance(item.get('no'), int):
            numbers.add(int(item['no']))
    return numbers


def enforce_expected_question_numbers(
    client: OllamaClient,
    cfg: dict,
    output_json: str,
    expected_nos: list[int],
) -> str:
    if not expected_nos:
        return output_json
    current = parse_item_numbers(output_json)
    missing = [n for n in expected_nos if n not in current]
    if not missing:
        return output_json

    fix_prompt = (
        'Rewrite the JSON to include ALL required question numbers in items.no.\n'
        f'Required numbers: {expected_nos}\n'
        f'Missing numbers: {missing}\n'
        'Do not hallucinate details; use type="other" and short notes when unclear.\n\n'
        f'Current JSON:\n{output_json}'
    )
    fixed_raw = run_generate(client, cfg, prompt=fix_prompt)
    return parse_json_or_retry(fixed_raw, client, cfg)


def finalize_output(raw: str, client: OllamaClient, cfg: dict) -> str:
    if cfg.get('output_mode', 'text') == 'json':
        return parse_json_or_retry(raw, client, cfg)
    return raw


def run_postprocess(client: OllamaClient, text: str, cfg: dict, *, input_payload: dict | None = None) -> dict:
    stages = cfg.get('stages', {})
    use_two_stage = bool(stages.get('enabled', False))

    if not use_two_stage:
        messages, run_cfg = build_messages(text, cfg, input_payload=input_payload)
        raw = run_generate(client, run_cfg, messages=messages)
        final_output = finalize_output(raw, client, run_cfg)
        if str(run_cfg.get('prompt_mode', '')).lower() == 'workbook_json':
            expected_nos = list((input_payload or {}).get('expected_question_nos', []))
            final_output = enforce_expected_question_numbers(client, run_cfg, final_output, expected_nos)
        return {
            'two_stage': False,
            'fast_model': None,
            'fast_output': None,
            'final_model': run_cfg.get('model', 'llama3.1:8b'),
            'final_output': final_output,
        }

    fast_cfg = merge_cfg(cfg, stages.get('fast', {}))
    final_cfg = merge_cfg(cfg, stages.get('final', {}))

    fast_messages, fast_cfg = build_messages(text, fast_cfg, input_payload=input_payload)
    fast_raw = run_generate(client, fast_cfg, messages=fast_messages)
    fast_output = finalize_output(fast_raw, client, fast_cfg)

    final_messages, final_cfg = build_messages(
        text,
        final_cfg,
        input_payload=input_payload,
        fast_output=fast_output,
    )
    final_raw = run_generate(client, final_cfg, messages=final_messages)
    final_output = finalize_output(final_raw, client, final_cfg)
    if str(final_cfg.get('prompt_mode', '')).lower() == 'workbook_json':
        expected_nos = list((input_payload or {}).get('expected_question_nos', []))
        final_output = enforce_expected_question_numbers(client, final_cfg, final_output, expected_nos)

    return {
        'two_stage': True,
        'fast_model': fast_cfg.get('model'),
        'fast_output': fast_output,
        'final_model': final_cfg.get('model'),
        'final_output': final_output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    host = cfg.get('host', get_default_host())

    client = OllamaClient(host)
    result = run_postprocess(client, args.text, cfg)
    if result.get('two_stage'):
        print('=== FAST MODEL DRAFT ===')
        print(result.get('fast_output') or '')
        print('\n=== FINAL CONFIRMED ===')
    print(result.get('final_output') or '')


if __name__ == '__main__':
    main()
