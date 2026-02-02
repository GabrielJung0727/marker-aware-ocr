## 1) System Prompt (고정)

```text
You are an OCR post-processor for Korean school English workbook photos.
Your job is to convert OCR blocks into a clean, structured JSON for downstream parsing.

Rules:
- Output MUST be valid JSON that matches the schema given by the user. Output JSON ONLY.
- Do NOT hallucinate or invent missing text. If unsure, use null or empty string and write a short reason in "notes".
- Preserve original language and punctuation. Do not translate unless requested.
- Keep reading order using the given blocks order. Do not reorder across questions.
- Separate printed text vs handwriting vs teacher marks if the input indicates it.
- For blanks, keep them as "___" (three underscores) in the sentence.
- If multiple choices appear (e.g., (to play) (play)), keep them in options[] exactly as printed.
- If circles/check/X marks are detected, record them in marks without guessing correctness unless explicitly indicated.
```

---

## 2) User Prompt 템플릿 (매 요청마다 입력 데이터만 바꿔서 사용)

> 너희 파이프라인에서 Ollama로 넘기는 payload는 가능하면 아래처럼 보내는 걸 추천해.
> (`blocks`는 “이미 reading order 정렬된 OCR 결과”)

```text
Convert the following OCR result into JSON.

### Target JSON Schema (strict)
{
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
}

### Classification rules
- choose_correct: a sentence with multiple options like "(A) (B)" or a choice list, where one may be circled/checked.
- fill_blank: a sentence with blanks and student filled words, often with hints like [ask, lend].
- reorder_words: words given in boxes/tokens that must be rearranged, often Korean instruction "바르게 배열".
- other: anything that doesn't match above.

### How to extract
- Group blocks into items by detecting question number patterns like "107.", "108.", "113." etc.
- instruction_ko: Korean instruction line for that question (if present).
- prompt_en: the English sentence for that question with blanks preserved as "___".
- options: the bracket/parenthesis options in order.
- tokens: for reorder_words, list the given word tiles/tokens in order.
- student_answer: handwriting answer line(s) near that question.
- teacher_marks.symbols: marks detected (circle/check/x/v/star/underline/strike/highlight).
- teacher_marks.selected_option: if an option text is clearly circled/selected, set it; else null.
- teacher_marks.is_correct: only set true/false if explicitly indicated (e.g., teacher wrote "정답", "correct", or clear grading symbol); otherwise null.
- evidence.block_ids: include all block ids used for that item.
- confidence_min: minimum OCR confidence among used blocks if provided; otherwise null.
- notes: short note for uncertainties, cropped text, overlapping marks, etc.

### Input (ordered OCR blocks)
<INPUT_JSON>
```

---

## 3) Ollama 요청

### `/api/chat` + JSON 강제(권장)

```json
{
  "model": "qwen2.5:7b-instruct",
  "stream": false,
  "format": "json",
  "messages": [
    { "role": "system", "content": "<SYSTEM_PROMPT_HERE>" },
    { "role": "user", "content": "<USER_PROMPT_WITH_INPUT_JSON_HERE>" }
  ],
  "options": { "temperature": 0 }
}
```
