import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=False, default='data/processed/pipeline_output.json')
    parser.add_argument('--out', required=False, default='data/processed/eval_report.json')
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f'Prediction file not found: {pred_path}')

    data = json.loads(pred_path.read_text(encoding='utf-8'))
    report = {
        'has_raw_text': bool(data.get('raw_text')),
        'has_corrected': bool(data.get('corrected')),
        'option_count': len(data.get('options', [])),
        'formula_count': len(data.get('formula_map', {})),
        'ocr_records': len(data.get('ocr_records', [])),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
