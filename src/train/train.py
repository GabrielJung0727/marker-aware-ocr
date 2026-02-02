import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
from ultralytics import YOLO


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def normalize_device(device: str | int | None) -> str | int | None:
    if device is None:
        return None
    device_text = str(device).strip().lower()
    if device_text in {'cpu', 'mps', ''}:
        return device
    try:
        import torch
    except Exception:
        print('[WARN] torch를 불러오지 못해 device를 cpu로 강제합니다.')
        return 'cpu'
    if not torch.cuda.is_available():
        print('[WARN] CUDA를 사용할 수 없어 device를 cpu로 변경합니다.')
        return 'cpu'
    return device


def run_train(data_config: str, train_config: str) -> None:
    data_cfg = load_yaml(data_config)
    train_cfg = load_yaml(train_config)

    model_path = train_cfg.get('model', 'yolov8n.pt')
    yolo = YOLO(model_path)

    train_args = dict(train_cfg)
    train_args.pop('model', None)
    train_args['data'] = data_config
    train_args['device'] = normalize_device(train_args.get('device'))

    print('Data config:', data_cfg)
    print('Train config:', train_cfg)
    yolo.train(**train_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    run_train(args.data, args.config)


if __name__ == '__main__':
    main()
