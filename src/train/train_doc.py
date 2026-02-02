import argparse

from .train import run_train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='configs/data.yaml')
    parser.add_argument('--config', required=False, default='configs/yolo_doc.yaml')
    args = parser.parse_args()

    run_train(args.data, args.config)


if __name__ == '__main__':
    main()
