"""Command-line entry point for CamVid data, training, and inference."""

import argparse
import sys
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="verify CamVid pairs and class counts")
    train = commands.add_parser("train", help="fine-tune LR-ASPP on CamVid")
    evaluate = commands.add_parser("evaluate", help="evaluate the held-out CamVid split")
    predict = commands.add_parser("predict", help="predict one image for the MATLAB app")

    for command in (prepare, train, evaluate):
        command.add_argument("--dataset", type=Path, default=Path("data/CamVid"))
        command.add_argument("--manifest", type=Path, default=Path(__file__).with_name("camvid_splits.json"))
    for command in (train, evaluate, predict):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for command in (evaluate, predict):
        command.add_argument("--checkpoint", type=Path, default=Path("artifacts/best.pt"))

    train.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train.add_argument("--epochs", type=positive_int, default=10)
    train.add_argument("--batch-size", type=positive_int, default=4)
    train.add_argument("--workers", type=nonnegative_int, default=2)
    train.add_argument("--seed", type=nonnegative_int, default=0)
    train.add_argument("--learning-rate", type=positive_float, default=3e-4)
    train.add_argument("--backbone-lr", type=positive_float, default=1e-4)
    train.add_argument("--smoke", action="store_true", help="train two batches and reload the checkpoint")
    evaluate.add_argument("--output-dir", type=Path, default=Path("results/camvid"))
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = arguments()
    try:
        if args.command == "prepare":
            from .data import CLASS_NAMES, OFF_PALETTE, VOID, class_histogram, read_splits

            for split, frames in read_splits(args.dataset, args.manifest).items():
                counts = class_histogram(frames)
                print(f"{split}: {len(frames)} frames", flush=True)
                for index, name in enumerate(CLASS_NAMES):
                    print(f"  {name:12} {counts[index]:>12,}")
                print(f"  {'Void':12} {counts[VOID]:>12,}", flush=True)
                if counts[OFF_PALETTE]:
                    print(f"  Off-palette pixels ignored: {counts[OFF_PALETTE]:,}", flush=True)
        else:
            from . import pipeline

            if args.command == "train":
                if args.smoke and args.output_dir.resolve() == Path("artifacts").resolve():
                    args.output_dir = Path("artifacts/smoke")
                pipeline.train(args)
            elif args.command == "evaluate":
                pipeline.evaluate(args)
            else:
                pipeline.predict(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"CamVid {args.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
