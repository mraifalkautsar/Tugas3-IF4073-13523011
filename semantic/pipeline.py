"""Train a CamVid checkpoint and export held-out results."""

import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .data import (
    CLASS_NAMES, INPUT_SIZE, NUM_CLASSES, VOID, class_histogram, color_mask,
    normalized_image, read_mask, read_rgb, read_splits,
)
from .metrics import confusion, summarize
from .model import ARCHITECTURE, PRETRAINED, build_pretrained, choose_device, load_checkpoint, predict_mask


class TrainingFrames(Dataset):
    def __init__(self, frames, augment=False):
        self.frames = frames
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        image = normalized_image(read_rgb(frame.image))
        labels = read_mask(frame.mask, INPUT_SIZE).astype(np.int64)
        if self.augment and random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            labels = labels[:, ::-1].copy()
        return image, labels


def _atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as file:
        temporary = Path(file.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, data):
    _atomic_write(path, lambda target: target.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    ))


@torch.inference_mode()
def _validate(model, loader, device):
    model.eval()
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for inputs, labels in loader:
        predicted = model(inputs.to(device))["out"].argmax(dim=1).cpu().numpy()
        matrix += confusion(labels.numpy(), predicted)
    return summarize(matrix)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    use_amp = device.type == "cuda"
    frames = read_splits(args.dataset, args.manifest)
    training = TrainingFrames(frames["train"], augment=True)
    validation = TrainingFrames(frames["val"])
    if args.smoke:
        training = Subset(training, range(min(len(training), 2 * args.batch_size)))
        validation = Subset(validation, range(min(len(validation), args.batch_size)))

    loader_args = {
        "batch_size": args.batch_size, "num_workers": args.workers,
        "pin_memory": use_amp, "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        training, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_args
    )
    val_loader = DataLoader(validation, shuffle=False, **loader_args)
    counts = class_histogram(frames["train"], INPUT_SIZE)[:NUM_CLASSES]
    if not counts.any():
        raise ValueError("The training set has no labeled pixels.")
    class_weights = 1 / np.sqrt(np.maximum(counts, 1))
    class_weights /= class_weights.mean()
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device), ignore_index=VOID
    )

    print(f"Training {ARCHITECTURE} on {device} with {len(training)} frames.", flush=True)
    model = build_pretrained().to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": args.learning_rate},
    ], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, history = -1.0, []
    for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, masks in train_loader:
            if not (masks != VOID).any():
                continue
            images = images.to(device, non_blocking=use_amp)
            masks = masks.to(device, non_blocking=use_amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_function(model(images)["out"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            batches += 1

        if not batches:
            raise ValueError("The training batches had no labeled pixels.")
        metrics = _validate(model, val_loader, device)
        loss_value = total_loss / batches
        score = metrics["mean_iou"]
        history.append({"epoch": epoch, "train_loss": loss_value, "validation": metrics})
        if score > best:
            best = score
            checkpoint = {
                "architecture": ARCHITECTURE,
                "classes": CLASS_NAMES,
                "input_size": list(INPUT_SIZE),
                "pretrained": PRETRAINED.name,
                "epoch": epoch,
                "validation_mean_iou": score,
                "state_dict": model.state_dict(),
            }
            _atomic_write(args.output_dir / "best.pt", lambda path: torch.save(checkpoint, path))
        write_json(args.output_dir / "training.json", {
            "epochs": history, "smoke": args.smoke, "batch_size": args.batch_size,
            "seed": args.seed, "device": str(device), "learning_rate": args.learning_rate,
            "backbone_lr": args.backbone_lr, "train_class_pixels": counts.tolist(),
        })
        print(f"Epoch {epoch}: loss={loss_value:.4f}, validation mIoU={score:.4f}", flush=True)
        if args.smoke or epoch == 1:
            load_checkpoint(args.output_dir / "best.pt", torch.device("cpu"))


def evaluate(args):
    device = choose_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    frames = read_splits(args.dataset, args.manifest)["test"]
    panel_positions = set(np.linspace(0, len(frames) - 1, min(6, len(frames)), dtype=int))
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    predictions = args.output_dir / "predictions"
    panels = args.output_dir / "panels"
    predictions.mkdir(parents=True, exist_ok=True)
    panels.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        rgb = read_rgb(frame.image)
        truth = read_mask(frame.mask)
        mask = predict_mask(model, rgb, device)
        matrix += confusion(truth, mask)
        Image.fromarray(mask).save(predictions / f"{frame.id}.png")
        if index in panel_positions:
            width, height = rgb.size
            panel = Image.new("RGB", (3 * width, height))
            for column, image in enumerate((rgb, color_mask(truth), color_mask(mask))):
                panel.paste(image, (column * width, 0))
            panel.save(panels / f"{frame.id}.png")
        if (index + 1) % 25 == 0 or index + 1 == len(frames):
            print(f"Evaluated {index + 1}/{len(frames)} frames.", flush=True)

    metrics = {"test_frames": len(frames), **summarize(matrix)}
    write_json(args.output_dir / "metrics.json", metrics)
    print(f"Test mIoU={metrics['mean_iou']:.4f}, pixel accuracy={metrics['pixel_accuracy']:.4f}")


def predict(args):
    device = choose_device(args.device)
    model = load_checkpoint(args.checkpoint, device)
    mask = predict_mask(model, read_rgb(args.input), device)
    _atomic_write(args.output, lambda path: Image.fromarray(mask).save(path, format="PNG"))
