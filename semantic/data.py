"""Pair original CamVid frames with masks and decode the 32 RGB labels."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SPLIT_FILE = HERE / "camvid_splits.json"
CLASS_SCHEMA = json.loads((HERE / "camvid_classes.json").read_text(encoding="utf-8"))
CLASS_NAMES = [entry["name"] for entry in CLASS_SCHEMA]
NUM_CLASSES = len(CLASS_NAMES)
VOID = 255
OFF_PALETTE = 254
INPUT_SIZE = (240, 320)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class Frame:
    id: str
    image: Path
    mask: Path


def _index(directory, suffixes=()):
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing CamVid directory: {directory}")
    files = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() != ".png":
            continue
        key = path.stem.casefold()
        for suffix in suffixes:
            key = key.removesuffix(suffix)
        if key in files:
            raise ValueError(f"Duplicate CamVid frame ID {key!r} in {directory}")
        files[key] = path
    return files


def read_splits(dataset: Path, manifest: Path = SPLIT_FILE):
    images = _index(dataset / "CamVid_RGB")
    masks = _index(dataset / "CamVid_Label", suffixes=("_l", "_p"))
    splits = json.loads(manifest.read_text(encoding="utf-8"))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("The split manifest must contain train, val, and test lists.")

    seen = set()
    frames = {}
    for split, ids in splits.items():
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Empty or invalid CamVid split: {split}")
        frames[split] = []
        for frame_id in ids:
            key = frame_id.casefold()
            if key in seen:
                raise ValueError(f"CamVid frame {frame_id!r} appears more than once in the manifest.")
            if key not in images or key not in masks:
                raise ValueError(f"Missing RGB frame or original mask for {frame_id!r} in {dataset}")
            seen.add(key)
            frames[split].append(Frame(frame_id, images[key], masks[key]))

    if seen != set(images) or seen != set(masks):
        raise ValueError("CamVid images and masks must match the manifest one-to-one.")
    return frames


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _rgb_code(rgb):
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


@lru_cache(maxsize=1)
def _label_lookup():
    # RGB triples index one lookup table so decoding remains vectorized.
    lookup = np.full(1 << 24, OFF_PALETTE, dtype=np.uint8)
    lookup[0] = VOID
    for class_id, entry in enumerate(CLASS_SCHEMA):
        for color in entry["source"]:
            lookup[_rgb_code(color)] = class_id
    return lookup


def _decode_mask(path, size):
    rgb = read_rgb(path)
    if size is not None:
        rgb = rgb.resize((size[1], size[0]), Image.Resampling.NEAREST)
    channels = np.asarray(rgb, dtype=np.uint32)
    codes = (channels[:, :, 0] << 16) | (channels[:, :, 1] << 8) | channels[:, :, 2]
    return _label_lookup()[codes]


def read_mask(path, size=None):
    mask = _decode_mask(path, size)
    # The source mirror also treats RGB values outside the label palette as void.
    mask[mask == OFF_PALETTE] = VOID
    return mask


def normalized_image(rgb, size=INPUT_SIZE):
    resized = rgb.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    channels = np.asarray(resized, dtype=np.float32) / 255.0
    return ((channels - MEAN) / STD).transpose(2, 0, 1).copy()


def class_histogram(frames, size=None):
    counts = np.zeros(256, dtype=np.int64)
    for frame in frames:
        mask = _decode_mask(frame.mask, size)
        if size is None:
            with Image.open(frame.image) as image:
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Frame and mask dimensions differ for {frame.id}.")
        counts += np.bincount(mask.ravel(), minlength=len(counts))
    return counts


def color_mask(labels):
    palette = np.zeros((VOID + 1, 3), dtype=np.uint8)
    palette[:NUM_CLASSES] = [entry["color"] for entry in CLASS_SCHEMA]
    return Image.fromarray(palette[labels])
