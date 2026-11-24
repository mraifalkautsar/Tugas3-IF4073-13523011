"""Verify CamVid metadata, label decoding, and evaluation math."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from semantic.data import CLASS_SCHEMA, OFF_PALETTE, SPLIT_FILE, VOID, Frame, class_histogram, read_mask, read_splits
from semantic.metrics import confusion, summarize


class CamVidTests(unittest.TestCase):
    def test_published_manifest(self):
        splits = json.loads(SPLIT_FILE.read_text())
        self.assertEqual({name: len(ids) for name, ids in splits.items()},
                         {"train": 367, "val": 101, "test": 233})
        self.assertEqual(len({image for ids in splits.values() for image in ids}), 701)

    def test_original_colormap_and_nearest_resize(self):
        source, expected = [(0, 0, 0)], [VOID]
        for class_id, entry in enumerate(CLASS_SCHEMA):
            source.extend(tuple(rgb) for rgb in entry["source"])
            expected.extend([class_id] * len(entry["source"]))
        self.assertEqual(len(set(source)), 32)
        rgb = Image.fromarray(np.asarray([source], dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.png"
            rgb.save(path)
            np.testing.assert_array_equal(read_mask(path), [expected])
            expanded = read_mask(path, (2, 64))
            np.testing.assert_array_equal(expanded[0], np.repeat(expected, 2))
            np.testing.assert_array_equal(expanded[0], expanded[1])
            rgb.putpixel((0, 0), (1, 2, 3))
            rgb.save(path)
            np.testing.assert_array_equal(read_mask(path), [expected])
            self.assertEqual(class_histogram([Frame("sample", path, path)])[OFF_PALETTE], 1)

    def test_frame_ids_pair_independently_of_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CamVid_RGB").mkdir()
            (root / "CamVid_Label").mkdir()
            for key in ("b", "C", "a"):
                (root / "CamVid_RGB" / f"{key}.png").touch()
                (root / "CamVid_Label" / f"{key.lower()}_L.png").touch()
            manifest = root / "splits.json"
            manifest.write_text(json.dumps({"train": ["C"], "val": ["a"], "test": ["b"]}))
            frames = read_splits(root, manifest)
            self.assertEqual(frames["train"][0].mask.name, "c_L.png")
            manifest.write_text(json.dumps({"train": ["C"], "val": ["C"], "test": ["b"]}))
            with self.assertRaisesRegex(ValueError, "more than once"):
                read_splits(root, manifest)

    def test_void_excluded_from_semantic_metrics(self):
        truth = np.array([[0, 1, VOID, 1]], dtype=np.uint8)
        predicted = np.array([[0, 0, 10, 1]], dtype=np.uint8)
        metrics = summarize(confusion(truth, predicted))
        self.assertEqual(metrics["pixel_accuracy"], 2/3)
        self.assertEqual(metrics["mean_iou"], 0.5)
        self.assertIsNone(metrics["per_class_iou"]["Bicyclist"])


if __name__ == "__main__":
    unittest.main()
