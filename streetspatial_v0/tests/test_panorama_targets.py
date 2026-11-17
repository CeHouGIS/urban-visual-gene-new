"""Target selection and output validation for the full-panorama follow-up."""

import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.download_panoramas import (fetch_legacy_tile, legacy_tile_url, prepare_targets,
                                    stitch_legacy_tiles, valid_panorama, validate_target)


class PanoramaTargetsTest(unittest.TestCase):
    def test_pilot_scenes_are_kept_and_extra_panoids_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            pilot = Path(temporary) / "pilot.jsonl"
            source = Path(temporary) / "source.jsonl"
            pilot.write_text("\n".join(json.dumps(row) for row in [
                {"image_id": "Austin_panoA_0", "scene_id": "Austin_panoA", "transformation": "original"},
                {"image_id": "Austin_panoA_0", "scene_id": "Austin_panoA", "transformation": "mirror"},
            ]) + "\n")
            source.write_text("\n".join(json.dumps(row) for row in [
                {"id": "Austin_panoA_90"}, {"id": "London_panoB_0"}, {"id": "London_panoB_180"},
            ]) + "\n")
            result = prepare_targets(pilot, source, 2, 42)
            self.assertEqual([(r["city"], r["panoid"]) for r in result],
                             [("Austin", "panoA"), ("London", "panoB")])

    def test_rejects_bad_targets_and_non_panoramic_images(self):
        with self.assertRaises(ValueError):
            validate_target({"city": "../Austin", "panoid": "abc"})
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "view.jpg"
            Image.new("RGB", (400, 300), "red").save(image_path)
            self.assertFalse(valid_panorama(image_path))
            self.assertFalse(valid_panorama(image_path, expected_size=(13312, 6656)))

    def test_legacy_tile_grid_is_stitched_in_position(self):
        self.assertIn("/v1/tile?cb_client=maps_sv.tactile&panoid=panoA&zoom=5&x=1&y=0",
                      legacy_tile_url("panoA", 1, 0))

        def tile_bytes(color):
            buffer = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buffer, "PNG")
            return buffer.getvalue()

        def fetch(_panoid, x, _y):
            return tile_bytes("red" if x == 0 else "green")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "panorama.jpg"
            with patch("src.download_panoramas.fetch_legacy_tile", side_effect=fetch):
                stitch_legacy_tiles("panoA", output, tile_workers=2,
                                    columns=2, rows=1, tile_size=8)
            with Image.open(output) as panorama:
                self.assertEqual(panorama.size, (16, 8))
                self.assertGreater(panorama.getpixel((2, 4))[0], 100)
                self.assertGreater(panorama.getpixel((12, 4))[1], 70)

    def test_blank_tile_placeholder_is_rejected_before_stitching(self):
        buffer = io.BytesIO()
        Image.new("L", (512, 512), 0).save(buffer, "JPEG")
        response = type("Response", (), {"content": buffer.getvalue(),
                                         "raise_for_status": lambda self: None})()
        with patch("src.download_panoramas.requests.get", return_value=response):
            with self.assertRaisesRegex(ValueError, "Blank Street View placeholder"):
                fetch_legacy_tile("missing", 0, 0)


if __name__ == "__main__":
    unittest.main()
