"""A compact view of fixed real validation examples; no new model inference."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "outputs/v2"
RUN = BASE / "m1_real_streets_cpu_20261109"


def main():
    provenance = json.loads((BASE / "data/real_streets_20261109/provenance.json").read_text())
    records = [json.loads(line) for line in
               (BASE / "data/real_streets_20261109/images.validation.jsonl").read_text().splitlines()]
    by_id = {row["id"]: (i, row) for i, row in enumerate(records)}
    # Retain the first validation sample of each city, as selected before training.
    preview_ids = provenance["preview_ids"][::2]
    zh = ImageFont.truetype(str(BASE / "report_assets/chinese_explainer_subset.ttf"), 27)
    en = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    rng = np.random.default_rng(777)
    palette = rng.integers(35, 240, (32, 3), dtype=np.uint8)
    tile, gap, margin, row_height = 280, 12, 16, 314
    canvas = Image.new("RGB", (3 * tile + 2 * gap + 2 * margin,
                              94 + len(preview_ids) * row_height + 42), "#faf9f6")
    draw = ImageDraw.Draw(canvas)
    for col, (title, sub) in enumerate(zip(
            ("输入图片", "模型分组", "模型实际输出"),
            ("Original photograph", "Encoder grouping", "Decoder grouping"))):
        x = margin + col * (tile + gap)
        draw.text((x, 13), title, font=zh, fill="#202020")
        draw.text((x, 50), sub, font=en, fill="#555555")
    for row_index, identifier in enumerate(preview_ids):
        index, record = by_id[identifier]
        sample = np.load(RUN / f"validation/sample_{index:04d}.npz")
        with Image.open(record["image"]) as opened:
            photo = opened.convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
        images = [photo]
        for key in ("encoder_ids", "decoder_ids"):
            color = Image.fromarray(palette[sample[key]]).resize((tile, tile), Image.Resampling.NEAREST)
            images.append(Image.blend(photo, color, .55))
        y = 94 + row_index * row_height
        draw.text((margin, y - 21), identifier.split("_", 1)[0], font=en, fill="#333333")
        for col, image in enumerate(images):
            canvas.paste(image, (margin + col * (tile + gap), y))
    draw.text((margin, canvas.height - 37), "颜色表示一个候选单元", font=zh, fill="#333333")
    target = RUN / "validation/real_street_result_simple.png"
    canvas.save(target)
    print(target)


if __name__ == "__main__":
    main()
