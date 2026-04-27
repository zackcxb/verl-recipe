"""Create tiny synthetic parquet files for DeepEyes gateway smoke training."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


COLORS = [
    ("red", (210, 56, 64)),
    ("green", (58, 150, 92)),
    ("blue", (58, 112, 198)),
    ("yellow", (220, 183, 54)),
    ("purple", (137, 82, 174)),
]


def _make_png_bytes(label: str, color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (160, 120), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 150, 110), outline=(255, 255, 255), width=3)
    draw.text((22, 48), label.upper(), fill=(255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_rows(count: int, *, split: str) -> list[dict]:
    rows = []
    for index in range(count):
        color_name, color = COLORS[index % len(COLORS)]
        label = f"{split}-{index:02d}-{color_name}"
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": "Legacy system prompt replaced by recipe dataset."},
                    {
                        "role": "user",
                        "content": (
                            "Look at <image>. The synthetic image contains a color label. "
                            "Answer with the color name in <answer> tags."
                        ),
                    },
                ],
                "images": [{"bytes": _make_png_bytes(label, color)}],
                "data_source": "deepeyes_gateway_smoke",
                "reward_model": {"ground_truth": color_name},
                "extra_info": {"index": index},
            }
        )
    return rows


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    pd.DataFrame(_make_rows(20, split="train")).to_parquet(data_dir / "smoke_train.parquet", index=False)
    pd.DataFrame(_make_rows(8, split="val")).to_parquet(data_dir / "smoke_val.parquet", index=False)


if __name__ == "__main__":
    main()
