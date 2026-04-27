"""Minimal dataset for the DeepEyes gateway recipe.

Produces ``raw_prompt`` and reward-related fields only.
It does not perform tokenization or vision processing.
"""

from __future__ import annotations

import copy
import base64
import re
from io import BytesIO
import logging

import torch
from PIL import Image

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


class DeepEyesGatewayDataset(RLHFDataset):
    """Thin dataset that leaves prompt encoding and vision extraction to the gateway."""

    def _build_messages(self, example: dict, key: str):
        """Replace media placeholders with OpenAI-style content blocks.

        Tokenization still belongs to the gateway. This method only converts
        source parquet fields like ``"<image>"`` + ``images`` bytes into the
        structured message shape that gateway-side vision extraction expects.
        """
        messages = copy.deepcopy(example[key])
        images = example.get(self.image_key, None) or []
        videos = example.get(self.video_key, None) or []

        image_offset = 0
        video_offset = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [_normalize_content_part(part) for part in content]
                continue
            if not isinstance(content, str) or ("<image>" not in content and "<video>" not in content):
                continue

            content_list = []
            segments = [segment for segment in re.split("(<image>|<video>)", content) if segment]
            for segment in segments:
                if segment == "<image>":
                    if image_offset >= len(images):
                        raise ValueError(f"image placeholder count exceeds images at index {image_offset}")
                    image = _load_image(images[image_offset])
                    content_list.append({"type": "image", "image": _image_to_data_uri(image)})
                    image_offset += 1
                elif segment == "<video>":
                    if video_offset >= len(videos):
                        raise ValueError(f"video placeholder count exceeds videos at index {video_offset}")
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        if image_offset != len(images):
            raise ValueError(f"image placeholder count {image_offset} does not match images count {len(images)}")
        if video_offset != len(videos):
            raise ValueError(f"video placeholder count {video_offset} does not match videos count {len(videos)}")
        return messages

    def maybe_filter_out_long_prompts(self, dataframe=None):
        """Skip base prompt filtering because phase 1 must not tokenize or preprocess vision here."""
        return self.dataframe if dataframe is None else dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        raw_messages = self._build_messages(row_dict, key=self.prompt_key)

        # DeepEyes contract: source data must have [system, user] or [user] shape.
        # We always produce [system, user] output with our fixed system prompt.
        if not raw_messages or not isinstance(raw_messages, list):
            raise ValueError(
                f"Expected non-empty list of messages at index {item}, "
                f"got {type(raw_messages).__name__}: {raw_messages!r}"
            )
        # Extract user content: expect it at index 1 (after system) or index 0 (user-only).
        if len(raw_messages) >= 2 and raw_messages[1].get("role") == "user":
            user_content = raw_messages[1]["content"]
        elif raw_messages[0].get("role") == "user":
            user_content = raw_messages[0]["content"]
        else:
            raise ValueError(
                f"Cannot find user message in raw_messages at index {item}. "
                f"Expected [system, user] or [user] shape, got roles: "
                f"{[m.get('role') for m in raw_messages]}"
            )

        row_dict["raw_prompt"] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. You can call functions to assist "
                    "with the user query. Important: You must call only one function "
                    "at a time."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        if self.negative_prompt_key in row_dict:
            row_dict["raw_negative_prompt"] = self._build_messages(row_dict, key=self.negative_prompt_key)

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        extra_info = row_dict["extra_info"]
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        first_image = _first_image_from_messages(row_dict["raw_prompt"])
        if not tools_kwargs and first_image is not None:
            tools_kwargs = {"image_zoom_in_tool": {"create_kwargs": {"image": first_image}}}
        need_tools_kwargs = extra_info.get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"

        return row_dict


def _load_image(image_data):
    if isinstance(image_data, Image.Image):
        return image_data.convert("RGB")
    if isinstance(image_data, dict):
        if isinstance(image_data.get("image"), Image.Image):
            return image_data["image"].convert("RGB")
        if "bytes" in image_data:
            return Image.open(BytesIO(image_data["bytes"])).convert("RGB")
    raise TypeError(f"image must be dict or PIL.Image, unsupported image type: {type(image_data)}")


def _normalize_content_part(part):
    if not isinstance(part, dict):
        return part
    if part.get("type") in {"image", "image_url"} and "bytes" in part and "image" not in part:
        normalized = dict(part)
        normalized["type"] = "image"
        normalized["image"] = _image_to_data_uri(_load_image(part))
        normalized.pop("bytes", None)
        return normalized
    if part.get("type") in {"image", "image_url"} and isinstance(part.get("image"), Image.Image):
        normalized = dict(part)
        normalized["type"] = "image"
        normalized["image"] = _image_to_data_uri(part["image"].convert("RGB"))
        return normalized
    return part


def _image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_from_data_uri(data_uri: str) -> Image.Image | None:
    if not data_uri.startswith("data:image") or "base64," not in data_uri:
        return None
    _, encoded = data_uri.split("base64,", 1)
    return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")


def _first_image_from_messages(messages):
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image", "image_url"}:
                image = part.get("image")
                if isinstance(image, Image.Image):
                    return image
                if isinstance(image, str):
                    decoded = _image_from_data_uri(image)
                    if decoded is not None:
                        return decoded
    return None
