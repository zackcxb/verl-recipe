"""Minimal dataset for the DeepEyes gateway recipe.

Produces ``raw_prompt`` and reward-related fields only.
It does not perform tokenization or vision processing.
"""

from __future__ import annotations

import torch

from verl.utils.dataset.rl_dataset import RLHFDataset


class DeepEyesGatewayDataset(RLHFDataset):
    """Thin dataset that leaves prompt encoding and vision extraction to the gateway."""

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        raw_messages = self._build_messages(row_dict, key=self.prompt_key)
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
                "content": raw_messages[1]["content"] if len(raw_messages) > 1 else raw_messages[0].get("content", ""),
            },
        ]

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        extra_info = row_dict.get("extra_info") or {}
        index = extra_info.get("index", 0)
        tools_kwargs = extra_info.get("tools_kwargs", {})
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs

        return row_dict
